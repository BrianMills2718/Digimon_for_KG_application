"""Pipeline executor for operator chains.

Executes validated ExecutionPlans by resolving cross-step data flow
and dispatching to operator implementations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue


# Sentinel to mark a failed step (distinct from empty dict)
_FAILED_STEP = "__FAILED__"


class PipelineExecutionError(Exception):
    """Raised when a pipeline step fails and fail_fast is True."""
    pass


class PipelineExecutor:
    def __init__(self, registry, ctx):
        """
        Args:
            registry: OperatorRegistry with all operators registered
            ctx: OperatorContext with graph, VDB, LLM, etc.
        """
        self.registry = registry
        self.ctx = ctx
        self.step_outputs: Dict[str, Dict[str, SlotValue]] = {}

    async def execute(self, plan, fail_fast: bool = True) -> Dict[str, SlotValue]:
        """Execute all steps in an ExecutionPlan, resolving cross-step data flow.

        Args:
            plan: ExecutionPlan to execute
            fail_fast: If True (default), stop on first operator failure
                      instead of continuing with broken state.

        Returns the outputs of the final step (or all accumulated outputs).
        """
        from Core.AgentSchema.plan import (
            ConditionalBranch,
            DynamicToolChainConfig,
            LoopConfig,
        )

        # Build step lookup
        step_map = {step.step_id: step for step in plan.steps}

        for step in plan.steps:
            if isinstance(step.action, DynamicToolChainConfig):
                await self._execute_tool_chain(step, plan, fail_fast=fail_fast)
            elif isinstance(step.action, LoopConfig):
                await self._execute_loop(step, step_map, plan, fail_fast=fail_fast)
            elif isinstance(step.action, ConditionalBranch):
                await self._execute_conditional(step, step_map, plan, fail_fast=fail_fast)
            else:
                logger.warning(f"Skipping step {step.step_id}: unsupported action type {type(step.action)}")

        return self.step_outputs

    async def _execute_tool_chain(self, step, plan, fail_fast: bool = True) -> None:
        """Execute a DynamicToolChainConfig step."""
        chain_outputs = {}

        for tool_call in step.action.tools:
            op_desc = self.registry.get(tool_call.tool_id)
            if op_desc is None:
                logger.error(f"Unknown operator: {tool_call.tool_id}")
                continue
            if op_desc.implementation is None:
                logger.error(f"Operator {tool_call.tool_id} has no implementation")
                continue

            # Make in-progress chain_outputs visible for intra-step references
            # (e.g., second tool in a chain referencing first tool's output)
            self.step_outputs[step.step_id] = chain_outputs

            # Resolve inputs (will raise if a required input references a failed step)
            inputs = self._resolve_inputs(tool_call, step.step_id, plan)

            # Pre-dispatch validation: check slot names and types match the operator descriptor
            self._validate_inputs_against_descriptor(op_desc, inputs, tool_call, step.step_id)

            # Execute operator
            try:
                result = await op_desc.implementation(
                    inputs=inputs,
                    ctx=self.ctx,
                    params=tool_call.parameters,
                )
            except Exception as e:
                logger.exception(f"Operator {tool_call.tool_id} failed in step {step.step_id}: {e}")
                if fail_fast:
                    raise PipelineExecutionError(
                        f"Operator '{tool_call.tool_id}' failed in step '{step.step_id}': {e}"
                    ) from e
                # Mark this step as failed so downstream steps get a clear error
                chain_outputs = _FAILED_STEP
                break

            # Store named outputs
            if tool_call.named_outputs:
                for out_name, out_desc in tool_call.named_outputs.items():
                    if out_name in result:
                        chain_outputs[out_name] = result[out_name]
                    else:
                        logger.warning(
                            f"Operator {tool_call.tool_id} did not produce expected output '{out_name}'. "
                            f"Available keys: {list(result.keys())}. Skipping."
                        )
            else:
                chain_outputs.update(result)

        self.step_outputs[step.step_id] = chain_outputs

    async def _execute_loop(self, step, step_map, plan, fail_fast: bool = True) -> None:
        """Execute a LoopConfig while preserving carried slot contracts."""
        loop = step.action
        accumulated_data: Dict[str, list] = {}
        accumulated_kinds: Dict[str, SlotKind] = {}
        accumulated_metadata: Dict[str, Dict[str, Any]] = {}

        for iteration in range(loop.max_iterations):
            logger.info(
                f"Loop {step.step_id}: iteration {iteration + 1}/{loop.max_iterations}"
            )

            for body_step_id in loop.body_step_ids:
                body_step = step_map.get(body_step_id)
                if body_step is None:
                    logger.error(f"Loop body step not found: {body_step_id}")
                    continue
                from Core.AgentSchema.plan import DynamicToolChainConfig
                if isinstance(body_step.action, DynamicToolChainConfig):
                    await self._execute_tool_chain(
                        body_step,
                        plan,
                        fail_fast=fail_fast,
                    )

            # Carry forward the outputs produced by this iteration before testing
            # termination so a terminating iteration is not silently discarded.
            for output_key in loop.carry_forward_outputs:
                for body_step_id in loop.body_step_ids:
                    outputs = self.step_outputs.get(body_step_id)
                    if not outputs or outputs is _FAILED_STEP:
                        continue
                    slot = outputs.get(output_key)
                    if slot is None:
                        continue

                    previous_kind = accumulated_kinds.get(output_key)
                    if previous_kind is not None and previous_kind != slot.kind:
                        raise PipelineExecutionError(
                            f"Loop '{step.step_id}' carry-forward output '{output_key}' "
                            f"changed kind from '{previous_kind.value}' to '{slot.kind.value}'."
                        )
                    accumulated_kinds[output_key] = slot.kind

                    if output_key not in accumulated_data:
                        accumulated_data[output_key] = []
                    if isinstance(slot.data, list):
                        accumulated_data[output_key].extend(slot.data)
                    elif slot.data is not None:
                        accumulated_data[output_key].append(slot.data)

                    metadata = accumulated_metadata.setdefault(output_key, {})
                    metadata.update(dict(slot.metadata or {}))

            if self._evaluate_condition(loop.termination_condition):
                logger.info(
                    f"Loop {step.step_id}: termination condition met at iteration "
                    f"{iteration + 1}"
                )
                break

        self.step_outputs[step.step_id] = {
            output_key: SlotValue(
                kind=accumulated_kinds[output_key],
                data=data,
                producer=f"loop.{step.step_id}",
                metadata={
                    **accumulated_metadata.get(output_key, {}),
                    "iterations_accumulated": loop.max_iterations,
                },
            )
            for output_key, data in accumulated_data.items()
        }

    async def _execute_conditional(self, step, step_map, plan, fail_fast: bool = True) -> None:
        """Execute a ConditionalBranch step."""
        branch = step.action
        condition_met = self._evaluate_condition(branch.condition)

        steps_to_run = branch.if_true_steps if condition_met else branch.if_false_steps
        for sub_step_id in steps_to_run:
            sub_step = step_map.get(sub_step_id)
            if sub_step is None:
                logger.error(f"Conditional sub-step not found: {sub_step_id}")
                continue
            from Core.AgentSchema.plan import DynamicToolChainConfig
            if isinstance(sub_step.action, DynamicToolChainConfig):
                await self._execute_tool_chain(sub_step, plan, fail_fast=fail_fast)

    def _validate_inputs_against_descriptor(
        self, op_desc, inputs: Dict[str, SlotValue], tool_call, step_id: str
    ) -> None:
        """Validate resolved inputs match operator descriptor before dispatch.

        Checks two things:
        1. Slot names: every resolved input key must be a known input slot name
        2. Slot types: the SlotKind of each resolved input must match the descriptor
        3. Required slots: every required input slot must be present

        Raises PipelineExecutionError with actionable messages for agents.
        """
        expected_slots = {s.name: s for s in op_desc.input_slots}
        resolved_names = set(inputs.keys())
        expected_names = set(expected_slots.keys())

        # Check for unknown input names (likely slot name typo)
        unknown = resolved_names - expected_names
        if unknown:
            raise PipelineExecutionError(
                f"Slot name mismatch in step '{step_id}', operator '{op_desc.operator_id}': "
                f"plan provides input(s) {sorted(unknown)} but operator expects {sorted(expected_names)}. "
                f"Fix the input key names in the plan to match the operator descriptor."
            )

        # Check required slots are present
        for slot_spec in op_desc.input_slots:
            if slot_spec.required and slot_spec.name not in resolved_names:
                raise PipelineExecutionError(
                    f"Missing required input in step '{step_id}', operator '{op_desc.operator_id}': "
                    f"slot '{slot_spec.name}' ({slot_spec.kind.value}) is required but not provided. "
                    f"Available inputs: {sorted(resolved_names)}."
                )

        # Check type compatibility
        for input_name, slot_val in inputs.items():
            if input_name in expected_slots:
                expected_kind = expected_slots[input_name].kind
                if slot_val.kind != expected_kind:
                    raise PipelineExecutionError(
                        f"Type mismatch in step '{step_id}', operator '{op_desc.operator_id}': "
                        f"input '{input_name}' has kind '{slot_val.kind.value}' but operator "
                        f"expects '{expected_kind.value}'. Insert a compatible operator between "
                        f"the source and this operator."
                    )

    def _resolve_inputs(
        self, tool_call, current_step_id: str, plan
    ) -> Dict[str, SlotValue]:
        """Resolve tool call inputs from plan_inputs and prior step outputs."""
        inputs = {}

        if not tool_call.inputs:
            return inputs

        for input_name, source in tool_call.inputs.items():
            if isinstance(source, str):
                if source.startswith("plan_inputs."):
                    key = source.split(".", 1)[1]
                    value = (plan.plan_inputs or {}).get(key, "")
                    inputs[input_name] = SlotValue(
                        kind=SlotKind.QUERY_TEXT,
                        data=value,
                        producer="plan_inputs",
                    )
                else:
                    # Literal string value
                    inputs[input_name] = SlotValue(
                        kind=SlotKind.QUERY_TEXT,
                        data=source,
                        producer="literal",
                    )
            elif hasattr(source, "from_step_id"):
                # ToolInputSource — check if the referenced step failed
                step_out = self.step_outputs.get(source.from_step_id)

                if step_out is _FAILED_STEP:
                    raise PipelineExecutionError(
                        f"Cannot resolve input '{input_name}' for operator "
                        f"'{tool_call.tool_id}' in step '{current_step_id}': "
                        f"upstream step '{source.from_step_id}' failed."
                    )

                if step_out is None:
                    step_out = {}

                slot = step_out.get(source.named_output_key)
                if slot is not None:
                    inputs[input_name] = slot
                else:
                    raise PipelineExecutionError(
                        f"Cannot resolve input '{input_name}' for operator "
                        f"'{tool_call.tool_id}' in step '{current_step_id}': "
                        f"step '{source.from_step_id}' has no output '{source.named_output_key}'. "
                        f"Available outputs: {list(step_out.keys())}"
                    )
            else:
                # Literal value
                inputs[input_name] = SlotValue(
                    kind=SlotKind.QUERY_TEXT,
                    data=source,
                    producer="literal",
                )

        return inputs

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a simple condition string against step outputs.

        Supports basic checks like:
        - "entities.data == []"
        - "len(entities.data) > 0"
        """
        try:
            # Build a simple namespace from step outputs
            ns = {}
            for step_id, outputs in self.step_outputs.items():
                if outputs is _FAILED_STEP:
                    continue
                for slot_name, slot_val in outputs.items():
                    ns[slot_name] = slot_val
                    ns[f"{step_id}.{slot_name}"] = slot_val

            return bool(eval(condition, {"__builtins__": {"len": len}}, ns))
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {condition} -> {e}")
            return False
