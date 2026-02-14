"""Pipeline executor for operator chains.

Executes validated ExecutionPlans by resolving cross-step data flow
and dispatching to operator implementations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue


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

    async def execute(self, plan) -> Dict[str, SlotValue]:
        """Execute all steps in an ExecutionPlan, resolving cross-step data flow.

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
                await self._execute_tool_chain(step, plan)
            elif isinstance(step.action, LoopConfig):
                await self._execute_loop(step, step_map, plan)
            elif isinstance(step.action, ConditionalBranch):
                await self._execute_conditional(step, step_map, plan)
            else:
                logger.warning(f"Skipping step {step.step_id}: unsupported action type {type(step.action)}")

        return self.step_outputs

    async def _execute_tool_chain(self, step, plan) -> None:
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

            # Resolve inputs
            inputs = self._resolve_inputs(tool_call, step.step_id, plan)

            # Execute operator
            try:
                result = await op_desc.implementation(
                    inputs=inputs,
                    ctx=self.ctx,
                    params=tool_call.parameters,
                )
            except Exception as e:
                logger.exception(f"Operator {tool_call.tool_id} failed: {e}")
                result = {}

            # Store named outputs
            if tool_call.named_outputs:
                for out_name, out_desc in tool_call.named_outputs.items():
                    if out_name in result:
                        chain_outputs[out_name] = result[out_name]
                    else:
                        # Try matching by position
                        for key, val in result.items():
                            if key not in chain_outputs:
                                chain_outputs[out_name] = val
                                break
            else:
                chain_outputs.update(result)

        self.step_outputs[step.step_id] = chain_outputs

    async def _execute_loop(self, step, step_map, plan) -> None:
        """Execute a LoopConfig step."""
        loop = step.action
        accumulated = {}

        for iteration in range(loop.max_iterations):
            logger.info(f"Loop {step.step_id}: iteration {iteration + 1}/{loop.max_iterations}")

            for body_step_id in loop.body_step_ids:
                body_step = step_map.get(body_step_id)
                if body_step is None:
                    logger.error(f"Loop body step not found: {body_step_id}")
                    continue
                from Core.AgentSchema.plan import DynamicToolChainConfig
                if isinstance(body_step.action, DynamicToolChainConfig):
                    await self._execute_tool_chain(body_step, plan)

            # Check termination
            if self._evaluate_condition(loop.termination_condition):
                logger.info(f"Loop {step.step_id}: termination condition met at iteration {iteration + 1}")
                break

            # Carry forward outputs
            for output_key in loop.carry_forward_outputs:
                for sid, outputs in self.step_outputs.items():
                    if output_key in outputs:
                        if output_key not in accumulated:
                            accumulated[output_key] = []
                        val = outputs[output_key]
                        if hasattr(val, "data") and isinstance(val.data, list):
                            accumulated[output_key].extend(val.data)

        # Store accumulated outputs
        self.step_outputs[step.step_id] = {
            k: SlotValue(kind=SlotKind.ENTITY_SET, data=v, producer=f"loop.{step.step_id}")
            for k, v in accumulated.items()
        }

    async def _execute_conditional(self, step, step_map, plan) -> None:
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
                await self._execute_tool_chain(sub_step, plan)

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
                # ToolInputSource
                step_out = self.step_outputs.get(source.from_step_id, {})
                slot = step_out.get(source.named_output_key)
                if slot is not None:
                    inputs[input_name] = slot
                else:
                    logger.warning(
                        f"Could not resolve input {input_name} from "
                        f"{source.from_step_id}.{source.named_output_key}"
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
                for slot_name, slot_val in outputs.items():
                    ns[slot_name] = slot_val
                    ns[f"{step_id}.{slot_name}"] = slot_val

            return bool(eval(condition, {"__builtins__": {"len": len}}, ns))
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {condition} -> {e}")
            return False
