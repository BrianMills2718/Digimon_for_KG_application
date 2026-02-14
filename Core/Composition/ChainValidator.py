"""Chain validation for operator pipelines.

Validates that every step's inputs are satisfied by prior outputs or plan_inputs.
Can suggest adapter insertions for type mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind


@dataclass
class ValidationError:
    step_id: str
    tool_id: str
    slot_name: str
    expected_kind: SlotKind
    message: str


@dataclass
class AdapterSuggestion:
    after_step_id: str
    adapter_id: str
    converts_from: SlotKind
    converts_to: SlotKind
    reason: str


@dataclass
class ValidationResult:
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ChainValidator:
    def __init__(self, registry):
        self.registry = registry

    def validate(self, plan, plan_input_kinds: Optional[Set[SlotKind]] = None) -> ValidationResult:
        """Check every step's inputs are satisfied by prior outputs or plan_inputs.

        Args:
            plan: ExecutionPlan
            plan_input_kinds: Set of SlotKinds available as plan inputs (e.g., {QUERY_TEXT})
        """
        from Core.AgentSchema.plan import DynamicToolChainConfig

        errors = []
        warnings = []
        available: Dict[str, SlotKind] = {}  # "step_id.slot_name" -> kind

        # Plan inputs are always available as QUERY_TEXT
        if plan_input_kinds:
            for kind in plan_input_kinds:
                available[f"plan_inputs.{kind.value}"] = kind

        # Also mark standard plan_inputs
        if plan.plan_inputs:
            for key in plan.plan_inputs:
                if "query" in key.lower():
                    available[f"plan_inputs.{key}"] = SlotKind.QUERY_TEXT

        for step in plan.steps:
            if not isinstance(step.action, DynamicToolChainConfig):
                continue

            for tool_call in step.action.tools:
                op = self.registry.get(tool_call.tool_id)
                if op is None:
                    errors.append(ValidationError(
                        step_id=step.step_id,
                        tool_id=tool_call.tool_id,
                        slot_name="",
                        expected_kind=SlotKind.QUERY_TEXT,
                        message=f"Unknown operator: {tool_call.tool_id}",
                    ))
                    continue

                # Check required inputs are available
                for slot_spec in op.input_slots:
                    if not slot_spec.required:
                        continue

                    satisfied = False

                    if tool_call.inputs:
                        # Find the input wired to this slot by matching input_key to slot name
                        source = tool_call.inputs.get(slot_spec.name)
                        if source is None:
                            # Also check all inputs for any that provide the right kind
                            for input_key, src in tool_call.inputs.items():
                                if isinstance(src, str) and src.startswith("plan_inputs."):
                                    if slot_spec.kind == SlotKind.QUERY_TEXT:
                                        satisfied = True
                                        break
                                elif hasattr(src, "from_step_id"):
                                    ref_key = f"{src.from_step_id}.{src.named_output_key}"
                                    if ref_key in available and available[ref_key] == slot_spec.kind:
                                        satisfied = True
                                        break
                        elif isinstance(source, str) and source.startswith("plan_inputs."):
                            satisfied = True
                        elif hasattr(source, "from_step_id"):
                            ref_key = f"{source.from_step_id}.{source.named_output_key}"
                            if ref_key in available:
                                if available[ref_key] == slot_spec.kind:
                                    satisfied = True
                                else:
                                    errors.append(ValidationError(
                                        step_id=step.step_id,
                                        tool_id=tool_call.tool_id,
                                        slot_name=slot_spec.name,
                                        expected_kind=slot_spec.kind,
                                        message=f"Type mismatch: {ref_key} is {available[ref_key]}, expected {slot_spec.kind}",
                                    ))
                                    satisfied = True

                    # Fallback: check if the kind is available from any prior step
                    if not satisfied:
                        kind_available = any(v == slot_spec.kind for v in available.values())
                        if kind_available:
                            satisfied = True
                            warnings.append(
                                f"Step {step.step_id}/{tool_call.tool_id}: input '{slot_spec.name}' "
                                f"({slot_spec.kind}) available but not explicitly wired"
                            )

                    if not satisfied:
                        errors.append(ValidationError(
                            step_id=step.step_id,
                            tool_id=tool_call.tool_id,
                            slot_name=slot_spec.name,
                            expected_kind=slot_spec.kind,
                            message=f"Required input '{slot_spec.name}' ({slot_spec.kind}) not satisfied by any prior step",
                        ))

                # Register outputs
                if tool_call.named_outputs:
                    for out_name in tool_call.named_outputs:
                        # Find matching output slot
                        for out_spec in op.output_slots:
                            if out_spec.name == out_name or out_name in out_spec.name:
                                available[f"{step.step_id}.{out_name}"] = out_spec.kind
                                break
                        else:
                            # Default: register with first output kind
                            if op.output_slots:
                                available[f"{step.step_id}.{out_name}"] = op.output_slots[0].kind

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def suggest_adapters(self, plan) -> List[AdapterSuggestion]:
        """Suggest adapter insertions for type mismatches."""
        result = self.validate(plan)
        suggestions = []

        for error in result.errors:
            if "Type mismatch" in error.message:
                # Suggest an adapter
                if error.expected_kind == SlotKind.ENTITY_SET:
                    suggestions.append(AdapterSuggestion(
                        after_step_id=error.step_id,
                        adapter_id="adapter.entities_to_names",
                        converts_from=SlotKind.ENTITY_SET,
                        converts_to=SlotKind.ENTITY_SET,
                        reason=error.message,
                    ))

        return suggestions
