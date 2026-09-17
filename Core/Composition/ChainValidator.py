"""Static validation for typed operator pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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
    """Validate explicit slot wiring before a plan reaches the executor."""

    def __init__(self, registry):
        self.registry = registry

    def validate(
        self,
        plan,
        plan_input_kinds: Optional[Set[SlotKind]] = None,
    ) -> ValidationResult:
        """Check that required inputs/outputs are available and type-compatible."""
        from Core.AgentSchema.plan import DynamicToolChainConfig

        errors: List[ValidationError] = []
        warnings: List[str] = []
        available: Dict[str, SlotKind] = {}

        for kind in plan_input_kinds or set():
            available[f"plan_inputs.{kind.value}"] = kind

        if plan.plan_inputs:
            for key in plan.plan_inputs:
                if "query" in key.lower():
                    available[f"plan_inputs.{key}"] = SlotKind.QUERY_TEXT

        for step in plan.steps:
            if not isinstance(step.action, DynamicToolChainConfig):
                continue

            for tool_call in step.action.tools:
                operator = self.registry.get(tool_call.tool_id)
                if operator is None:
                    errors.append(
                        ValidationError(
                            step_id=step.step_id,
                            tool_id=tool_call.tool_id,
                            slot_name="",
                            expected_kind=SlotKind.QUERY_TEXT,
                            message=f"Unknown operator: {tool_call.tool_id}",
                        )
                    )
                    continue

                for slot_spec in operator.input_slots:
                    if not slot_spec.required:
                        continue

                    satisfied = False
                    explicit_source = (
                        tool_call.inputs.get(slot_spec.name)
                        if tool_call.inputs
                        else None
                    )

                    if explicit_source is not None:
                        if isinstance(explicit_source, str) and explicit_source.startswith(
                            "plan_inputs."
                        ):
                            actual_kind = available.get(explicit_source)
                            if actual_kind is None:
                                errors.append(
                                    ValidationError(
                                        step_id=step.step_id,
                                        tool_id=tool_call.tool_id,
                                        slot_name=slot_spec.name,
                                        expected_kind=slot_spec.kind,
                                        message=(
                                            "Unknown or untyped plan input reference: "
                                            f"{explicit_source}; expected {slot_spec.kind}"
                                        ),
                                    )
                                )
                            elif actual_kind != slot_spec.kind:
                                errors.append(
                                    ValidationError(
                                        step_id=step.step_id,
                                        tool_id=tool_call.tool_id,
                                        slot_name=slot_spec.name,
                                        expected_kind=slot_spec.kind,
                                        message=(
                                            f"Type mismatch: {explicit_source} is "
                                            f"{actual_kind}, expected {slot_spec.kind}"
                                        ),
                                    )
                                )
                            satisfied = True

                        elif hasattr(explicit_source, "from_step_id"):
                            ref_key = (
                                f"{explicit_source.from_step_id}."
                                f"{explicit_source.named_output_key}"
                            )
                            actual_kind = available.get(ref_key)
                            if actual_kind is None:
                                errors.append(
                                    ValidationError(
                                        step_id=step.step_id,
                                        tool_id=tool_call.tool_id,
                                        slot_name=slot_spec.name,
                                        expected_kind=slot_spec.kind,
                                        message=(
                                            f"Unknown upstream output reference: {ref_key}; "
                                            f"expected {slot_spec.kind}"
                                        ),
                                    )
                                )
                            elif actual_kind != slot_spec.kind:
                                errors.append(
                                    ValidationError(
                                        step_id=step.step_id,
                                        tool_id=tool_call.tool_id,
                                        slot_name=slot_spec.name,
                                        expected_kind=slot_spec.kind,
                                        message=(
                                            f"Type mismatch: {ref_key} is {actual_kind}, "
                                            f"expected {slot_spec.kind}"
                                        ),
                                    )
                                )
                            satisfied = True

                        else:
                            warnings.append(
                                f"Step {step.step_id}/{tool_call.tool_id}: cannot "
                                f"statically type literal input '{slot_spec.name}'"
                            )
                            satisfied = True

                    elif tool_call.inputs:
                        # Legacy convenience: a differently named input can
                        # satisfy a slot only when its source kind is known and
                        # exactly compatible.
                        for source in tool_call.inputs.values():
                            if isinstance(source, str) and source.startswith(
                                "plan_inputs."
                            ):
                                if available.get(source) == slot_spec.kind:
                                    satisfied = True
                                    break
                            elif hasattr(source, "from_step_id"):
                                ref_key = (
                                    f"{source.from_step_id}."
                                    f"{source.named_output_key}"
                                )
                                if available.get(ref_key) == slot_spec.kind:
                                    satisfied = True
                                    break

                    if not satisfied:
                        kind_available = any(
                            kind == slot_spec.kind for kind in available.values()
                        )
                        if kind_available:
                            satisfied = True
                            warnings.append(
                                f"Step {step.step_id}/{tool_call.tool_id}: input "
                                f"'{slot_spec.name}' ({slot_spec.kind}) is available "
                                "but not explicitly wired"
                            )

                    if not satisfied:
                        errors.append(
                            ValidationError(
                                step_id=step.step_id,
                                tool_id=tool_call.tool_id,
                                slot_name=slot_spec.name,
                                expected_kind=slot_spec.kind,
                                message=(
                                    f"Required input '{slot_spec.name}' "
                                    f"({slot_spec.kind}) not satisfied by any prior step"
                                ),
                            )
                        )

                if tool_call.named_outputs:
                    output_specs = {
                        output_spec.name: output_spec
                        for output_spec in operator.output_slots
                    }
                    for output_name in tool_call.named_outputs:
                        output_spec = output_specs.get(output_name)
                        if output_spec is None:
                            errors.append(
                                ValidationError(
                                    step_id=step.step_id,
                                    tool_id=tool_call.tool_id,
                                    slot_name=output_name,
                                    expected_kind=(
                                        operator.output_slots[0].kind
                                        if operator.output_slots
                                        else SlotKind.QUERY_TEXT
                                    ),
                                    message=(
                                        f"Unknown named output '{output_name}' for "
                                        f"operator '{tool_call.tool_id}'. Available outputs: "
                                        f"{sorted(output_specs)}"
                                    ),
                                )
                            )
                            continue
                        available[
                            f"{step.step_id}.{output_name}"
                        ] = output_spec.kind

        return ValidationResult(
            valid=not errors,
            errors=errors,
            warnings=warnings,
        )

    def suggest_adapters(self, plan) -> List[AdapterSuggestion]:
        """Suggest adapters for known type-mismatch classes."""
        result = self.validate(plan)
        suggestions = []
        for error in result.errors:
            if (
                "Type mismatch" in error.message
                and error.expected_kind == SlotKind.ENTITY_SET
            ):
                suggestions.append(
                    AdapterSuggestion(
                        after_step_id=error.step_id,
                        adapter_id="adapter.entities_to_names",
                        converts_from=SlotKind.ENTITY_SET,
                        converts_to=SlotKind.ENTITY_SET,
                        reason=error.message,
                    )
                )
        return suggestions
