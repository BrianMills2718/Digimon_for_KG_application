"""OperatorComposer — method profiling, plan building, validation, and execution."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from Core.Common.Logger import logger
from Core.Schema.OperatorDescriptor import CostTier


@dataclass
class MethodProfile:
    """Machine-readable profile of a retrieval method."""

    name: str
    description: str
    operator_chain: List[str]
    requires_entity_vdb: bool
    requires_relationship_vdb: bool
    requires_community: bool
    requires_sparse_matrices: bool
    cost_tier: str
    has_loop: bool
    uses_llm_operators: bool
    good_for: str


_COST_ORDER = {
    CostTier.FREE: 0,
    CostTier.CHEAP: 1,
    CostTier.MODERATE: 2,
    CostTier.EXPENSIVE: 3,
}

_METHOD_GUIDANCE = {
    "basic_local": "Simple local retrieval. Best for straightforward factual questions with a well-built entity VDB.",
    "basic_global": "Global community-based retrieval. Best for broad/thematic questions requiring high-level summaries.",
    "lightrag": "Keyword-enriched relationship retrieval. Best when relationships carry rich descriptions and keywords.",
    "fastgraphrag": "PPR-based score propagation. Best for multi-hop questions where graph topology matters.",
    "hipporag": "LLM entity extraction + PPR. Best when query entities aren't in the VDB vocabulary.",
    "tog": "Iterative LLM-guided graph exploration. Best for complex multi-hop reasoning requiring depth.",
    "gr": "PCST subgraph optimization. Best for finding compact, informative subgraphs from dual VDB search.",
    "dalk": "Entity linking + path filtering. Best for questions requiring specific knowledge paths.",
    "kgp": "TF-IDF + iterative neighbor reasoning. Best when entity descriptions are rich text.",
    "med": "Subgraph extraction with Steiner tree. Best for domain-specific connected subgraph queries.",
}


def _to_transport_value(value: Any) -> Any:
    """Recursively convert operator payloads to JSON-friendly primitives.

    Context-only method results must remain machine-readable for the harness.
    Dataclass records are therefore serialized structurally rather than relying
    on ``json.dumps(default=str)`` at the MCP boundary.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _to_transport_value(value.value)
    if hasattr(value, "model_dump"):
        return _to_transport_value(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_transport_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _to_transport_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_transport_value(item) for item in value]
    if isinstance(value, set):
        return [
            _to_transport_value(item)
            for item in sorted(value, key=lambda item: str(item))
        ]

    # NumPy arrays/scalars and similar numeric containers expose tolist/item.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_transport_value(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_transport_value(item())
        except Exception:
            pass

    # Large runtime-only objects (e.g. NetworkX handles) should never make the
    # transport fail. Their useful identifiers belong in record metadata.
    return str(value)


class OperatorComposer:
    """Profiles method plans and executes them through the operator pipeline."""

    def __init__(self, registry):
        self.registry = registry
        self.profiles: Dict[str, MethodProfile] = self._build_profiles()

    def _build_profiles(self) -> Dict[str, MethodProfile]:
        from Core.AgentSchema.plan import DynamicToolChainConfig, LoopConfig
        from Core.Methods import METHOD_PLANS

        profiles = {}
        for name, plan_fn in METHOD_PLANS.items():
            plan = plan_fn(query="<profile_query>")

            operator_chain = []
            has_loop = False
            requires_entity_vdb = False
            requires_relationship_vdb = False
            requires_community = False
            requires_sparse_matrices = False
            uses_llm = False
            max_cost = CostTier.FREE

            for step in plan.steps:
                if isinstance(step.action, DynamicToolChainConfig):
                    for tool_call in step.action.tools:
                        op_id = tool_call.tool_id
                        operator_chain.append(op_id)
                        desc = self.registry.get(op_id)
                        if desc:
                            requires_entity_vdb |= desc.requires_entity_vdb
                            requires_relationship_vdb |= desc.requires_relationship_vdb
                            requires_community |= desc.requires_community
                            requires_sparse_matrices |= desc.requires_sparse_matrices
                            uses_llm |= desc.requires_llm
                            if _COST_ORDER.get(desc.cost_tier, 0) > _COST_ORDER.get(
                                max_cost, 0
                            ):
                                max_cost = desc.cost_tier
                elif isinstance(step.action, LoopConfig):
                    has_loop = True
                    for body_id in step.action.body_step_ids:
                        body_step = next(
                            (
                                candidate
                                for candidate in plan.steps
                                if candidate.step_id == body_id
                            ),
                            None,
                        )
                        if body_step and isinstance(
                            body_step.action, DynamicToolChainConfig
                        ):
                            for tool_call in body_step.action.tools:
                                op_id = tool_call.tool_id
                                if op_id not in operator_chain:
                                    operator_chain.append(op_id)
                                desc = self.registry.get(op_id)
                                if desc:
                                    uses_llm |= desc.requires_llm
                                    if _COST_ORDER.get(
                                        desc.cost_tier, 0
                                    ) > _COST_ORDER.get(max_cost, 0):
                                        max_cost = desc.cost_tier

            profiles[name] = MethodProfile(
                name=name,
                description=plan.plan_description,
                operator_chain=operator_chain,
                requires_entity_vdb=requires_entity_vdb,
                requires_relationship_vdb=requires_relationship_vdb,
                requires_community=requires_community,
                requires_sparse_matrices=requires_sparse_matrices,
                cost_tier=max_cost.value,
                has_loop=has_loop,
                uses_llm_operators=uses_llm,
                good_for=_METHOD_GUIDANCE.get(name, ""),
            )

        return profiles

    def get_method_profiles(self) -> List[MethodProfile]:
        return list(self.profiles.values())

    def get_profile(self, method_name: str) -> Optional[MethodProfile]:
        return self.profiles.get(method_name)

    def build_plan(
        self,
        method_name: str,
        query: str,
        return_context_only: bool = False,
        **kwargs: Any,
    ):
        from Core.AgentSchema.plan import DynamicToolChainConfig
        from Core.Methods import METHOD_PLANS

        if method_name not in METHOD_PLANS:
            raise ValueError(
                f"Unknown method: {method_name}. Available: {sorted(METHOD_PLANS.keys())}"
            )

        plan = METHOD_PLANS[method_name](query=query, **kwargs)
        if return_context_only and plan.steps:
            last_step = plan.steps[-1]
            if isinstance(last_step.action, DynamicToolChainConfig):
                tools = last_step.action.tools
                if tools and tools[-1].tool_id == "meta.generate_answer":
                    plan.steps = plan.steps[:-1]
        return plan

    def validate_plan(self, plan) -> bool:
        from Core.Composition.ChainValidator import ChainValidator
        from Core.Schema.SlotTypes import SlotKind

        result = ChainValidator(self.registry).validate(
            plan,
            plan_input_kinds={SlotKind.QUERY_TEXT},
        )
        if not result.valid:
            for error in result.errors:
                logger.warning(
                    f"Validation error in {error.step_id}/{error.tool_id}: {error.message}"
                )
        for warning in result.warnings:
            logger.debug(f"Validation warning: {warning}")
        return result.valid

    @staticmethod
    def _serialize_slot(slot_val):
        raw_value = slot_val.data if hasattr(slot_val, "data") else slot_val
        return _to_transport_value(raw_value)

    @staticmethod
    def _slot_metadata(slot_val) -> Dict[str, Any]:
        """Return transport-safe slot metadata without changing output shape."""
        if not hasattr(slot_val, "data"):
            return {}
        metadata = dict(getattr(slot_val, "metadata", {}) or {})
        producer = getattr(slot_val, "producer", "")
        kind = getattr(slot_val, "kind", None)
        if producer:
            metadata.setdefault("producer", producer)
        if kind is not None:
            metadata.setdefault("kind", getattr(kind, "value", str(kind)))
        serialized = _to_transport_value(metadata)
        return serialized if isinstance(serialized, dict) else {}

    async def execute(
        self,
        plan,
        ctx,
        fail_fast: bool = True,
        allow_invalid_plan: bool = False,
    ) -> Dict[str, Any]:
        """Validate and execute an operator plan.

        ``all_step_outputs``/``final_output`` preserve their public shape while
        containing only transport-safe structured values. Parallel metadata maps
        carry provenance, evidence IDs, status, producer, and slot-kind data.
        """
        from Core.Composition.PipelineExecutor import (
            PipelineExecutionError,
            PipelineExecutor,
            _FAILED_STEP,
        )

        is_valid = self.validate_plan(plan)
        if not is_valid and not allow_invalid_plan:
            raise PipelineExecutionError(
                "Execution plan failed static validation. "
                "Pass allow_invalid_plan=True only for explicit best-effort debugging."
            )
        if not is_valid:
            logger.warning(
                "Plan has validation errors — explicit best-effort execution requested"
            )

        step_outputs = await PipelineExecutor(self.registry, ctx).execute(
            plan,
            fail_fast=fail_fast,
        )

        result = {
            "all_step_outputs": {},
            "all_step_metadata": {},
            "final_output": {},
            "final_metadata": {},
        }

        for step_id, outputs in step_outputs.items():
            if outputs is _FAILED_STEP:
                result["all_step_outputs"][step_id] = {"__error__": "step failed"}
                result["all_step_metadata"][step_id] = {"__error__": "step failed"}
                continue

            step_data = {}
            step_metadata = {}
            for slot_name, slot_val in outputs.items():
                step_data[slot_name] = self._serialize_slot(slot_val)
                metadata = self._slot_metadata(slot_val)
                if metadata:
                    step_metadata[slot_name] = metadata
            result["all_step_outputs"][step_id] = step_data
            result["all_step_metadata"][step_id] = step_metadata

        if step_outputs:
            last_step_id = list(step_outputs.keys())[-1]
            last_outputs = step_outputs[last_step_id]
            if last_outputs is not _FAILED_STEP:
                for slot_name, slot_val in last_outputs.items():
                    result["final_output"][slot_name] = self._serialize_slot(slot_val)
                    metadata = self._slot_metadata(slot_val)
                    if metadata:
                        result["final_metadata"][slot_name] = metadata

        return result
