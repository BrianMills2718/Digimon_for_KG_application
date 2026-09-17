"""Merge relationship sets while preserving source provenance.

Used by explicit multi-hop plans such as ToG to accumulate selected graph
relations across hops before materializing source chunks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Utils import split_string_by_multi_markers
from Core.Schema.SlotTypes import RelationshipRecord, SlotKind, SlotValue


def _merge_source_ids(*values: str) -> str:
    ids = []
    for value in values:
        if not value:
            continue
        for chunk_id in split_string_by_multi_markers(
            str(value), [GRAPH_FIELD_SEP]
        ):
            if chunk_id and chunk_id not in ids:
                ids.append(chunk_id)
    return GRAPH_FIELD_SEP.join(ids)


async def relationship_merge(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    left = list(inputs["left"].data or [])
    right = list(inputs["right"].data or [])

    merged: dict[tuple[str, str, str], RelationshipRecord] = {}
    order = []

    for record in left + right:
        key = (
            str(record.src_id),
            str(record.tgt_id),
            str(record.relation_name),
        )
        if key not in merged:
            merged[key] = RelationshipRecord(
                src_id=record.src_id,
                tgt_id=record.tgt_id,
                relation_name=record.relation_name,
                description=record.description,
                weight=record.weight,
                keywords=record.keywords,
                source_id=record.source_id,
                score=record.score,
                extra=dict(record.extra or {}),
            )
            order.append(key)
            continue

        current = merged[key]
        current.source_id = _merge_source_ids(
            current.source_id,
            record.source_id,
        )
        if record.score is not None and (
            current.score is None or record.score > current.score
        ):
            current.score = record.score
        current.weight = max(
            float(current.weight or 0.0),
            float(record.weight or 0.0),
        )
        if not current.description and record.description:
            current.description = record.description
        if not current.keywords and record.keywords:
            current.keywords = record.keywords

    return {
        "relationships": SlotValue(
            kind=SlotKind.RELATIONSHIP_SET,
            data=[merged[key] for key in order],
            producer="relationship.merge",
        )
    }


def ensure_relationship_merge_registered() -> None:
    from Core.Operators.registry import REGISTRY
    from Core.Schema.OperatorDescriptor import CostTier, OperatorDescriptor, SlotSpec

    if REGISTRY.get("relationship.merge") is not None:
        return

    REGISTRY.register(
        OperatorDescriptor(
            operator_id="relationship.merge",
            display_name="Merge Relationship Sets",
            category="relationship",
            input_slots=[
                SlotSpec("left", SlotKind.RELATIONSHIP_SET),
                SlotSpec("right", SlotKind.RELATIONSHIP_SET),
            ],
            output_slots=[
                SlotSpec("relationships", SlotKind.RELATIONSHIP_SET)
            ],
            cost_tier=CostTier.FREE,
            when_to_use=(
                "Accumulate selected relationship evidence across explicit "
                "multi-hop reasoning steps while preserving source chunk IDs."
            ),
            implementation=relationship_merge,
        )
    )
