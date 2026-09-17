"""Merge retrieved chunk sets while preserving exact evidence provenance.

This utility is used by explicit multi-hop plans that need to retain evidence
from earlier hops without introducing a generic loop accumulator. Duplicate
chunk IDs are collapsed deterministically; the strongest score and all metadata
are preserved.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


def _copy_chunk(record: ChunkRecord) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=str(record.chunk_id),
        text=str(record.text or ""),
        score=record.score,
        extra=dict(record.extra or {}),
    )


async def chunk_merge(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    left = list(inputs["left"].data or [])
    right = list(inputs["right"].data or [])

    merged: dict[str, ChunkRecord] = {}
    order: list[str] = []

    for record in left + right:
        chunk_id = str(record.chunk_id)
        if not chunk_id:
            continue

        if chunk_id not in merged:
            merged[chunk_id] = _copy_chunk(record)
            order.append(chunk_id)
            continue

        current = merged[chunk_id]
        if not current.text and record.text:
            current.text = str(record.text)
        if record.score is not None and (
            current.score is None or float(record.score) > float(current.score)
        ):
            current.score = float(record.score)
        current.extra.update(dict(record.extra or {}))

    return {
        "chunks": SlotValue(
            kind=SlotKind.CHUNK_SET,
            data=[merged[chunk_id] for chunk_id in order],
            producer="chunk.merge",
            metadata={"unique_evidence_chunks": len(order)},
        )
    }


def ensure_chunk_merge_registered() -> None:
    """Register the utility lazily without changing the canonical operator catalog."""
    from Core.Operators.registry import REGISTRY
    from Core.Schema.OperatorDescriptor import CostTier, OperatorDescriptor, SlotSpec

    if REGISTRY.get("chunk.merge") is not None:
        return

    REGISTRY.register(
        OperatorDescriptor(
            operator_id="chunk.merge",
            display_name="Merge Evidence Chunks",
            category="chunk",
            input_slots=[
                SlotSpec("left", SlotKind.CHUNK_SET),
                SlotSpec("right", SlotKind.CHUNK_SET),
            ],
            output_slots=[SlotSpec("chunks", SlotKind.CHUNK_SET)],
            cost_tier=CostTier.FREE,
            when_to_use=(
                "Accumulate exact retrieved evidence across explicit multi-hop "
                "reasoning steps while deduplicating by source chunk ID."
            ),
            implementation=chunk_merge,
        )
    )
