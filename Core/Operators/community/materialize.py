"""Convert selected community reports into chunk-like answer context."""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


async def community_materialize(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    communities = inputs["communities"].data
    chunks = []

    for community in communities or []:
        text = community.report or community.title
        if not text:
            continue
        source_chunk_ids = list(
            dict.fromkeys(
                str(chunk_id)
                for chunk_id in community.extra.get("source_chunk_ids", [])
                if chunk_id
            )
        )
        chunks.append(
            ChunkRecord(
                chunk_id=f"community:{community.community_id}",
                text=text,
                score=float(community.rating or community.occurrence or 0.0),
                extra={
                    "community_id": community.community_id,
                    "level": community.level,
                    "title": community.title,
                    "occurrence": community.occurrence,
                    "rating": community.rating,
                    "source_chunk_ids": source_chunk_ids,
                    "source_nodes": sorted(str(node) for node in community.nodes),
                },
            )
        )

    return {
        "chunks": SlotValue(
            kind=SlotKind.CHUNK_SET,
            data=chunks,
            producer="community.materialize",
        )
    }


def ensure_community_materialize_registered() -> None:
    from Core.Operators.registry import REGISTRY
    from Core.Schema.OperatorDescriptor import CostTier, OperatorDescriptor, SlotSpec

    if REGISTRY.get("community.materialize") is not None:
        return

    REGISTRY.register(
        OperatorDescriptor(
            operator_id="community.materialize",
            display_name="Materialize Community Reports",
            category="community",
            input_slots=[SlotSpec("communities", SlotKind.COMMUNITY_SET)],
            output_slots=[SlotSpec("chunks", SlotKind.CHUNK_SET)],
            cost_tier=CostTier.FREE,
            when_to_use=(
                "Convert selected community reports into answer context while "
                "preserving community identity and source chunk provenance."
            ),
            implementation=community_materialize,
        )
    )
