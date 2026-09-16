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
                "preserving community identifiers and scores."
            ),
            implementation=community_materialize,
        )
    )
