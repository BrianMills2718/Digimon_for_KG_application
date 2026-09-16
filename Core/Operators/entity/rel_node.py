"""Entity-from-relationships adapter.

For ordinary relationship sets this resolves unique endpoint entities from the
graph. For ToG relationship-agent output it also preserves the relation scoring
metadata needed by ``entity.agent`` for the next-hop candidate selection.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from Core.Common.Utils import truncate_list_by_token_size
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def entity_rel_node(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    rels = inputs["relationships"].data
    if not rels:
        return {
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[],
                producer="entity.rel_node",
            )
        }

    records = []
    ordinary_names = set()

    for relationship in rels:
        relations_dict = relationship.extra.get("relations_dict")
        if relations_dict:
            head = bool(
                relationship.extra.get("head", bool(relationship.src_id))
            )
            topic_entity = (
                relationship.src_id if head else relationship.tgt_id
            )
            if not topic_entity:
                continue
            records.append(
                EntityRecord(
                    entity_name=str(topic_entity),
                    score=relationship.score,
                    extra={
                        "relation": relationship.relation_name,
                        "head": head,
                        "relations_dict": relations_dict,
                    },
                )
            )
            continue

        if relationship.src_id:
            ordinary_names.add(relationship.src_id)
        if relationship.tgt_id:
            ordinary_names.add(relationship.tgt_id)

    if ordinary_names:
        entity_names = list(ordinary_names)
        node_data_list = await asyncio.gather(
            *[ctx.graph.get_node(name) for name in entity_names]
        )
        degrees = await asyncio.gather(
            *[ctx.graph.node_degree(name) for name in entity_names]
        )

        for name, node_data, degree in zip(
            entity_names, node_data_list, degrees
        ):
            if node_data is None:
                continue
            records.append(
                EntityRecord(
                    entity_name=name,
                    source_id=node_data.get("source_id", ""),
                    entity_type=node_data.get("entity_type", ""),
                    description=node_data.get("description", ""),
                    rank=degree or 0,
                )
            )

    if ctx.config and hasattr(ctx.config, "max_token_for_local_context"):
        records = truncate_list_by_token_size(
            records,
            key=lambda item: item.description,
            max_token_size=ctx.config.max_token_for_local_context,
        )

    return {
        "entities": SlotValue(
            kind=SlotKind.ENTITY_SET,
            data=records,
            producer="entity.rel_node",
        )
    }
