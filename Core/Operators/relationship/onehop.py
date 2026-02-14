"""Relationship one-hop operator.

Find all relationships connected to a set of entities.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from Core.Common.Utils import truncate_list_by_token_size
from Core.Schema.SlotTypes import RelationshipRecord, SlotKind, SlotValue


async def relationship_onehop(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": SlotValue(ENTITY_SET)}
    Outputs: {"relationships": SlotValue(RELATIONSHIP_SET)}
    """
    entities = inputs["entities"].data  # List[EntityRecord]
    if not entities:
        return {"relationships": SlotValue(kind=SlotKind.RELATIONSHIP_SET, data=[], producer="relationship.onehop")}

    names = [e.entity_name for e in entities]
    all_edge_lists = await asyncio.gather(
        *[ctx.graph.get_node_edges(n) for n in names]
    )

    all_edges = set()
    for edge_list in all_edge_lists:
        if edge_list:
            all_edges.update(tuple(sorted(e)) for e in edge_list)
    all_edges = list(all_edges)

    edge_data_list = await asyncio.gather(
        *[ctx.graph.get_edge(e[0], e[1]) for e in all_edges]
    )
    edge_degrees = await asyncio.gather(
        *[ctx.graph.edge_degree(e[0], e[1]) for e in all_edges]
    )

    records = []
    for (src, tgt), data, deg in zip(all_edges, edge_data_list, edge_degrees):
        if data is None:
            continue
        records.append(RelationshipRecord(
            src_id=src,
            tgt_id=tgt,
            relation_name=data.get("relation_name", ""),
            description=data.get("description", ""),
            weight=data.get("weight", 0.0),
            keywords=data.get("keywords", ""),
            source_id=data.get("source_id", ""),
            extra={"rank": deg or 0},
        ))

    records.sort(key=lambda x: (x.extra.get("rank", 0), x.weight), reverse=True)

    if ctx.config and hasattr(ctx.config, "max_token_for_local_context"):
        records = truncate_list_by_token_size(
            records,
            key=lambda x: x.description,
            max_token_size=ctx.config.max_token_for_local_context,
        )

    return {"relationships": SlotValue(kind=SlotKind.RELATIONSHIP_SET, data=records, producer="relationship.onehop")}
