"""Entity linking operator.

Link entity mentions to graph entities via VDB top-1 matching.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def entity_link(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": SlotValue(ENTITY_SET)}  -- entities with names to link
    Outputs: {"entities": SlotValue(ENTITY_SET)}   -- linked entities from graph
    """
    seed = inputs["entities"].data  # List[EntityRecord]
    if not seed:
        return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=[], producer="entity.link")}

    queries = [r.entity_name for r in seed]
    results = await asyncio.gather(
        *[ctx.entities_vdb.retrieval_nodes(q, top_k=1, graph=ctx.graph) for q in queries]
    )

    records = []
    for q, res in zip(queries, results):
        if not res or not res[0]:
            continue
        nd = res[0]
        name = nd.get(ctx.graph.entity_metakey, nd.get("entity_name", q))
        records.append(EntityRecord(
            entity_name=str(name),
            source_id=nd.get("source_id", ""),
            entity_type=nd.get("entity_type", ""),
            description=nd.get("description", ""),
            extra={"linked_from": q},
        ))

    return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=records, producer="entity.link")}
