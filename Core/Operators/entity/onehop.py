"""Entity one-hop neighbors operator.

Expand an entity set by finding their immediate graph neighbors.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def entity_onehop(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": SlotValue(ENTITY_SET)}
    Outputs: {"entities": SlotValue(ENTITY_SET)}
    """
    seed = inputs["entities"].data  # List[EntityRecord]
    if not seed:
        return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=[], producer="entity.onehop")}

    names = [r.entity_name for r in seed]
    neighbor_lists = await asyncio.gather(
        *[ctx.graph.get_neighbors(n) for n in names]
    )

    seen = set(names)
    all_neighbors = []
    for neighbors in neighbor_lists:
        for n in (neighbors or []):
            if n not in seen:
                seen.add(n)
                all_neighbors.append(n)

    # Fetch node data for neighbors
    node_data_list = await asyncio.gather(
        *[ctx.graph.get_node(n) for n in all_neighbors]
    )
    degrees = await asyncio.gather(
        *[ctx.graph.node_degree(n) for n in all_neighbors]
    )

    records = []
    for name, nd, deg in zip(all_neighbors, node_data_list, degrees):
        if nd is None:
            continue
        records.append(EntityRecord(
            entity_name=name,
            source_id=nd.get("source_id", ""),
            entity_type=nd.get("entity_type", ""),
            description=nd.get("description", ""),
            rank=deg or 0,
        ))

    return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=records, producer="entity.onehop")}
