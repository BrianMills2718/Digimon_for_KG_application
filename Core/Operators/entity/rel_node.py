"""Entity from relationships operator.

Extract entity nodes from a set of relationships (their src/tgt endpoints).
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
    """
    Inputs:  {"relationships": SlotValue(RELATIONSHIP_SET)}
    Outputs: {"entities": SlotValue(ENTITY_SET)}
    """
    rels = inputs["relationships"].data  # List[RelationshipRecord]
    if not rels:
        return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=[], producer="entity.rel_node")}

    entity_names = set()
    for r in rels:
        entity_names.add(r.src_id)
        entity_names.add(r.tgt_id)

    entity_names = list(entity_names)
    node_data_list = await asyncio.gather(
        *[ctx.graph.get_node(n) for n in entity_names]
    )
    degrees = await asyncio.gather(
        *[ctx.graph.node_degree(n) for n in entity_names]
    )

    records = []
    for name, nd, deg in zip(entity_names, node_data_list, degrees):
        if nd is None:
            continue
        records.append(EntityRecord(
            entity_name=name,
            source_id=nd.get("source_id", ""),
            entity_type=nd.get("entity_type", ""),
            description=nd.get("description", ""),
            rank=deg or 0,
        ))

    # Truncate by token size if config available
    if ctx.config and hasattr(ctx.config, "max_token_for_local_context"):
        records = truncate_list_by_token_size(
            records,
            key=lambda x: x.description,
            max_token_size=ctx.config.max_token_for_local_context,
        )

    return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=records, producer="entity.rel_node")}
