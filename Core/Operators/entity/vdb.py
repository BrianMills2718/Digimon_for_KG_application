"""Entity VDB search operator.

Find entities semantically similar to a query via vector database.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def entity_vdb(
    inputs: Dict[str, SlotValue],
    ctx: Any,  # OperatorContext
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": SlotValue(QUERY_TEXT)}
    Outputs: {"entities": SlotValue(ENTITY_SET)}
    Params:  {"top_k": int, "tree_node": bool}
    """
    query = inputs["query"].data
    p = params or {}
    top_k = p.get("top_k", ctx.config.top_k)
    tree_node = p.get("tree_node", False)

    raw = await ctx.entities_vdb.retrieval_nodes(
        query=query, top_k=top_k, graph=ctx.graph,
        tree_node=tree_node, need_score=True,
    )

    if isinstance(raw, tuple) and len(raw) == 2:
        nodes_data, scores = raw
    else:
        nodes_data, scores = raw, None

    if not nodes_data:
        return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=[], producer="entity.vdb")}

    records = []
    for i, nd in enumerate(nodes_data):
        if nd is None:
            continue

        if tree_node:
            eid = nd.get("id", nd.get("index", i))
            text = nd.get("text", "")
        else:
            eid = nd.get(ctx.graph.entity_metakey, f"missing_id_{i}")
            text = nd.get("content", "")

        vdb_score = scores[i] if scores and i < len(scores) and scores[i] is not None else None

        rec = EntityRecord(
            entity_name=str(eid),
            source_id=nd.get("source_id", ""),
            entity_type=nd.get("entity_type", ""),
            description=nd.get("description", text),
            score=float(vdb_score) if vdb_score is not None else None,
            extra={"layer": nd.get("layer", 0), "text": text},
        )

        # Attach clusters if community available
        if ctx.community and hasattr(ctx.community, "community_node_map") and ctx.community.community_node_map:
            try:
                cluster_info = await ctx.community.community_node_map.get_by_id(eid)
                if cluster_info:
                    rec.clusters = cluster_info
            except Exception as e:
                logger.warning(f"Failed to fetch clusters for '{eid}': {e}")

        records.append(rec)

    # Add rank (node degree)
    if records:
        names = [r.entity_name for r in records]
        try:
            degrees = await asyncio.gather(*[ctx.graph.node_degree(n) for n in names])
            for rec, deg in zip(records, degrees):
                rec.rank = deg or 0
        except Exception as e:
            logger.error(f"Error calculating node degrees: {e}")

    return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=records, producer="entity.vdb")}
