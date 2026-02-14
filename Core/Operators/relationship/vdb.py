"""Relationship VDB search operator.

Find relationships semantically similar to a query via vector database.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Common.Utils import truncate_list_by_token_size
from Core.Schema.SlotTypes import RelationshipRecord, SlotKind, SlotValue


async def relationship_vdb(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": SlotValue(QUERY_TEXT)}
    Outputs: {"relationships": SlotValue(RELATIONSHIP_SET)}
    Params:  {"top_k": int}
    """
    query = inputs["query"].data
    p = params or {}
    top_k = p.get("top_k", ctx.config.top_k)

    try:
        edge_datas = await ctx.relations_vdb.retrieval_edges(
            query=query, top_k=top_k, graph=ctx.graph, need_score=True,
        )
        if not edge_datas:
            return {"relationships": SlotValue(kind=SlotKind.RELATIONSHIP_SET, data=[], producer="relationship.vdb")}

        # Build relationship context with degree ranking
        if not all(e is not None for e in edge_datas):
            logger.warning("Some edges are missing from VDB results")

        edge_degrees = await asyncio.gather(
            *[ctx.graph.edge_degree(r["src_id"], r["tgt_id"]) for r in edge_datas if r]
        )

        records = []
        for ed, deg in zip([e for e in edge_datas if e], edge_degrees):
            records.append(RelationshipRecord(
                src_id=ed["src_id"],
                tgt_id=ed["tgt_id"],
                relation_name=ed.get("relation_name", ""),
                description=ed.get("description", ""),
                weight=ed.get("weight", 0.0),
                keywords=ed.get("keywords", ""),
                source_id=ed.get("source_id", ""),
                extra={"rank": deg or 0},
            ))

        records.sort(key=lambda x: (x.extra.get("rank", 0), x.weight), reverse=True)

        if ctx.config and hasattr(ctx.config, "max_token_for_global_context"):
            records = truncate_list_by_token_size(
                records,
                key=lambda x: x.description,
                max_token_size=ctx.config.max_token_for_global_context,
            )

        return {"relationships": SlotValue(kind=SlotKind.RELATIONSHIP_SET, data=records, producer="relationship.vdb")}

    except Exception as e:
        logger.exception(f"relationship_vdb failed: {e}")
        return {"relationships": SlotValue(kind=SlotKind.RELATIONSHIP_SET, data=[], producer="relationship.vdb")}
