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
        raw = await ctx.relations_vdb.retrieval_edges(
            query=query,
            top_k=top_k,
            graph=ctx.graph,
            need_score=True,
        )
        if not raw:
            return {
                "relationships": SlotValue(
                    kind=SlotKind.RELATIONSHIP_SET,
                    data=[],
                    producer="relationship.vdb",
                )
            }

        if isinstance(raw, tuple) and len(raw) == 2:
            edge_datas, scores = raw
        else:
            edge_datas, scores = raw, None

        valid_edges = [edge for edge in edge_datas if edge is not None]
        if not valid_edges:
            return {
                "relationships": SlotValue(
                    kind=SlotKind.RELATIONSHIP_SET,
                    data=[],
                    producer="relationship.vdb",
                )
            }
        if len(valid_edges) != len(edge_datas):
            logger.warning("Some edges are missing from relationship VDB results")

        edge_degrees = await asyncio.gather(
            *[ctx.graph.edge_degree(edge["src_id"], edge["tgt_id"]) for edge in valid_edges]
        )

        records = []
        valid_index = 0
        for result_index, edge in enumerate(edge_datas):
            if edge is None:
                continue
            score = None
            if scores and result_index < len(scores) and scores[result_index] is not None:
                score = float(scores[result_index])
            degree = edge_degrees[valid_index]
            valid_index += 1

            records.append(
                RelationshipRecord(
                    src_id=edge["src_id"],
                    tgt_id=edge["tgt_id"],
                    relation_name=edge.get("relation_name", ""),
                    description=edge.get("description", ""),
                    weight=edge.get("weight", 0.0),
                    keywords=edge.get("keywords", ""),
                    source_id=edge.get("source_id", ""),
                    score=score,
                    extra={"rank": degree or 0},
                )
            )

        records.sort(
            key=lambda record: (
                record.score if record.score is not None else float("-inf"),
                record.extra.get("rank", 0),
                record.weight,
            ),
            reverse=True,
        )

        if ctx.config and hasattr(ctx.config, "max_token_for_global_context"):
            records = truncate_list_by_token_size(
                records,
                key=lambda record: record.description,
                max_token_size=ctx.config.max_token_for_global_context,
            )

        return {
            "relationships": SlotValue(
                kind=SlotKind.RELATIONSHIP_SET,
                data=records,
                producer="relationship.vdb",
            )
        }

    except Exception as e:
        logger.exception(f"relationship_vdb failed: {e}")
        return {
            "relationships": SlotValue(
                kind=SlotKind.RELATIONSHIP_SET,
                data=[],
                producer="relationship.vdb",
                metadata={"error": str(e)},
            )
        }
