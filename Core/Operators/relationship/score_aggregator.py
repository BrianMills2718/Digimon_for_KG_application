"""Relationship score aggregator operator.

Propagate entity PPR scores through the entity-to-relationship sparse matrix.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import RelationshipRecord, SlotKind, SlotValue


async def relationship_score_agg(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    score_vector = inputs["score_vector"].data
    p = params or {}
    top_k = max(0, int(p.get("top_k", ctx.config.top_k)))

    if score_vector is None or len(score_vector) == 0 or top_k == 0:
        return {
            "relationships": SlotValue(
                kind=SlotKind.RELATIONSHIP_SET,
                data=[],
                producer="relationship.score_agg",
            )
        }

    try:
        e2r = ctx.sparse_matrices["entity_to_rel"]
        node_scores = np.asarray(score_vector, dtype=float).reshape(-1)
        if e2r.shape[0] != node_scores.shape[0]:
            raise ValueError(
                "Sparse matrix/entity score shape mismatch: "
                f"entity_to_rel rows={e2r.shape[0]}, score_vector={node_scores.shape[0]}"
            )

        edge_scores = np.asarray(e2r.T.dot(node_scores)).reshape(-1)
        if edge_scores.size == 0:
            topk_indices = np.array([], dtype=int)
        else:
            topk_indices = np.argsort(edge_scores, kind="mergesort")[-top_k:][::-1]

        edges = await ctx.graph.get_edge_by_indices(topk_indices.tolist())
        records = []
        for idx, edge_data in zip(topk_indices, edges):
            if edge_data is None:
                continue
            records.append(
                RelationshipRecord(
                    src_id=edge_data.get("src_id", ""),
                    tgt_id=edge_data.get("tgt_id", ""),
                    relation_name=edge_data.get("relation_name", ""),
                    description=edge_data.get("description", ""),
                    weight=edge_data.get("weight", 0.0),
                    keywords=edge_data.get("keywords", ""),
                    source_id=edge_data.get("source_id", ""),
                    score=float(edge_scores[int(idx)]),
                    extra={"edge_index": int(idx)},
                )
            )

        return {
            "relationships": SlotValue(
                kind=SlotKind.RELATIONSHIP_SET,
                data=records,
                producer="relationship.score_agg",
            )
        }
    except Exception as exc:
        logger.exception(f"relationship_score_agg failed: {exc}")
        return {
            "relationships": SlotValue(
                kind=SlotKind.RELATIONSHIP_SET,
                data=[],
                producer="relationship.score_agg",
                metadata={"error": str(exc)},
            )
        }
