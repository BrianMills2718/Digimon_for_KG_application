"""Relationship score aggregator operator.

Propagate entity PPR scores through entity-to-relationship sparse matrix
to score relationships.
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
    """
    Inputs:  {"entities": SlotValue(ENTITY_SET), "score_vector": SlotValue(SCORE_VECTOR)}
    Outputs: {"relationships": SlotValue(RELATIONSHIP_SET)}
    Params:  {"top_k": int}
    """
    score_vector = inputs["score_vector"].data  # np.ndarray (PPR node scores)
    p = params or {}
    top_k = p.get("top_k", ctx.config.top_k)

    if score_vector is None or len(score_vector) == 0:
        return {"relationships": SlotValue(kind=SlotKind.RELATIONSHIP_SET, data=[], producer="relationship.score_agg")}

    try:
        e2r = ctx.sparse_matrices["entity_to_rel"]
        edge_scores = e2r.T.dot(score_vector)
        topk_indices = np.argsort(edge_scores)[-top_k:]
        edges = await ctx.graph.get_edge_by_indices(topk_indices)

        records = []
        for i, ed in enumerate(edges):
            if ed is None:
                continue
            idx = topk_indices[i]
            records.append(RelationshipRecord(
                src_id=ed.get("src_id", ""),
                tgt_id=ed.get("tgt_id", ""),
                relation_name=ed.get("relation_name", ""),
                description=ed.get("description", ""),
                weight=ed.get("weight", 0.0),
                keywords=ed.get("keywords", ""),
                source_id=ed.get("source_id", ""),
                score=float(edge_scores[idx]),
                extra={"edge_index": int(idx)},
            ))

        return {"relationships": SlotValue(
            kind=SlotKind.RELATIONSHIP_SET, data=records, producer="relationship.score_agg",
        )}

    except Exception as e:
        logger.exception(f"relationship_score_agg failed: {e}")
        return {"relationships": SlotValue(kind=SlotKind.RELATIONSHIP_SET, data=[], producer="relationship.score_agg")}
