"""Chunk score aggregator operator.

Propagate relationship scores through rel-to-chunk sparse matrix
to score and retrieve text chunks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from Core.Common.Logger import logger
from Core.Common.Utils import min_max_normalize
from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


async def chunk_aggregator(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"score_vector": SlotValue(SCORE_VECTOR)}  -- PPR node scores
    Outputs: {"chunks": SlotValue(CHUNK_SET)}
    Params:  {"top_k": int}

    Uses entity_to_rel and rel_to_chunk sparse matrices to propagate
    node PPR scores all the way to chunks.
    """
    score_vector = inputs["score_vector"].data  # np.ndarray
    p = params or {}
    top_k = p.get("top_k", ctx.config.top_k)

    if score_vector is None or len(score_vector) == 0:
        return {"chunks": SlotValue(kind=SlotKind.CHUNK_SET, data=[], producer="chunk.aggregator")}

    try:
        e2r = ctx.sparse_matrices["entity_to_rel"]
        r2c = ctx.sparse_matrices["rel_to_chunk"]

        edge_prob = e2r.T.dot(score_vector)
        chunk_prob = r2c.T.dot(edge_prob)
        chunk_prob = min_max_normalize(chunk_prob)

        sorted_ids = np.argsort(chunk_prob, kind="mergesort")[::-1]
        sorted_scores = chunk_prob[sorted_ids]

        docs = await ctx.doc_chunks.get_data_by_indices(sorted_ids[:top_k])

        records = []
        for i, doc in enumerate(docs):
            if doc is None:
                continue
            records.append(ChunkRecord(
                chunk_id=str(sorted_ids[i]),
                text=doc,
                score=float(sorted_scores[i]),
            ))

        return {"chunks": SlotValue(kind=SlotKind.CHUNK_SET, data=records, producer="chunk.aggregator")}

    except Exception as e:
        logger.exception(f"chunk_aggregator failed: {e}")
        return {"chunks": SlotValue(kind=SlotKind.CHUNK_SET, data=[], producer="chunk.aggregator")}
