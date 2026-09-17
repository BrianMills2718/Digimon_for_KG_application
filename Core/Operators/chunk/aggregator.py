"""Chunk score aggregator operator.

Propagate entity/PPR scores through entity→relationship→chunk sparse matrices
while preserving the source chunk identifiers used by the corpus store.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


def _normalize_nonnegative_scores(values) -> np.ndarray:
    """Normalize non-negative evidence scores without producing NaNs."""
    scores = np.asarray(values, dtype=float).reshape(-1)
    if scores.size == 0:
        return scores

    scores = np.where(np.isfinite(scores), scores, 0.0)
    max_score = float(np.max(scores))
    min_score = float(np.min(scores))

    if max_score <= 0.0:
        return np.zeros_like(scores)
    if max_score == min_score:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


def _chunk_id_from_store(doc_chunks: Any, index: int, doc: Any) -> Optional[str]:
    """Resolve a matrix column back to a real corpus chunk ID.

    If the store cannot recover an exact source identifier, return ``None``
    instead of inventing a matrix-position pseudo ID.
    """
    if hasattr(doc, "chunk_id") and getattr(doc, "chunk_id"):
        return str(doc.chunk_id)
    if isinstance(doc, dict):
        for key in ("chunk_id", "id"):
            if doc.get(key):
                return str(doc[key])

    mapping = getattr(doc_chunks, "_chunks", None)
    if isinstance(mapping, dict):
        keys = list(mapping.keys())
        if 0 <= index < len(keys):
            return str(keys[index])
    return None


def _chunk_text(doc: Any) -> str:
    """Return real textual evidence or an empty string for unknown objects."""
    if isinstance(doc, str):
        return doc.strip()
    if hasattr(doc, "content"):
        return str(doc.content or "").strip()
    if hasattr(doc, "text"):
        return str(doc.text or "").strip()
    if isinstance(doc, dict):
        return str(doc.get("content", doc.get("text", "")) or "").strip()
    return ""


async def chunk_aggregator(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"score_vector": SCORE_VECTOR}
    Outputs: {"chunks": CHUNK_SET}
    Params:  {"top_k": int}
    """
    score_vector = inputs["score_vector"].data
    p = params or {}
    top_k = p.get("top_k", ctx.config.top_k)

    if score_vector is None or len(score_vector) == 0:
        return {
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=[],
                producer="chunk.aggregator",
            )
        }

    try:
        e2r = ctx.sparse_matrices["entity_to_rel"]
        r2c = ctx.sparse_matrices["rel_to_chunk"]

        node_scores = np.asarray(score_vector, dtype=float).reshape(-1)
        if e2r.shape[0] != node_scores.shape[0]:
            raise ValueError(
                "Sparse matrix/entity score shape mismatch: "
                f"entity_to_rel rows={e2r.shape[0]}, score_vector={node_scores.shape[0]}"
            )

        edge_scores = np.asarray(e2r.T.dot(node_scores)).reshape(-1)
        if r2c.shape[0] != edge_scores.shape[0]:
            raise ValueError(
                "Sparse matrix relationship shape mismatch: "
                f"rel_to_chunk rows={r2c.shape[0]}, edge_scores={edge_scores.shape[0]}"
            )

        chunk_scores_raw = np.asarray(r2c.T.dot(edge_scores)).reshape(-1)
        chunk_scores = _normalize_nonnegative_scores(chunk_scores_raw)

        if chunk_scores.size == 0 or float(np.max(chunk_scores)) <= 0.0:
            return {
                "chunks": SlotValue(
                    kind=SlotKind.CHUNK_SET,
                    data=[],
                    producer="chunk.aggregator",
                )
            }

        ranked_indices = np.argsort(chunk_scores, kind="mergesort")[::-1][:top_k]
        docs = await ctx.doc_chunks.get_data_by_indices(ranked_indices.tolist())

        records = []
        skipped_unresolved = 0
        for index, doc in zip(ranked_indices, docs):
            if doc is None:
                continue
            index_int = int(index)
            chunk_id = _chunk_id_from_store(ctx.doc_chunks, index_int, doc)
            text = _chunk_text(doc)
            if not chunk_id or not text:
                skipped_unresolved += 1
                continue

            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(chunk_scores[index_int]),
                    extra={
                        "matrix_index": index_int,
                        "raw_propagated_score": float(chunk_scores_raw[index_int]),
                    },
                )
            )

        metadata = {}
        if skipped_unresolved:
            metadata["skipped_unresolved_evidence"] = skipped_unresolved

        return {
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=records,
                producer="chunk.aggregator",
                metadata=metadata,
            )
        }
    except Exception as exc:
        logger.exception(f"chunk_aggregator failed: {exc}")
        return {
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=[],
                producer="chunk.aggregator",
                metadata={"error": str(exc)},
            )
        }
