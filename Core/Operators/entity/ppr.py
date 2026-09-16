"""Entity PPR (Personalized PageRank) operator."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def _run_ppr(
    ctx: Any,
    query: str,
    seed_entities: list,
    damping: float = 0.85,
) -> np.ndarray:
    """Run Personalized PageRank from seed entities."""
    if not 0.0 <= damping <= 1.0:
        raise ValueError("PPR damping must be between 0 and 1")

    reset_prob = np.zeros(ctx.graph.node_num)

    if ctx.config.use_entity_similarity_for_ppr:
        # FastGraphRAG-style: combine VDB similarities for supplied seeds and query.
        reset_prob += await ctx.entities_vdb.retrieval_nodes_with_score_matrix(
            seed_entities,
            top_k=1,
            graph=ctx.graph,
        )
        reset_prob += await ctx.entities_vdb.retrieval_nodes_with_score_matrix(
            query,
            top_k=ctx.config.top_k_entity_for_ppr,
            graph=ctx.graph,
        )
    else:
        # HippoRAG-style: weight linked seed entities by inverse document frequency.
        if (
            ctx.sparse_matrices
            and "entity_to_rel" in ctx.sparse_matrices
            and "rel_to_chunk" in ctx.sparse_matrices
        ):
            e2r = ctx.sparse_matrices["entity_to_rel"]
            r2c = ctx.sparse_matrices["rel_to_chunk"]
            c2e = e2r.dot(r2c).T
            c2e[c2e.nonzero()] = 1
            entity_chunk_count = np.asarray(c2e.sum(0)).reshape(-1)
        else:
            entity_chunk_count = np.ones(ctx.graph.node_num)

        for entity in seed_entities:
            name = (
                entity.entity_name
                if hasattr(entity, "entity_name")
                else entity["entity_name"]
                if isinstance(entity, dict)
                else str(entity)
            )
            idx = await ctx.graph.get_node_index(name)
            if idx is None:
                logger.warning(f"PPR: Seed entity '{name}' not found in graph, skipping")
                continue
            if ctx.config.node_specificity:
                count = float(entity_chunk_count[idx])
                reset_prob[idx] = 1.0 / count if count > 0 else 1.0
            else:
                reset_prob[idx] = 1.0

    if not np.any(reset_prob):
        return np.zeros(ctx.graph.node_num)

    return await ctx.graph.personalized_pagerank(
        [reset_prob],
        damping=damping,
    )


async def entity_ppr(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": QUERY_TEXT, "entities": ENTITY_SET}
    Outputs: {"entities": ENTITY_SET, "score_vector": SCORE_VECTOR}
    Params:  {"top_k": int, "damping": float}
    """
    query = inputs["query"].data
    seed = inputs["entities"].data
    p = params or {}
    top_k = p.get("top_k", ctx.config.top_k)
    damping = float(p.get("damping", 0.85))

    if not seed:
        return {
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[],
                producer="entity.ppr",
            ),
            "score_vector": SlotValue(
                kind=SlotKind.SCORE_VECTOR,
                data=np.array([]),
                producer="entity.ppr",
            ),
        }

    ppr_matrix = await _run_ppr(ctx, query, seed, damping=damping)
    if ppr_matrix.size == 0:
        topk_indices = np.array([], dtype=int)
    else:
        topk_indices = np.argsort(ppr_matrix)[-top_k:][::-1]

    nodes = await ctx.graph.get_node_by_indices(topk_indices)
    records = []
    for idx, node_data in zip(topk_indices, nodes):
        if node_data is None:
            continue
        name = node_data.get(
            ctx.graph.entity_metakey,
            node_data.get("entity_name", f"idx_{idx}"),
        )
        records.append(
            EntityRecord(
                entity_name=str(name),
                source_id=node_data.get("source_id", ""),
                entity_type=node_data.get("entity_type", ""),
                description=node_data.get("description", ""),
                score=float(ppr_matrix[idx]),
                extra={"ppr_index": int(idx)},
            )
        )

    return {
        "entities": SlotValue(
            kind=SlotKind.ENTITY_SET,
            data=records,
            producer="entity.ppr",
        ),
        "score_vector": SlotValue(
            kind=SlotKind.SCORE_VECTOR,
            data=ppr_matrix,
            producer="entity.ppr",
        ),
    }
