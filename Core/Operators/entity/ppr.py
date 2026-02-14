"""Entity PPR (Personalized PageRank) operator.

Run PPR from seed entities to find topologically important entities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def _run_ppr(ctx: Any, query: str, seed_entities: list) -> np.ndarray:
    """Run Personalized PageRank from seed entities. Mirrors BaseRetriever._run_personalized_pagerank."""
    reset_prob = np.zeros(ctx.graph.node_num)

    if ctx.config.use_entity_similarity_for_ppr:
        # FastGraphRAG-style
        reset_prob += await ctx.entities_vdb.retrieval_nodes_with_score_matrix(
            seed_entities, top_k=1, graph=ctx.graph,
        )
        reset_prob += await ctx.entities_vdb.retrieval_nodes_with_score_matrix(
            query, top_k=ctx.config.top_k_entity_for_ppr, graph=ctx.graph,
        )
    else:
        # HippoRAG-style: weight by inverse document frequency
        entity_chunk_count = None
        if ctx.sparse_matrices and "entity_to_rel" in ctx.sparse_matrices and "rel_to_chunk" in ctx.sparse_matrices:
            e2r = ctx.sparse_matrices["entity_to_rel"]
            r2c = ctx.sparse_matrices["rel_to_chunk"]
            c2e = e2r.dot(r2c).T
            c2e[c2e.nonzero()] = 1
            entity_chunk_count = c2e.sum(0).T
        else:
            num_nodes = ctx.graph.node_num
            entity_chunk_count = np.ones(num_nodes)

        for entity in seed_entities:
            name = entity.entity_name if hasattr(entity, "entity_name") else (
                entity["entity_name"] if isinstance(entity, dict) else str(entity)
            )
            idx = await ctx.graph.get_node_index(name)
            if idx is None:
                logger.warning(f"PPR: Seed entity '{name}' not found in graph, skipping")
                continue
            if ctx.config.node_specificity:
                w = 1 / float(entity_chunk_count[idx]) if entity_chunk_count[idx] != 0 else 1
                reset_prob[idx] = w
            else:
                reset_prob[idx] = 1.0

    return await ctx.graph.personalized_pagerank([reset_prob])


async def entity_ppr(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": SlotValue(QUERY_TEXT), "entities": SlotValue(ENTITY_SET)}
    Outputs: {"entities": SlotValue(ENTITY_SET), "score_vector": SlotValue(SCORE_VECTOR)}
    Params:  {"top_k": int, "link_entity": bool}
    """
    query = inputs["query"].data
    seed = inputs["entities"].data  # List[EntityRecord]
    p = params or {}
    top_k = p.get("top_k", ctx.config.top_k)

    if not seed:
        return {
            "entities": SlotValue(kind=SlotKind.ENTITY_SET, data=[], producer="entity.ppr"),
            "score_vector": SlotValue(kind=SlotKind.SCORE_VECTOR, data=np.array([]), producer="entity.ppr"),
        }

    ppr_matrix = await _run_ppr(ctx, query, seed)
    topk_indices = np.argsort(ppr_matrix)[-top_k:]
    nodes = await ctx.graph.get_node_by_indices(topk_indices)

    records = []
    for i, nd in enumerate(nodes):
        if nd is None:
            continue
        idx = topk_indices[i]
        name = nd.get(ctx.graph.entity_metakey, nd.get("entity_name", f"idx_{idx}"))
        records.append(EntityRecord(
            entity_name=str(name),
            source_id=nd.get("source_id", ""),
            entity_type=nd.get("entity_type", ""),
            description=nd.get("description", ""),
            score=float(ppr_matrix[idx]),
            extra={"ppr_index": int(idx)},
        ))

    return {
        "entities": SlotValue(kind=SlotKind.ENTITY_SET, data=records, producer="entity.ppr"),
        "score_vector": SlotValue(kind=SlotKind.SCORE_VECTOR, data=ppr_matrix, producer="entity.ppr"),
    }
