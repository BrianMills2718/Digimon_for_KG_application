# Core/AgentTools/entity_tools.py

import logging
import json as _json
from typing import Dict, List, Optional

import numpy as np

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    EntityAgentInputs,
    EntityAgentOutputs,
    EntityLinkInputs,
    EntityLinkOutputs,
    EntityPPRInputs,
    EntityPPROutputs,
    EntityTFIDFInputs,
    EntityTFIDFOutputs,
    EntityVDBSearchInputs,
    EntityVDBSearchOutputs,
    ExtractedEntityData,
    LinkedEntityPair,
    VDBSearchResultItem,
)
from Core.Graph.BaseGraph import BaseGraph

logger = logging.getLogger(__name__)


def ppr_damping_from_teleport_alpha(alpha: Optional[float]) -> float:
    """Convert teleport/reset probability into igraph's damping factor."""
    teleport_alpha = 0.15 if alpha is None else float(alpha)
    if not 0.0 <= teleport_alpha <= 1.0:
        raise ValueError("personalization_weight_alpha must be between 0 and 1")
    return 1.0 - teleport_alpha


async def entity_vdb_search_tool(
    params: EntityVDBSearchInputs,
    graphrag_context: GraphRAGContext,
) -> EntityVDBSearchOutputs:
    logger.info(
        f"Executing tool 'Entity.VDBSearch' with parameters: "
        f"vdb_reference_id='{params.vdb_reference_id}', query_text='{params.query_text}', "
        f"top_k_results={params.top_k_results}"
    )

    if not (params.query_text or params.query_embedding):
        logger.error("Entity.VDBSearch: Either query_text or query_embedding must be provided.")
        return EntityVDBSearchOutputs(similar_entities=[])

    vdb_instance = graphrag_context.get_vdb_instance(params.vdb_reference_id)
    if not vdb_instance:
        logger.error(
            f"Entity.VDBSearch: VDB reference '{params.vdb_reference_id}' not found in context. "
            f"Available VDBs: {list(graphrag_context.vdbs.keys())}"
        )
        return EntityVDBSearchOutputs(similar_entities=[])

    try:
        if params.query_text:
            from Core.AgentTools.query_expansion import query_expander

            expanded_terms = query_expander.expand_query(params.query_text)
            all_results = []
            seen_entities = set()

            results = await vdb_instance.retrieval(
                query=params.query_text,
                top_k=params.top_k_results * 2,
            )

            for node_with_score in results:
                node = node_with_score.node
                entity_name = node.metadata.get(
                    "name",
                    node.metadata.get("entity_name", node.metadata.get("id", "")),
                )
                node_id = node.metadata.get("id", node.node_id)
                score = (
                    float(node_with_score.score)
                    if node_with_score.score is not None
                    else 0.0
                )
                if entity_name and entity_name not in seen_entities:
                    seen_entities.add(entity_name)
                    all_results.append((node, entity_name, node_id, score))

            if len(all_results) < params.top_k_results:
                for term in expanded_terms[:5]:
                    if term == params.query_text.lower():
                        continue
                    try:
                        expanded_results = await vdb_instance.retrieval(
                            query=term,
                            top_k=params.top_k_results,
                        )
                        for node_with_score in expanded_results:
                            node = node_with_score.node
                            entity_name = node.metadata.get(
                                "name",
                                node.metadata.get(
                                    "entity_name", node.metadata.get("id", "")
                                ),
                            )
                            node_id = node.metadata.get("id", node.node_id)
                            if entity_name and entity_name not in seen_entities:
                                seen_entities.add(entity_name)
                                raw_score = (
                                    float(node_with_score.score)
                                    if node_with_score.score is not None
                                    else 0.0
                                )
                                all_results.append(
                                    (node, entity_name, node_id, raw_score * 0.9)
                                )
                        if len(all_results) >= params.top_k_results * 2:
                            break
                    except Exception as exc:
                        logger.warning(
                            f"Entity.VDBSearch: Error searching expanded term '{term}': {exc}"
                        )

            all_results.sort(key=lambda item: item[3], reverse=True)
            output_entities = [
                VDBSearchResultItem(
                    node_id=str(node_id),
                    entity_name=str(entity_name or node.text[:50]),
                    score=score,
                )
                for node, entity_name, node_id, score in all_results[: params.top_k_results]
            ]
            return EntityVDBSearchOutputs(similar_entities=output_entities)

        logger.warning(
            "Entity.VDBSearch: Querying by direct embedding is not implemented yet for FaissIndex."
        )
        return EntityVDBSearchOutputs(similar_entities=[])

    except Exception as exc:
        logger.error(f"Entity.VDBSearch: Error during VDB search: {exc}", exc_info=True)
        return EntityVDBSearchOutputs(similar_entities=[])


async def entity_ppr_tool(
    params: EntityPPRInputs,
    graphrag_context: GraphRAGContext,
) -> EntityPPROutputs:
    """Compute Personalized PageRank from seed entity IDs."""
    logger.info(
        f"Executing tool 'Entity.PPR' with parameters: {params.model_dump_json(indent=2)}"
    )

    graph_instance: Optional[BaseGraph] = graphrag_context.get_graph_instance(
        params.graph_reference_id
    )
    if graph_instance is None:
        raise ValueError("Graph instance is required for PPR.")
    if not params.seed_entity_ids:
        return EntityPPROutputs(ranked_entities=[])

    node_count = graph_instance.node_num
    if not isinstance(node_count, int) or node_count <= 0:
        raise ValueError("Graph node count is unavailable or invalid for PPR.")

    seed_node_indices = []
    for entity_id in params.seed_entity_ids:
        try:
            node_idx = await graph_instance.get_node_index(entity_id)
            if node_idx is not None and 0 <= node_idx < node_count:
                seed_node_indices.append(node_idx)
            else:
                logger.warning(
                    f"Entity.PPR: Seed entity_id '{entity_id}' not found in graph; skipping."
                )
        except Exception as exc:
            logger.warning(
                f"Entity.PPR: Error getting index for seed_id '{entity_id}': {exc}; skipping."
            )

    if not seed_node_indices:
        return EntityPPROutputs(ranked_entities=[])

    personalization_vector = np.zeros(node_count)
    seed_weight = 1.0 / len(seed_node_indices)
    for idx in seed_node_indices:
        personalization_vector[idx] = seed_weight

    # EntityPPRInputs defines personalization_weight_alpha as teleport/reset
    # probability. igraph defines damping as the probability of continuing the
    # walk, so damping = 1 - alpha.
    damping = ppr_damping_from_teleport_alpha(params.personalization_weight_alpha)
    logger.info(
        f"Entity.PPR: teleport_alpha={params.personalization_weight_alpha}, damping={damping}"
    )

    ppr_scores_array = await graph_instance.personalized_pagerank(
        reset_prob_chunk=[personalization_vector],
        damping=damping,
    )
    if ppr_scores_array is None or len(ppr_scores_array) == 0:
        return EntityPPROutputs(ranked_entities=[])

    node_list = list(graph_instance._graph.graph.nodes())
    ranked = sorted(
        (
            (node_list[idx], float(score))
            for idx, score in enumerate(ppr_scores_array)
            if idx < len(node_list)
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    top_k = params.top_k_results
    if top_k is not None and top_k > 0:
        ranked = ranked[:top_k]
    return EntityPPROutputs(ranked_entities=ranked)


async def entity_agent_tool(
    params: EntityAgentInputs,
    graphrag_context: GraphRAGContext,
) -> EntityAgentOutputs:
    """Use an LLM to extract entities from text context guided by a query."""
    text = params.text_context
    if isinstance(text, list):
        text = "\n\n".join(text)

    max_entities = params.max_entities_to_extract or 10
    types_str = (
        ", ".join(params.target_entity_types)
        if params.target_entity_types
        else "any type"
    )
    prompt = f"""Extract entities from the following text that are relevant to the query.
Return a JSON array of objects with fields: "entity_name", "entity_type", "description".
Extract at most {max_entities} entities. Focus on entity types: {types_str}.

Query: {params.query_text}

Text:
{text[:4000]}

Return ONLY a JSON array. No other text."""

    llm = graphrag_context.llm_provider
    if llm is None:
        return EntityAgentOutputs(extracted_entities=[])

    try:
        response = await llm.aask(prompt)
        resp_text = response.strip()
        if resp_text.startswith("```"):
            resp_text = resp_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        entities_raw = _json.loads(resp_text)
        if not isinstance(entities_raw, list):
            entities_raw = [entities_raw]

        extracted = []
        for entity in entities_raw[:max_entities]:
            record = ExtractedEntityData(
                entity_name=entity.get("entity_name", "unknown"),
                source_id="entity_agent_tool",
                entity_type=entity.get("entity_type", "unknown"),
                description=entity.get("description", ""),
            )
            record.extraction_confidence = entity.get("confidence", 0.8)
            extracted.append(record)
        return EntityAgentOutputs(extracted_entities=extracted)
    except Exception as exc:
        logger.error(f"Entity.Agent: LLM extraction failed: {exc}", exc_info=True)
        return EntityAgentOutputs(extracted_entities=[])


async def entity_link_tool(
    params: EntityLinkInputs,
    graphrag_context: GraphRAGContext,
) -> EntityLinkOutputs:
    """Link source entity mentions to the closest entity in a VDB."""
    vdb_instance = (
        graphrag_context.get_vdb_instance(params.knowledge_base_reference_id)
        if params.knowledge_base_reference_id
        else None
    )
    if vdb_instance is None:
        return EntityLinkOutputs(
            linked_entities_results=[
                LinkedEntityPair(
                    source_entity_mention=(
                        src
                        if isinstance(src, str)
                        else getattr(src, "entity_name", str(src))
                    ),
                    link_status="not_found",
                )
                for src in params.source_entities
            ]
        )

    threshold = params.similarity_threshold or 0.0
    results = []
    for src in params.source_entities:
        mention = (
            src if isinstance(src, str) else getattr(src, "entity_name", str(src))
        )
        try:
            search_results = await vdb_instance.retrieval(query=mention, top_k=1)
            if not search_results:
                results.append(
                    LinkedEntityPair(
                        source_entity_mention=mention,
                        link_status="not_found",
                    )
                )
                continue

            top = search_results[0]
            score = float(top.score) if top.score is not None else 0.0
            entity_name = top.node.metadata.get(
                "name", top.node.metadata.get("entity_name", top.node.text[:50])
            )
            if score >= threshold:
                results.append(
                    LinkedEntityPair(
                        source_entity_mention=mention,
                        linked_entity_id=entity_name,
                        linked_entity_description=top.node.text[:200],
                        similarity_score=score,
                        link_status="linked",
                    )
                )
            else:
                results.append(
                    LinkedEntityPair(
                        source_entity_mention=mention,
                        similarity_score=score,
                        link_status="not_found",
                    )
                )
        except Exception as exc:
            logger.warning(f"Entity.Link: Error linking '{mention}': {exc}")
            results.append(
                LinkedEntityPair(
                    source_entity_mention=mention,
                    link_status="not_found",
                )
            )

    return EntityLinkOutputs(linked_entities_results=results)


async def entity_tfidf_tool(
    params: EntityTFIDFInputs,
    graphrag_context: GraphRAGContext,
) -> EntityTFIDFOutputs:
    """Rank candidate entities by TF-IDF cosine similarity to a query."""
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
    if graph_instance is None:
        return EntityTFIDFOutputs(ranked_entities=[])

    nx_graph = None
    if (
        hasattr(graph_instance, "_graph")
        and hasattr(graph_instance._graph, "graph")
        and isinstance(graph_instance._graph.graph, nx.Graph)
    ):
        nx_graph = graph_instance._graph.graph
    elif hasattr(graph_instance, "_graph") and isinstance(graph_instance._graph, nx.Graph):
        nx_graph = graph_instance._graph
    if nx_graph is None:
        return EntityTFIDFOutputs(ranked_entities=[])

    entity_ids = []
    entity_docs = []
    for entity_id in params.candidate_entity_ids:
        if entity_id not in nx_graph:
            continue
        node_data = nx_graph.nodes[entity_id]
        description = node_data.get("description", "")
        entity_type = node_data.get("entity_type", "")
        document = f"{entity_id} {entity_type} {description}".strip()
        if document:
            entity_ids.append(entity_id)
            entity_docs.append(document)

    if not entity_docs:
        return EntityTFIDFOutputs(ranked_entities=[])

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(entity_docs)
        query_vec = vectorizer.transform([params.query_text])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    except Exception as exc:
        logger.error(f"Entity.TFIDF: TF-IDF computation failed: {exc}")
        return EntityTFIDFOutputs(ranked_entities=[])

    ranked = sorted(
        zip(entity_ids, scores.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return EntityTFIDFOutputs(ranked_entities=ranked[: (params.top_k or 10)])
