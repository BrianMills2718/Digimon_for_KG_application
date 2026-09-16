# Core/AgentTools/entity_tools.py

import os
import logging
import json as _json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from pydantic import BaseModel

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    EntityPPRInputs,
    EntityPPROutputs,
    EntityVDBSearchInputs,
    EntityVDBSearchOutputs,
    VDBSearchResultItem,
    EntityOneHopInput,
    EntityOneHopOutput,
    EntityRelNodeInput,
    EntityRelNodeOutputs,
    EntityAgentInputs, EntityAgentOutputs, ExtractedEntityData,
    EntityLinkInputs, EntityLinkOutputs, LinkedEntityPair,
    EntityTFIDFInputs, EntityTFIDFOutputs,
)
from Core.Schema.EntityRelation import Entity as CoreEntity
from Core.Graph.BaseGraph import BaseGraph
from Core.Index.FaissIndex import FaissIndex
from Core.Index.BaseIndex import BaseIndex
from Core.Index.Schema import FAISSIndexConfig
from Core.AgentTools.index_config_helper import create_faiss_index_config
from Core.Provider.BaseEmb import BaseEmb

logger = logging.getLogger(__name__)


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

            logger.info(f"Entity.VDBSearch: Original query: '{params.query_text}'")
            logger.info(f"Entity.VDBSearch: Expanded to {len(expanded_terms)} terms")

            # Keep score as our own scalar instead of mutating third-party
            # NodeWithScore objects. This makes query expansion independent of
            # the concrete retrieval result class returned by LlamaIndex.
            all_results = []
            seen_entities = set()

            logger.info(
                f"Entity.VDBSearch: Searching with original query: '{params.query_text}'"
            )
            results = await vdb_instance.retrieval(
                query=params.query_text,
                top_k=params.top_k_results * 2,
            )

            for node_with_score in results:
                node = node_with_score.node
                entity_name = node.metadata.get(
                    "name",
                    node.metadata.get(
                        "entity_name", node.metadata.get("id", "")
                    ),
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
                    if term != params.query_text.lower():
                        logger.debug(
                            f"Entity.VDBSearch: Searching with expanded term: '{term}'"
                        )
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

                        except Exception as e:
                            logger.warning(
                                f"Entity.VDBSearch: Error searching with term '{term}': {e}"
                            )

            all_results.sort(key=lambda item: item[3], reverse=True)
            top_results = all_results[:params.top_k_results]

            output_entities: List[VDBSearchResultItem] = []
            for node, entity_name, node_id, score in top_results:
                if not entity_name:
                    entity_name = node.text[:50]
                    logger.warning(
                        f"Entity.VDBSearch: No entity name found for node {node_id}, using text excerpt"
                    )

                output_entities.append(
                    VDBSearchResultItem(
                        node_id=str(node_id),
                        entity_name=str(entity_name),
                        score=score,
                    )
                )
                logger.debug(
                    f"Entity.VDBSearch: Found entity '{entity_name}' with score {score}"
                )

        elif params.query_embedding:
            logger.warning(
                "Entity.VDBSearch: Querying by direct embedding is not implemented yet for FaissIndex."
            )
            return EntityVDBSearchOutputs(similar_entities=[])

        logger.info(
            f"Entity.VDBSearch: Found {len(output_entities)} similar entities."
        )
        return EntityVDBSearchOutputs(similar_entities=output_entities)

    except Exception as e:
        logger.error(
            f"Entity.VDBSearch: Error during VDB search: {e}",
            exc_info=True,
        )
        return EntityVDBSearchOutputs(similar_entities=[])

# --- Tool Implementation for: Entity Personalized PageRank (PPR) ---
# tool_id: "Entity.PPR"
from Core.Graph.BaseGraph import BaseGraph  # For type hinting graph_instance

async def entity_ppr_tool(
    params: EntityPPRInputs,
    graphrag_context: GraphRAGContext
) -> EntityPPROutputs:
    """
    Computes Personalized PageRank for entities in a graph based on seed entity IDs.
    """
    logger.info(f"Executing tool 'Entity.PPR' with parameters: {params.model_dump_json(indent=2)}")

    graph_instance: Optional[BaseGraph] = graphrag_context.get_graph_instance(params.graph_reference_id)
    if not graph_instance:
        logger.error("Entity.PPR: Graph instance not found in GraphRAGContext.")
        raise ValueError("Graph instance is required for PPR.")

    if not params.seed_entity_ids:
        logger.warning("Entity.PPR: No seed_entity_ids provided. Returning empty results.")
        return EntityPPROutputs(ranked_entities=[])

    # 1. Prepare the personalization vector for PageRank
    # For this implementation, we'll create a simple personalization vector
    # where seed nodes get uniform non-zero probability.
    
    # First, ensure graph has node_num available and > 0
    if not hasattr(graph_instance, 'node_num') or not graph_instance.node_num or graph_instance.node_num <= 0:
        logger.error(f"Entity.PPR: graph_instance.node_num is not available or invalid: {getattr(graph_instance, 'node_num', 'Attribute Missing')}")
        # Attempt to get it dynamically if underlying graph exists
        if hasattr(graph_instance, '_graph') and graph_instance._graph is not None:
             try:
                 graph_instance.node_num = graph_instance._graph.number_of_nodes()
                 logger.info(f"Entity.PPR: Dynamically set graph_instance.node_num to {graph_instance.node_num}")
                 if graph_instance.node_num <= 0:
                     raise ValueError("Graph has no nodes after dynamic check.")
             except Exception as e:
                 logger.error(f"Entity.PPR: Could not dynamically determine node_num: {e}")
                 raise ValueError("Graph node count is unavailable or invalid for PPR.")
        else:
             raise ValueError("Graph node count is unavailable or invalid for PPR.")

    personalization_vector = np.zeros(graph_instance.node_num)
    valid_seed_indices_count = 0
    
    seed_node_indices = []
    for entity_id in params.seed_entity_ids:
        try:
            node_idx = await graph_instance.get_node_index(entity_id)
            if node_idx is not None and 0 <= node_idx < graph_instance.node_num:
                seed_node_indices.append(node_idx)
                valid_seed_indices_count += 1
            else:
                logger.warning(f"Entity.PPR: Seed entity_id '{entity_id}' not found in graph or index out of bounds. Skipping.")
        except Exception as e:
            logger.warning(f"Entity.PPR: Error getting index for seed_id '{entity_id}': {e}. Skipping.")
    
    if valid_seed_indices_count == 0:
        logger.warning("Entity.PPR: None of the provided seed_entity_ids were found in the graph. Returning empty results.")
        return EntityPPROutputs(ranked_entities=[])

    for idx in seed_node_indices:
         personalization_vector[idx] = 1.0 / valid_seed_indices_count
    
    logger.debug(f"Entity.PPR: Personalization vector created with {valid_seed_indices_count} active seed(s). Sum: {np.sum(personalization_vector)}")

    # 2. Call the graph's personalized_pagerank method
    # BaseGraph.personalized_pagerank(reset_prob_chunk, damping) returns a numpy array
    # indexed by node position, not a dict.
    damping = params.personalization_weight_alpha or 0.15
    try:
        logger.info(f"Entity.PPR: Calling graph.personalized_pagerank with damping={damping}")

        ppr_scores_array = await graph_instance.personalized_pagerank(
            reset_prob_chunk=[personalization_vector],
            damping=damping,
        )
        logger.debug(f"Entity.PPR: Received score array of length {len(ppr_scores_array)}")

    except Exception as e:
        logger.error(f"Entity.PPR: Error during personalized_pagerank execution: {e}", exc_info=True)
        raise

    # 3. Convert numpy array to {node_id: score} dict
    if ppr_scores_array is None or len(ppr_scores_array) == 0:
        logger.warning("Entity.PPR: personalized_pagerank returned empty or None scores.")
        return EntityPPROutputs(ranked_entities=[])

    # Map node indices back to node IDs
    node_list = list(graph_instance._graph.graph.nodes())
    ppr_scores_dict: Dict[str, float] = {}
    for idx, score in enumerate(ppr_scores_array):
        if idx < len(node_list):
            ppr_scores_dict[node_list[idx]] = float(score)

    # Sort by score in descending order
    sorted_ranked_entities = sorted(ppr_scores_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Truncate to top_k_results if specified
    top_k = params.top_k_results
    if top_k is not None and top_k > 0:
        final_ranked_entities = sorted_ranked_entities[:top_k]
    else:
        final_ranked_entities = sorted_ranked_entities
    
    logger.info(f"Entity.PPR: Tool execution finished. Returning {len(final_ranked_entities)} ranked entities.")
    return EntityPPROutputs(ranked_entities=final_ranked_entities)


async def entity_agent_tool(
    params: EntityAgentInputs,
    graphrag_context: GraphRAGContext,
) -> EntityAgentOutputs:
    """
    Uses an LLM to extract entities from text context guided by a query.
    """
    logger.info(
        f"Executing Entity.Agent: query='{params.query_text[:80]}', "
        f"types={params.target_entity_types}"
    )

    text = params.text_context
    if isinstance(text, list):
        text = "\n\n".join(text)

    max_entities = params.max_entities_to_extract or 10
    types_str = ", ".join(params.target_entity_types) if params.target_entity_types else "any type"

    prompt = f"""Extract entities from the following text that are relevant to the query.
Return a JSON array of objects with fields: "entity_name", "entity_type", "description".
Extract at most {max_entities} entities. Focus on entity types: {types_str}.

Query: {params.query_text}

Text:
{text[:4000]}

Return ONLY a JSON array. No other text."""

    llm = graphrag_context.llm_provider
    if llm is None:
        logger.error("Entity.Agent: No LLM provider available")
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
        for e in entities_raw[:max_entities]:
            eed = ExtractedEntityData(
                entity_name=e.get("entity_name", "unknown"),
                source_id="entity_agent_tool",
                entity_type=e.get("entity_type", "unknown"),
                description=e.get("description", ""),
            )
            eed.extraction_confidence = e.get("confidence", 0.8)
            extracted.append(eed)

        logger.info(f"Entity.Agent: Extracted {len(extracted)} entities")
        return EntityAgentOutputs(extracted_entities=extracted)

    except Exception as e:
        logger.error(f"Entity.Agent: LLM extraction failed: {e}", exc_info=True)
        return EntityAgentOutputs(extracted_entities=[])


# --- Tool Implementation for: Entity Operator - Link ---
# tool_id: "Entity.Link"

async def entity_link_tool(
    params: EntityLinkInputs,
    graphrag_context: GraphRAGContext,
) -> EntityLinkOutputs:
    """
    Links source entity mentions to canonical entities in a VDB.
    For each source entity, searches the VDB with top_k=1 and applies threshold.
    """
    logger.info(
        f"Executing Entity.Link: {len(params.source_entities)} entities, "
        f"kb='{params.knowledge_base_reference_id}'"
    )

    vdb_instance = None
    if params.knowledge_base_reference_id:
        vdb_instance = graphrag_context.get_vdb_instance(params.knowledge_base_reference_id)

    if vdb_instance is None:
        logger.error(f"Entity.Link: VDB '{params.knowledge_base_reference_id}' not found")
        results = []
        for src in params.source_entities:
            mention = src if isinstance(src, str) else getattr(src, 'entity_name', str(src))
            results.append(LinkedEntityPair(
                source_entity_mention=mention,
                link_status="not_found",
            ))
        return EntityLinkOutputs(linked_entities_results=results)

    threshold = params.similarity_threshold or 0.0
    results = []

    for src in params.source_entities:
        mention = src if isinstance(src, str) else getattr(src, 'entity_name', str(src))
        try:
            search_results = await vdb_instance.retrieval(query=mention, top_k=1)
            if search_results:
                top = search_results[0]
                score = float(top.score) if top.score is not None else 0.0
                entity_name = top.node.metadata.get(
                    "name", top.node.metadata.get("entity_name", top.node.text[:50])
                )
                if score >= threshold:
                    results.append(LinkedEntityPair(
                        source_entity_mention=mention,
                        linked_entity_id=entity_name,
                        linked_entity_description=top.node.text[:200],
                        similarity_score=score,
                        link_status="linked",
                    ))
                else:
                    results.append(LinkedEntityPair(
                        source_entity_mention=mention,
                        similarity_score=score,
                        link_status="not_found",
                    ))
            else:
                results.append(LinkedEntityPair(
                    source_entity_mention=mention,
                    link_status="not_found",
                ))
        except Exception as e:
            logger.warning(f"Entity.Link: Error linking '{mention}': {e}")
            results.append(LinkedEntityPair(
                source_entity_mention=mention,
                link_status="not_found",
            ))

    logger.info(f"Entity.Link: Linked {sum(1 for r in results if r.link_status == 'linked')}/{len(results)} entities")
    return EntityLinkOutputs(linked_entities_results=results)


# --- Tool Implementation for: Entity TF-IDF Ranking ---
# tool_id: "Entity.TFIDF"

async def entity_tfidf_tool(
    params: EntityTFIDFInputs,
    graphrag_context: GraphRAGContext,
) -> EntityTFIDFOutputs:
    """
    Ranks candidate entities by TF-IDF cosine similarity of their descriptions
    against a query text. Uses sklearn TfidfVectorizer.
    """
    logger.info(
        f"Executing Entity.TFIDF: {len(params.candidate_entity_ids)} candidates, "
        f"query='{params.query_text[:60]}', graph='{params.graph_reference_id}'"
    )

    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
    if graph_instance is None:
        logger.error(f"Entity.TFIDF: Graph '{params.graph_reference_id}' not found")
        return EntityTFIDFOutputs(ranked_entities=[])

    # Extract NetworkX graph
    nx_graph = None
    if hasattr(graph_instance, '_graph') and hasattr(graph_instance._graph, 'graph') and isinstance(graph_instance._graph.graph, nx.Graph):
        nx_graph = graph_instance._graph.graph
    elif hasattr(graph_instance, '_graph') and isinstance(graph_instance._graph, nx.Graph):
        nx_graph = graph_instance._graph
    if nx_graph is None:
        logger.error("Entity.TFIDF: Could not access NetworkX graph")
        return EntityTFIDFOutputs(ranked_entities=[])

    # Collect entity descriptions
    entity_ids = []
    entity_docs = []
    for eid in params.candidate_entity_ids:
        if eid in nx_graph:
            node_data = nx_graph.nodes[eid]
            desc = node_data.get("description", "")
            entity_type = node_data.get("entity_type", "")
            doc = f"{eid} {entity_type} {desc}".strip()
            if doc:
                entity_ids.append(eid)
                entity_docs.append(doc)

    if not entity_docs:
        logger.warning("Entity.TFIDF: No entity documents to rank")
        return EntityTFIDFOutputs(ranked_entities=[])

    # Build TF-IDF and compute similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(entity_docs)
        query_vec = vectorizer.transform([params.query_text])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    except Exception as e:
        logger.error(f"Entity.TFIDF: TF-IDF computation failed: {e}")
        return EntityTFIDFOutputs(ranked_entities=[])

    # Rank by score
    ranked = sorted(zip(entity_ids, scores.tolist()), key=lambda x: x[1], reverse=True)
    top_k = params.top_k or 10
    ranked = ranked[:top_k]

    logger.info(f"Entity.TFIDF: Returning {len(ranked)} ranked entities")
    return EntityTFIDFOutputs(ranked_entities=ranked)
