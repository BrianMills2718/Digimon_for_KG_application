from typing import List, Tuple, Dict, Any, Optional

import logging
import networkx as nx

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    RelationshipOneHopNeighborsInputs,
    RelationshipOneHopNeighborsOutputs,
    RelationshipData,
    RelationshipScoreAggregatorInputs,
    RelationshipScoreAggregatorOutputs,
    RelationshipVDBBuildInputs,
    RelationshipVDBBuildOutputs,
    RelationshipVDBSearchInputs,
    RelationshipVDBSearchOutputs,
    RelationshipAgentInputs,
    RelationshipAgentOutputs,
)
from Core.AgentTools.index_config_helper import create_faiss_index_config
from Core.Index.FaissIndex import FaissIndex

logger = logging.getLogger(__name__)


async def relationship_one_hop_neighbors_tool(
    params: RelationshipOneHopNeighborsInputs,
    graphrag_context: GraphRAGContext,
) -> RelationshipOneHopNeighborsOutputs:
    logger.info(
        f"Executing tool 'Relationship.OneHopNeighbors' with parameters: "
        f"entity_ids={params.entity_ids}, "
        f"graph_reference_id='{params.graph_reference_id}', "
        f"direction='{params.direction}', "
        f"types_to_include='{params.relationship_types_to_include}'"
    )
    output_details: List[RelationshipData] = []

    if graphrag_context is None:
        logger.error("Relationship.OneHopNeighbors: graphrag_context IS NONE!")
        return RelationshipOneHopNeighborsOutputs(one_hop_relationships=output_details)

    graph_instance_from_context = graphrag_context.get_graph_instance(params.graph_reference_id)
    logger.info(
        f"Relationship.OneHopNeighbors: Attempting to use graph_id '{params.graph_reference_id}'. "
        f"Found in context: {graph_instance_from_context is not None}. "
        f"Type: {type(graph_instance_from_context)}"
    )

    if graph_instance_from_context is None:
        logger.error(
            f"Relationship.OneHopNeighbors: Graph instance for ID '{params.graph_reference_id}' "
            f"not found in context. Available graphs: {list(graphrag_context.graphs.keys())}"
        )
        return RelationshipOneHopNeighborsOutputs(one_hop_relationships=output_details)

    actual_nx_graph = None
    if (
        hasattr(graph_instance_from_context, "_graph")
        and hasattr(graph_instance_from_context._graph, "graph")
        and isinstance(graph_instance_from_context._graph.graph, nx.Graph)
    ):
        actual_nx_graph = graph_instance_from_context._graph.graph
        logger.info(
            "Relationship.OneHopNeighbors: Successfully accessed NetworkX graph "
            f"via _graph.graph. Type: {type(actual_nx_graph)}"
        )
    else:
        logger.error(
            "Relationship.OneHopNeighbors: Could not access a valid NetworkX graph "
            f"from graph instance '{params.graph_reference_id}'"
        )
        return RelationshipOneHopNeighborsOutputs(one_hop_relationships=output_details)

    nx_graph = actual_nx_graph
    is_directed_graph = hasattr(nx_graph, "successors") and hasattr(nx_graph, "predecessors")
    graph_type_description = "directed" if is_directed_graph else "undirected (using neighbors())"
    logger.info(f"Relationship.OneHopNeighbors: Graph is considered {graph_type_description}.")

    for entity_id in params.entity_ids:
        if not nx_graph.has_node(entity_id):
            logger.warning(
                f"Relationship.OneHopNeighbors: Entity ID '{entity_id}' not found in the graph. Skipping."
            )
            continue
        try:
            processed_neighbor_pairs = set()

            if params.direction in ["outgoing", "both"] or not is_directed_graph:
                iterator = nx_graph.successors(entity_id) if is_directed_graph else nx_graph.neighbors(entity_id)
                for neighbor_id in iterator:
                    pair_key = tuple(sorted((entity_id, neighbor_id)))
                    if not is_directed_graph and pair_key in processed_neighbor_pairs:
                        continue
                    edge_data_dict = nx_graph.get_edge_data(entity_id, neighbor_id)
                    if not edge_data_dict:
                        continue
                    items_to_process = (
                        edge_data_dict.items()
                        if isinstance(nx_graph, (nx.MultiGraph, nx.MultiDiGraph))
                        else [("single_edge", edge_data_dict)]
                    )
                    for _edge_key, attributes in items_to_process:
                        rel_name = attributes.get("relation_name", "unknown_relationship")
                        if (
                            params.relationship_types_to_include
                            and rel_name not in params.relationship_types_to_include
                        ):
                            continue
                        output_details.append(
                            RelationshipData(
                                source_id=attributes.get("source_id", "graph_traversal_tool"),
                                src_id=entity_id,
                                tgt_id=neighbor_id,
                                relation_name=str(rel_name),
                                description=(
                                    str(attributes.get("description"))
                                    if attributes.get("description") is not None
                                    else None
                                ),
                                weight=float(attributes.get("weight", 1.0)),
                                attributes={
                                    k: v
                                    for k, v in attributes.items()
                                    if k not in ["relation_name", "description", "weight"]
                                }
                                or None,
                            )
                        )
                    if not is_directed_graph:
                        processed_neighbor_pairs.add(pair_key)

            if is_directed_graph and params.direction in ["incoming", "both"]:
                for predecessor_id in nx_graph.predecessors(entity_id):
                    edge_data_dict = nx_graph.get_edge_data(predecessor_id, entity_id)
                    if not edge_data_dict:
                        continue
                    items_to_process = (
                        edge_data_dict.items()
                        if isinstance(nx_graph, nx.MultiDiGraph)
                        else [("single_edge", edge_data_dict)]
                    )
                    for _edge_key, attributes in items_to_process:
                        rel_name = str(attributes.get("relation_name", "unknown_relationship"))
                        if (
                            params.relationship_types_to_include
                            and rel_name not in params.relationship_types_to_include
                        ):
                            continue
                        if params.direction == "both" and any(
                            detail.src_id == predecessor_id
                            and detail.tgt_id == entity_id
                            and detail.relation_name == rel_name
                            for detail in output_details
                        ):
                            continue
                        output_details.append(
                            RelationshipData(
                                source_id=attributes.get("source_id", "graph_traversal_tool"),
                                src_id=predecessor_id,
                                tgt_id=entity_id,
                                relation_name=rel_name,
                                description=(
                                    str(attributes.get("description"))
                                    if attributes.get("description") is not None
                                    else None
                                ),
                                weight=float(attributes.get("weight", 1.0)),
                                attributes={
                                    k: v
                                    for k, v in attributes.items()
                                    if k not in ["relation_name", "description", "weight"]
                                }
                                or None,
                            )
                        )
        except Exception as e:
            logger.error(
                f"Relationship.OneHopNeighbors: Error processing entity '{entity_id}'. Error: {e}",
                exc_info=True,
            )

    logger.info(
        f"Relationship.OneHopNeighbors: Found {len(output_details)} one-hop relationships."
    )
    return RelationshipOneHopNeighborsOutputs(one_hop_relationships=output_details)


async def relationship_vdb_build_tool(
    params: RelationshipVDBBuildInputs,
    graphrag_context: GraphRAGContext,
) -> RelationshipVDBBuildOutputs:
    """Build/load a relationship FAISS index and register it in context."""
    logger.info(
        f"Building relationship VDB: graph_id='{params.graph_reference_id}', "
        f"collection='{params.vdb_collection_name}', fields={params.embedding_fields}"
    )

    try:
        graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
        if not graph_instance:
            error_msg = f"Graph '{params.graph_reference_id}' not found in context"
            logger.error(error_msg)
            return RelationshipVDBBuildOutputs(
                vdb_reference_id="",
                num_relationships_indexed=0,
                status=f"Error: {error_msg}",
            )

        if hasattr(graph_instance, "_graph") and hasattr(graph_instance._graph, "graph"):
            nx_graph = graph_instance._graph.graph
        elif hasattr(graph_instance, "graph"):
            nx_graph = graph_instance.graph
        else:
            nx_graph = graph_instance

        logger.info(
            f"Retrieved graph with {nx_graph.number_of_nodes()} nodes and "
            f"{nx_graph.number_of_edges()} edges"
        )

        # The caller already supplies the collection ID. Do not silently mutate it.
        vdb_id = params.vdb_collection_name
        existing_vdb = graphrag_context.get_vdb_instance(vdb_id)
        if existing_vdb and not params.force_rebuild:
            logger.info(f"VDB '{vdb_id}' already exists and force_rebuild=False; reusing it")
            return RelationshipVDBBuildOutputs(
                vdb_reference_id=vdb_id,
                num_relationships_indexed=nx_graph.number_of_edges(),
                status="VDB already exists",
            )

        embedding_provider = graphrag_context.embedding_provider
        if not embedding_provider:
            error_msg = "No embedding provider available in context"
            logger.error(error_msg)
            return RelationshipVDBBuildOutputs(
                vdb_reference_id="",
                num_relationships_indexed=0,
                status=f"Error: {error_msg}",
            )

        relationships_data = []
        edge_metadata = ["source", "target", "id"]
        if params.include_metadata:
            metadata_keys = set()
            for _u, _v, data in nx_graph.edges(data=True):
                metadata_keys.update(data.keys())
            edge_metadata.extend(list(metadata_keys - set(params.embedding_fields)))

        for u, v, edge_data in nx_graph.edges(data=True):
            content_parts = [
                f"{field}: {edge_data[field]}"
                for field in params.embedding_fields
                if field in edge_data and edge_data[field] is not None
            ]
            if not content_parts:
                content_parts.append(f"Relationship from {u} to {v}")

            rel_doc = {
                "id": edge_data.get("id", f"{u}->{v}"),
                "content": " | ".join(content_parts),
                "source": u,
                "target": v,
            }
            if params.include_metadata:
                for key, value in edge_data.items():
                    if key != "id" and key not in params.embedding_fields:
                        rel_doc[key] = value
            relationships_data.append(rel_doc)

        if not relationships_data:
            logger.warning(f"No relationships found in graph '{params.graph_reference_id}'")
            return RelationshipVDBBuildOutputs(
                vdb_reference_id="",
                num_relationships_indexed=0,
                status="No relationships found in graph",
            )

        config = create_faiss_index_config(
            persist_path=f"storage/vdb/{vdb_id}",
            embed_model=embedding_provider,
            name=vdb_id,
        )
        relationship_vdb = FaissIndex(config)
        build_ok = await relationship_vdb.build_index(
            elements=relationships_data,
            meta_data=edge_metadata,
            force=params.force_rebuild,
        )
        if not build_ok:
            error_msg = f"Relationship VDB '{vdb_id}' failed to build or load a usable index"
            logger.error(error_msg)
            return RelationshipVDBBuildOutputs(
                vdb_reference_id="",
                num_relationships_indexed=0,
                status=f"Error: {error_msg}",
            )

        graphrag_context.add_vdb_instance(vdb_id, relationship_vdb)
        if vdb_id not in graphrag_context.list_vdbs():
            error_msg = f"Relationship VDB '{vdb_id}' built but failed context registration"
            logger.error(error_msg)
            return RelationshipVDBBuildOutputs(
                vdb_reference_id="",
                num_relationships_indexed=0,
                status=f"Error: {error_msg}",
            )

        logger.info(
            f"Successfully built relationship VDB '{vdb_id}' with "
            f"{len(relationships_data)} relationships indexed"
        )
        return RelationshipVDBBuildOutputs(
            vdb_reference_id=vdb_id,
            num_relationships_indexed=len(relationships_data),
            status=f"Successfully built VDB with {len(relationships_data)} relationships",
        )

    except Exception as e:
        logger.error(f"Error building relationship VDB: {e}", exc_info=True)
        return RelationshipVDBBuildOutputs(
            vdb_reference_id="",
            num_relationships_indexed=0,
            status=f"Error: {e}",
        )


async def relationship_vdb_search_tool(
    params: RelationshipVDBSearchInputs,
    graphrag_context: GraphRAGContext,
) -> RelationshipVDBSearchOutputs:
    """Search a registered relationship VDB using its BaseIndex retrieval API."""
    logger.info(
        f"Executing tool 'Relationship.VDB.Search' with parameters: "
        f"vdb_reference_id='{params.vdb_reference_id}', "
        f"query_text='{params.query_text}', "
        f"has_embedding={params.query_embedding is not None}, top_k={params.top_k}"
    )

    if not params.query_text and not params.query_embedding:
        return RelationshipVDBSearchOutputs(
            similar_relationships=[],
            metadata={"error": "Either query_text or query_embedding must be provided"},
        )

    vdb_instance = graphrag_context.get_vdb_instance(params.vdb_reference_id)
    if vdb_instance is None:
        return RelationshipVDBSearchOutputs(
            similar_relationships=[],
            metadata={"error": f"VDB '{params.vdb_reference_id}' not found"},
        )

    if params.query_embedding is not None and not params.query_text:
        # FaissIndex currently exposes text retrieval, not direct embedding retrieval.
        return RelationshipVDBSearchOutputs(
            similar_relationships=[],
            metadata={"error": "Direct relationship query_embedding search is not implemented"},
        )

    try:
        results = await vdb_instance.retrieval(
            query=params.query_text,
            top_k=params.top_k,
        )
        similar_relationships = []
        for result in results:
            node = getattr(result, "node", None)
            metadata = getattr(node, "metadata", {}) or {}
            rel_id = str(metadata.get("id", getattr(node, "node_id", "unknown")))
            rel_desc = str(getattr(node, "text", "") or metadata.get("content", ""))
            score = float(result.score) if getattr(result, "score", None) is not None else 0.0
            if params.score_threshold is not None and score < params.score_threshold:
                continue
            similar_relationships.append((rel_id, rel_desc, score))

        similar_relationships.sort(key=lambda item: item[2], reverse=True)
        return RelationshipVDBSearchOutputs(
            similar_relationships=similar_relationships,
            metadata={
                "vdb_id": params.vdb_reference_id,
                "num_results": len(similar_relationships),
                "query_type": "text",
            },
        )
    except Exception as e:
        logger.error(f"Error searching relationship VDB: {e}", exc_info=True)
        return RelationshipVDBSearchOutputs(
            similar_relationships=[],
            metadata={"error": str(e)},
        )


async def relationship_agent_tool(
    params: RelationshipAgentInputs,
    graphrag_context: GraphRAGContext,
) -> RelationshipAgentOutputs:
    """Use an LLM to extract relationships from text context."""
    import json as _json

    logger.info(
        f"Executing Relationship.Agent: query='{params.query_text[:80]}', "
        f"{len(params.context_entities)} context entities"
    )

    text = params.text_context
    if isinstance(text, list):
        text = "\n\n".join(text)

    max_rels = params.max_relationships_to_extract or 10
    entity_names = [getattr(e, "entity_name", str(e)) for e in params.context_entities]
    entities_str = ", ".join(entity_names[:20])
    types_str = (
        ", ".join(params.target_relationship_types)
        if params.target_relationship_types
        else "any type"
    )

    prompt = f"""Extract relationships between entities from the following text.
Known entities: {entities_str}
Focus on relationship types: {types_str}
Return a JSON array of objects with fields: "src_id", "tgt_id", "relation_name", "description".
Extract at most {max_rels} relationships.

Query: {params.query_text}

Text:
{text[:4000]}

Return ONLY a JSON array. No other text."""

    llm = graphrag_context.llm_provider
    if llm is None:
        logger.error("Relationship.Agent: No LLM provider available")
        return RelationshipAgentOutputs(extracted_relationships=[])

    try:
        response = await llm.aask(prompt)
        resp_text = response.strip()
        if resp_text.startswith("```"):
            resp_text = resp_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        rels_raw = _json.loads(resp_text)
        if not isinstance(rels_raw, list):
            rels_raw = [rels_raw]

        extracted = []
        for r in rels_raw[:max_rels]:
            extracted.append(
                RelationshipData(
                    src_id=r.get("src_id", "unknown"),
                    tgt_id=r.get("tgt_id", "unknown"),
                    source_id="relationship_agent_tool",
                    relation_name=r.get("relation_name", "related_to"),
                    description=r.get("description", ""),
                )
            )

        logger.info(f"Relationship.Agent: Extracted {len(extracted)} relationships")
        return RelationshipAgentOutputs(extracted_relationships=extracted)

    except Exception as e:
        logger.error(f"Relationship.Agent: LLM extraction failed: {e}", exc_info=True)
        return RelationshipAgentOutputs(extracted_relationships=[])


async def relationship_score_aggregator_tool(
    params: RelationshipScoreAggregatorInputs,
    graphrag_context: GraphRAGContext,
) -> RelationshipScoreAggregatorOutputs:
    """Aggregate entity scores onto graph relationships and return the top results."""
    logger.info(
        f"Executing tool 'Relationship.ScoreAggregator' with "
        f"{len(params.entity_scores)} entity scores, "
        f"graph='{params.graph_reference_id}', method='{params.aggregation_method}'"
    )

    graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
    if graph_instance is None:
        logger.error(f"Relationship.ScoreAggregator: Graph '{params.graph_reference_id}' not found")
        return RelationshipScoreAggregatorOutputs(scored_relationships=[])

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
        logger.error("Relationship.ScoreAggregator: Could not access NetworkX graph")
        return RelationshipScoreAggregatorOutputs(scored_relationships=[])

    scored_relationships: List[Tuple[RelationshipData, float]] = []
    method = params.aggregation_method or "sum"

    for u, v, edge_data in nx_graph.edges(data=True):
        score_u = params.entity_scores.get(u, 0.0)
        score_v = params.entity_scores.get(v, 0.0)
        if score_u == 0.0 and score_v == 0.0:
            continue

        if method == "average":
            agg_score = (score_u + score_v) / 2.0
        elif method == "max":
            agg_score = max(score_u, score_v)
        else:
            agg_score = score_u + score_v

        rel = RelationshipData(
            src_id=u,
            tgt_id=v,
            source_id=edge_data.get("source_id", "score_aggregator"),
            relation_name=str(edge_data.get("relation_name", edge_data.get("type", "unknown"))),
            description=(
                str(edge_data.get("description", ""))
                if edge_data.get("description")
                else None
            ),
            weight=float(edge_data.get("weight", 1.0)),
        )
        scored_relationships.append((rel, agg_score))

    scored_relationships.sort(key=lambda item: item[1], reverse=True)
    top_k = params.top_k_relationships if params.top_k_relationships is not None else 10
    scored_relationships = scored_relationships[:top_k]

    logger.info(
        f"Relationship.ScoreAggregator: Returning {len(scored_relationships)} scored relationships"
    )
    return RelationshipScoreAggregatorOutputs(scored_relationships=scored_relationships)
