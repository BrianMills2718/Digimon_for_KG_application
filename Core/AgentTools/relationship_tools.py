from typing import List, Tuple, Dict, Any, Optional
import networkx as nx
import logging
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
    RelationshipAgentOutputs
)
from Core.Index.FaissIndex import FaissIndex
from Core.Storage.PickleBlobStorage import PickleBlobStorage
from pydantic import BaseModel

# Import proper index configuration
from Core.Index.Schema import FAISSIndexConfig
from Core.AgentTools.index_config_helper import create_faiss_index_config

logger = logging.getLogger(__name__)


async def relationship_one_hop_neighbors_tool(
    params: RelationshipOneHopNeighborsInputs, 
    graphrag_context: GraphRAGContext 
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
    
    logger.info(f"Relationship.OneHopNeighbors: Attempting to use graph_id '{params.graph_reference_id}'. Found in context: {graph_instance_from_context is not None}. Type: {type(graph_instance_from_context)}")

    if graph_instance_from_context is None:
        logger.error(f"Relationship.OneHopNeighbors: Graph instance for ID '{params.graph_reference_id}' not found in context. Available graphs: {list(graphrag_context.graphs.keys())}")
        return RelationshipOneHopNeighborsOutputs(one_hop_relationships=output_details)
    
    actual_nx_graph = None
    if hasattr(graph_instance_from_context, '_graph') and \
       hasattr(graph_instance_from_context._graph, 'graph') and \
       isinstance(graph_instance_from_context._graph.graph, nx.Graph): 
        actual_nx_graph = graph_instance_from_context._graph.graph
        logger.info(f"Relationship.OneHopNeighbors: Successfully accessed NetworkX graph via _graph.graph. Type: {type(actual_nx_graph)}")
    else:
        logger.error(f"Relationship.OneHopNeighbors: Could not access a valid NetworkX graph from graph_instance_from_context._graph.graph. Graph object is: {graph_instance_from_context._graph if hasattr(graph_instance_from_context, '_graph') else 'No _graph attr'}")
        return RelationshipOneHopNeighborsOutputs(one_hop_relationships=output_details)

    nx_graph = actual_nx_graph 
    
    is_directed_graph = hasattr(nx_graph, 'successors') and hasattr(nx_graph, 'predecessors')
    graph_type_description = 'directed' if is_directed_graph else 'undirected (using neighbors())'
    logger.info(f"Relationship.OneHopNeighbors: Graph is considered {graph_type_description}.")

    for entity_id in params.entity_ids:
        if not nx_graph.has_node(entity_id):
            logger.warning(f"Relationship.OneHopNeighbors: Entity ID '{entity_id}' not found in the graph. Skipping.")
            continue
        try:
            edge_attr_for_relation_name = 'relation_name'  
            edge_attr_for_description = 'description'
            edge_attr_for_weight = 'weight'
            processed_neighbor_pairs = set()

            if params.direction in ["outgoing", "both"] or not is_directed_graph:
                iterator = nx_graph.successors(entity_id) if is_directed_graph else nx_graph.neighbors(entity_id)
                for neighbor_id in iterator:
                    if not is_directed_graph and tuple(sorted((entity_id, neighbor_id))) in processed_neighbor_pairs:
                        continue
                    edge_data_dict = nx_graph.get_edge_data(entity_id, neighbor_id)
                    if not edge_data_dict: continue
                    items_to_process = edge_data_dict.items() if isinstance(nx_graph, (nx.MultiGraph, nx.MultiDiGraph)) else [("single_edge", edge_data_dict)]
                    for _edge_key, attributes in items_to_process:
                        rel_name_from_edge = attributes.get(edge_attr_for_relation_name, "unknown_relationship")
                        if params.relationship_types_to_include and rel_name_from_edge not in params.relationship_types_to_include:
                            continue
                        output_details.append(RelationshipData(
                            source_id="graph_traversal_tool", 
                            src_id=entity_id,      
                            tgt_id=neighbor_id,    
                            relation_name=str(rel_name_from_edge) if rel_name_from_edge is not None else "unknown_relationship",
                            description=str(attributes.get(edge_attr_for_description)) if attributes.get(edge_attr_for_description) is not None else None,
                            weight=float(attributes.get(edge_attr_for_weight, 1.0)), 
                            attributes={k: v for k, v in attributes.items() if k not in [edge_attr_for_relation_name, edge_attr_for_description, edge_attr_for_weight]} or None
                        ))
                    if not is_directed_graph: processed_neighbor_pairs.add(tuple(sorted((entity_id, neighbor_id))))

            if is_directed_graph and params.direction in ["incoming", "both"]:
                for predecessor_id in nx_graph.predecessors(entity_id):
                    edge_data_dict = nx_graph.get_edge_data(predecessor_id, entity_id)
                    if not edge_data_dict: continue
                    items_to_process_incoming = edge_data_dict.items() if isinstance(nx_graph, nx.MultiDiGraph) else [("single_edge", edge_data_dict)]
                    for _edge_key, attributes in items_to_process_incoming:
                        rel_name_val_from_edge = str(attributes.get(edge_attr_for_relation_name, "unknown_relationship"))
                        if params.relationship_types_to_include and rel_name_val_from_edge not in params.relationship_types_to_include:
                            continue
                        is_already_processed_as_outgoing = False
                        if params.direction == "both": 
                            for detail in output_details:
                                if detail.src_id == predecessor_id and detail.tgt_id == entity_id and detail.relation_name == rel_name_val_from_edge:
                                    is_already_processed_as_outgoing = True; break
                        if is_already_processed_as_outgoing: continue
                        output_details.append(RelationshipData(
                            source_id="graph_traversal_tool",
                            src_id=predecessor_id,      
                            tgt_id=entity_id,          
                            relation_name=rel_name_val_from_edge, 
                            description=str(attributes.get(edge_attr_for_description)) if attributes.get(edge_attr_for_description) is not None else None,
                            weight=float(attributes.get(edge_attr_for_weight, 1.0)),
                            attributes={k: v for k, v in attributes.items() if k not in [edge_attr_for_relation_name, edge_attr_for_description, edge_attr_for_weight]} or None
                        ))       
        except Exception as e: 
            logger.error(f"Relationship.OneHopNeighbors: Error processing entity '{entity_id}'. Error: {e}", exc_info=True)

    logger.info(f"Relationship.OneHopNeighbors: Found {len(output_details)} one-hop relationships.")
    return RelationshipOneHopNeighborsOutputs(one_hop_relationships=output_details)


async def relationship_vdb_build_tool(
    params: RelationshipVDBBuildInputs,
    graphrag_context: GraphRAGContext
) -> RelationshipVDBBuildOutputs:
    """
    Build a vector database (VDB) for graph relationships.
    
    This tool creates a searchable index of relationships from a graph,
    allowing for similarity-based retrieval of edges based on their
    properties and descriptions.
    """
    logger.info(
        f"Building relationship VDB: graph_id='{params.graph_reference_id}', "
        f"collection='{params.vdb_collection_name}', fields={params.embedding_fields}"
    )
    
    try:
        # Get the graph instance
        graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
        if not graph_instance:
            error_msg = f"Graph '{params.graph_reference_id}' not found in context"
            logger.error(error_msg)
            return RelationshipVDBBuildOutputs(
                vdb_reference_id="",
                num_relationships_indexed=0,
                status=f"Error: {error_msg}"
            )
        
        # Extract the actual NetworkX graph
        if hasattr(graph_instance, '_graph') and hasattr(graph_instance._graph, 'graph'):
            nx_graph = graph_instance._graph.graph
        elif hasattr(graph_instance, 'graph'):
            nx_graph = graph_instance.graph
        else:
            nx_graph = graph_instance
            
        logger.info(f"Retrieved graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
        
        # Check if VDB already exists
        vdb_id = f"{params.vdb_collection_name}_relationships"
        existing_vdb = graphrag_context.get_vdb_instance(vdb_id)
        
        if existing_vdb and not params.force_rebuild:
            logger.info(f"VDB '{vdb_id}' already exists and force_rebuild=False, skipping build")
            return RelationshipVDBBuildOutputs(
                vdb_reference_id=vdb_id,
                num_relationships_indexed=nx_graph.number_of_edges(),
                status="VDB already exists"
            )
        
        # Get embedding provider
        embedding_provider = graphrag_context.embedding_provider
        if not embedding_provider:
            error_msg = "No embedding provider available in context"
            logger.error(error_msg)
            return RelationshipVDBBuildOutputs(
                vdb_reference_id="",
                num_relationships_indexed=0,
                status=f"Error: {error_msg}"
            )
        
        # Prepare relationship data
        relationships_data = []
        edge_metadata = ["source", "target", "id"]
        
        if params.include_metadata:
            # Collect all possible metadata keys from edges
            metadata_keys = set()
            for u, v, data in nx_graph.edges(data=True):
                metadata_keys.update(data.keys())
            edge_metadata.extend(list(metadata_keys - set(params.embedding_fields)))
        
        # Extract edges and their properties
        for u, v, edge_data in nx_graph.edges(data=True):
            # Create text content from embedding fields
            content_parts = []
            for field in params.embedding_fields:
                if field in edge_data:
                    content_parts.append(f"{field}: {edge_data[field]}")
            
            if not content_parts:
                # If no embedding fields found, use a default description
                content_parts.append(f"Relationship from {u} to {v}")
            
            content = " | ".join(content_parts)
            
            # Create relationship document
            rel_doc = {
                "id": edge_data.get("id", f"{u}->{v}"),
                "content": content,
                "source": u,
                "target": v
            }
            
            # Add metadata if requested
            if params.include_metadata:
                for key, value in edge_data.items():
                    if key not in ["id"] and key not in params.embedding_fields:
                        rel_doc[key] = value
            
            relationships_data.append(rel_doc)
        
        if not relationships_data:
            logger.warning(f"No relationships found in graph '{params.graph_reference_id}'")
            return RelationshipVDBBuildOutputs(
                vdb_reference_id="",
                num_relationships_indexed=0,
                status="No relationships found in graph"
            )
        
        logger.info(f"Prepared {len(relationships_data)} relationships for indexing")
        
        # Create VDB storage path
        vdb_storage_path = f"storage/vdb/{vdb_id}"
        
        # Create index configuration using proper schema
        config = create_faiss_index_config(
            persist_path=vdb_storage_path,
            embed_model=embedding_provider,
            name=vdb_id
        )
        
        # Create and build the index
        relationship_vdb = FaissIndex(config)
        
        # Build the index
        await relationship_vdb.build_index(
            elements=relationships_data,
            meta_data=edge_metadata,
            force=params.force_rebuild
        )
        
        # Register the VDB in context
        graphrag_context.add_vdb_instance(vdb_id, relationship_vdb)
        
        logger.info(
            f"Successfully built relationship VDB '{vdb_id}' with "
            f"{len(relationships_data)} relationships indexed"
        )
        
        return RelationshipVDBBuildOutputs(
            vdb_reference_id=vdb_id,
            num_relationships_indexed=len(relationships_data),
            status=f"Successfully built VDB with {len(relationships_data)} relationships"
        )
        
    except Exception as e:
        error_msg = f"Error building relationship VDB: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return RelationshipVDBBuildOutputs(
            vdb_reference_id="",
            num_relationships_indexed=0,
            status=f"Error: {str(e)}"
        )


async def relationship_score_aggregator_tool(
    params: RelationshipScoreAggregatorInputs,
    graphrag_context: Optional[Any] = None
) -> RelationshipScoreAggregatorOutputs:
    logger.info(f"Executing tool 'Relationship.ScoreAggregator' with params: {params}")
    return RelationshipScoreAggregatorOutputs(aggregated_relationship_scores={"example_rel": 0.9})

async def relationship_vdb_search_tool(
    params: RelationshipVDBSearchInputs,
    graphrag_context: GraphRAGContext
) -> RelationshipVDBSearchOutputs:
    """
    Search for similar relationships in a vector database.
    
    Args:
        params: Search parameters including VDB ID, query, and result limits
        graphrag_context: Context containing VDB instances
    
    Returns:
        Output with similar relationships and scores
    """
    logger.info(
        f"Executing tool 'Relationship.VDB.Search' with parameters: "
        f"vdb_reference_id='{params.vdb_reference_id}', "
        f"query_text='{params.query_text}', "
        f"has_embedding={params.query_embedding is not None}, "
        f"top_k={params.top_k}"
    )
    
    # Validate input
    if not params.query_text and not params.query_embedding:
        logger.error("Either query_text or query_embedding must be provided")
        return RelationshipVDBSearchOutputs(
            similar_relationships=[],
            metadata={"error": "Either query_text or query_embedding must be provided"}
        )
    
    # Get VDB instance
    vdb_instance = graphrag_context.get_vdb_instance(params.vdb_reference_id)
    if vdb_instance is None:
        logger.error(f"VDB '{params.vdb_reference_id}' not found in context")
        return RelationshipVDBSearchOutputs(
            similar_relationships=[],
            metadata={"error": f"VDB '{params.vdb_reference_id}' not found"}
        )
    
    try:
        # Perform search
        if params.query_text:
            logger.info(f"Searching with text query: '{params.query_text}'")
            results = await vdb_instance.search(
                query=params.query_text,
                k=params.top_k
            )
        else:
            logger.info("Searching with pre-computed embedding")
            results = await vdb_instance.search_by_embedding(
                embedding=params.query_embedding,
                k=params.top_k
            )
        
        # Process results
        similar_relationships = []
        for result in results:
            # Extract relationship info from result
            rel_id = result.get('id', 'unknown')
            rel_desc = result.get('content', '')
            similarity_score = result.get('score', 0.0)
            
            # Apply score threshold if specified
            if params.score_threshold and similarity_score < params.score_threshold:
                continue
            
            similar_relationships.append((rel_id, rel_desc, float(similarity_score)))
        
        # Sort by score (highest first)
        similar_relationships.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Found {len(similar_relationships)} similar relationships")
        
        return RelationshipVDBSearchOutputs(
            similar_relationships=similar_relationships,
            metadata={
                "vdb_id": params.vdb_reference_id,
                "num_results": len(similar_relationships),
                "query_type": "text" if params.query_text else "embedding"
            }
        )
        
    except Exception as e:
        logger.error(f"Error searching relationship VDB: {e}", exc_info=True)
        return RelationshipVDBSearchOutputs(
            similar_relationships=[],
            metadata={"error": str(e)}
        )

async def relationship_agent_tool(
    params: RelationshipAgentInputs,
    graphrag_context: Optional[Any] = None 
) -> RelationshipAgentOutputs:
    """
    Utilizes an LLM to find or extract useful relationships from the given context.
    Wraps core GraphRAG logic for LLM-based relationship extraction.
    """
    print(f"Executing tool 'Relationship.Agent' with parameters: {params}")

    # 1. Extract parameters from 'params: RelationshipAgentInputs'
    #    - query_text: str
    #    - text_context: Union[str, List[str]]
    #    - context_entities: List[ExtractedEntityData] (or their IDs)
    #    - target_relationship_types: Optional[List[str]]
    #    - llm_config_override_patch: Optional[Dict[str, Any]]
    #    - max_relationships_to_extract: Optional[int]

    # Placeholder: Prepare prompt for LLM.
    # Placeholder: Get LLM instance.
    # Placeholder: Make LLM call.
    # Placeholder: Parse LLM response into List[RelationshipData].

    print(f"Placeholder: Would use LLM to extract relationships from context '{params.text_context}' based on query '{params.query_text}'.")
    print(f"Context entities provided: {len(params.context_entities)}")
    print(f"Target relationship types: {params.target_relationship_types}")

    # Dummy results
    dummy_extracted_relationships = []
    num_to_extract = params.max_relationships_to_extract if params.max_relationships_to_extract is not None else 1

    # Use context_entities if available to make more meaningful dummy data
    entity_ids_for_dummy_rels = [getattr(e, 'entity_name', f"dummy_entity_{idx}") for idx, e in enumerate(params.context_entities)]
    if not entity_ids_for_dummy_rels:
        entity_ids_for_dummy_rels = ["dummy_src_1", "dummy_tgt_1"]
    if len(entity_ids_for_dummy_rels) == 1:
        entity_ids_for_dummy_rels.append(f"another_entity_for_{entity_ids_for_dummy_rels[0]}")


    for i in range(num_to_extract):
        rel_type = params.target_relationship_types[0] if params.target_relationship_types else "related_to_by_llm"
        dummy_extracted_relationships.append(
            RelationshipData( 
                src_id=entity_ids_for_dummy_rels[i % len(entity_ids_for_dummy_rels)],
                tgt_id=entity_ids_for_dummy_rels[(i+1) % len(entity_ids_for_dummy_rels)], 
                source_id="llm_agent_relationship_tool",
                relation_name=rel_type,
                description=f"LLM identified relationship {i+1} of type {rel_type}",
                relevance_score=0.90 - (i*0.02) 
                # Ensure all required fields from CoreRelationship are present
            )
        )

    return RelationshipAgentOutputs(extracted_relationships=dummy_extracted_relationships)


# --- Tool Implementation for: Relationship Score Aggregator ---
# tool_id: "Relationship.ScoreAggregator"

from Core.AgentSchema.tool_contracts import RelationshipScoreAggregatorInputs, RelationshipScoreAggregatorOutputs
from typing import Dict, List, Optional, Any, Tuple, Literal

async def relationship_score_aggregator_tool(
    params: RelationshipScoreAggregatorInputs,
    graphrag_context: Optional[Any] = None
) -> RelationshipScoreAggregatorOutputs:
    """
    Computes relationship scores based on entity scores (e.g., from PPR) and returns top-k relationships.
    Wraps core GraphRAG logic.
    """
    print(f"Executing tool 'Relationship.ScoreAggregator' with parameters: {params}")

    # 1. Extract parameters from 'params: RelationshipScoreAggregatorInputs'
    #    - entity_scores: Dict[str, float]
    #    - graph_reference_id: str
    #    - top_k_relationships: Optional[int]
    #    - aggregation_method: Optional[Literal["sum", "average", "max"]]

    # Placeholder: Access graph, get relationships for entities in entity_scores,
    # aggregate scores onto relationships, rank them.
    print(f"Placeholder: Would aggregate scores for relationships in graph '{params.graph_reference_id}' using {len(params.entity_scores)} entity scores.")

    # Dummy results
    dummy_scored_relationships = []
    num_to_return = params.top_k_relationships if params.top_k_relationships is not None else 2

    # Create some dummy RelationshipData if needed for the output structure
    example_rels = [
        RelationshipData(relationship_id=f"agg_rel_{i}", src_id=f"src_{i}", tgt_id=f"tgt_{i}", source_id="agg_tool", type="aggregated_score_type")
        for i in range(num_to_return)
    ]

    for i in range(num_to_return):
        dummy_scored_relationships.append(
            (example_rels[i], 0.8 - (i * 0.1)) 
        )

    return RelationshipScoreAggregatorOutputs(scored_relationships=dummy_scored_relationships)
