import logging
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    RelationshipOneHopNeighborsInputs,
    RelationshipOneHopNeighborsOutputs,
    RelationshipData,
    RelationshipScoreAggregatorInputs,
    RelationshipScoreAggregatorOutputs,
    RelationshipVDBSearchInputs,
    RelationshipVDBSearchOutputs,
    RelationshipAgentInputs,
    RelationshipAgentOutputs
)

logger = logging.getLogger(__name__)



async def relationship_one_hop_neighbors_tool(
    params: RelationshipOneHopNeighborsInputs, # Ensure imported
    graphrag_context: GraphRAGContext # Ensure imported
) -> RelationshipOneHopNeighborsOutputs: # Ensure imported
    logger.info(
        f"Executing tool 'Relationship.OneHopNeighbors' with parameters: "
        f"entity_ids={params.entity_ids}, "
        f"graph_reference_id='{params.graph_reference_id}', " # Log the ID it's looking for
        f"direction='{params.direction}', " 
        f"types_to_include='{params.relationship_types_to_include}'"
    )
    output_details: List[RelationshipData] = [] # Ensure RelationshipData is imported

    if graphrag_context is None: # Should not happen if orchestrator passes it
        logger.error("Relationship.OneHopNeighbors: graphrag_context IS NONE!")
        return RelationshipOneHopNeighborsOutputs(one_hop_relationships=output_details)

    # Use the get_graph_instance method from GraphRAGContext
    graph_instance_from_context = graphrag_context.get_graph_instance(params.graph_reference_id)
    
    logger.info(f"Relationship.OneHopNeighbors: Attempting to use graph_id '{params.graph_reference_id}'. Found in context: {graph_instance_from_context is not None}. Type: {type(graph_instance_from_context)}")

    if graph_instance_from_context is None:
        logger.error(f"Relationship.OneHopNeighbors: Graph instance for ID '{params.graph_reference_id}' not found in context. Available graphs: {list(graphrag_context.graphs.keys())}")
        return RelationshipOneHopNeighborsOutputs(one_hop_relationships=output_details)
    
    actual_nx_graph = None
    # Assuming graph_instance_from_context is an ERGraph (or similar BaseGraph wrapper)
    # and its _graph attribute is the NetworkXStorage instance, which holds the nx.Graph in its .graph attribute.
    if hasattr(graph_instance_from_context, '_graph') and \
       hasattr(graph_instance_from_context._graph, 'graph') and \
       isinstance(graph_instance_from_context._graph.graph, nx.Graph): # nx should be imported
        actual_nx_graph = graph_instance_from_context._graph.graph
        logger.info(f"Relationship.OneHopNeighbors: Successfully accessed NetworkX graph via _graph.graph. Type: {type(actual_nx_graph)}")
    else:
        logger.error(f"Relationship.OneHopNeighbors: Could not access a valid NetworkX graph from graph_instance_from_context._graph.graph. Graph object is: {graph_instance_from_context._graph if hasattr(graph_instance_from_context, '_graph') else 'No _graph attr'}")
        return RelationshipOneHopNeighborsOutputs(one_hop_relationships=output_details)

    nx_graph = actual_nx_graph # Use this for NetworkX operations
    
    # ... (the rest of the existing logic for iterating entities, finding neighbors, processing edges) ...
    # This part of your function (from the uploaded file) that processes based on direction,
    # gets edge_data, and appends to output_details, seems mostly correct, assuming nx_graph is valid.
    # Key is to ensure `edge_attr_for_relation_name` ('type') and `edge_attr_for_description` ('description')
    # match what ERGraph actually stores on edges.

    is_directed_graph = hasattr(nx_graph, 'successors') and hasattr(nx_graph, 'predecessors')
    graph_type_description = 'directed' if is_directed_graph else 'undirected (using neighbors())'
    logger.info(f"Relationship.OneHopNeighbors: Graph is considered {graph_type_description}.")

    for entity_id in params.entity_ids:
        if not nx_graph.has_node(entity_id):
            logger.warning(f"Relationship.OneHopNeighbors: Entity ID '{entity_id}' not found in the graph. Skipping.")
            continue
        try:
            edge_attr_for_relation_name = 'type' 
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
                            weight=float(attributes.get(edge_attr_for_weight, 1.0)), # Default weight to 1.0
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
                        if params.direction == "both": # Avoid duplicates if already processed as outgoing
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


async def relationship_score_aggregator_tool(
    params: RelationshipScoreAggregatorInputs,
    graphrag_context: GraphRAGContext
) -> RelationshipScoreAggregatorOutputs:
    logger.info(f"Executing tool 'Relationship.ScoreAggregator' with params: {params}")
    return RelationshipScoreAggregatorOutputs(aggregated_relationship_scores={"example_rel": 0.9})

async def relationship_vdb_search_tool(
    params: RelationshipVDBSearchInputs,
    graphrag_context: GraphRAGContext
) -> RelationshipVDBSearchOutputs:
    logger.info(f"Executing tool 'Relationship.VDBSearch' with params: {params}")
    return RelationshipVDBSearchOutputs(similar_relationships=[("rel1_id", "rel1_name", 0.8)])

async def relationship_agent_tool(
    params: RelationshipAgentInputs,
    graphrag_context: GraphRAGContext
) -> RelationshipAgentOutputs:
    logger.info(f"Executing tool 'Relationship.Agent' with params: {params}")
    return RelationshipAgentOutputs(determined_relationships=["agent_rel_id_1"])

async def relationship_agent_tool(
    params: RelationshipAgentInputs,
    graphrag_context: Optional[Any] = None # Placeholder for GraphRAG system context
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
            RelationshipData( # This should be the harmonized model
                src_id=entity_ids_for_dummy_rels[i % len(entity_ids_for_dummy_rels)],
                tgt_id=entity_ids_for_dummy_rels[(i+1) % len(entity_ids_for_dummy_rels)], # ensure different src/tgt for dummy
                source_id="llm_agent_relationship_tool",
                relation_name=rel_type,
                description=f"LLM identified relationship {i+1} of type {rel_type}",
                relevance_score=0.90 - (i*0.02) # Example additional field
                # Ensure all required fields from CoreRelationship are present
            )
        )

    return RelationshipAgentOutputs(extracted_relationships=dummy_extracted_relationships)


# --- Tool Implementation for: Relationship Vector Database Search ---
# tool_id: "Relationship.VDBSearch"

async def relationship_vdb_search_tool(
    params: RelationshipVDBSearchInputs,
    graphrag_context: Optional[Any] = None
) -> RelationshipVDBSearchOutputs:
    """
    Searches a vector database of relationships to find top-k most similar to a query.
    Wraps core GraphRAG logic.
    """
    print(f"Executing tool 'Relationship.VDBSearch' with parameters: {params}")

    # 1. Extract parameters from 'params: RelationshipVDBSearchInputs'
    #    - vdb_reference_id: str
    #    - query_text: Optional[str]
    #    - query_embedding: Optional[List[float]]
    #    - embedding_model_id: Optional[str]
    #    - top_k_results: int

    if not (params.query_text or params.query_embedding):
        raise ValueError("Either query_text or query_embedding must be provided for Relationship.VDBSearch.")

    print(f"Placeholder: Would search relationship VDB '{params.vdb_reference_id}' for query related to '{params.query_text}'.")

    # Dummy results
    dummy_similar_relationships = []
    for i in range(params.top_k_results):
        dummy_rel_data = RelationshipData(
            relationship_id=f"vdb_rel_{i+1}",
            src_id=f"src_node_vdb_rel_{i+1}",
            tgt_id=f"tgt_node_vdb_rel_{i+1}",
            source_id="relationship_vdb_tool",
            type="similar_to_query_by_vdb",
            description=f"Dummy relationship {i+1} found via VDB search.",
            relevance_score=0.9 - (i*0.05) # Already part of RelationshipData if harmonized
        )
        dummy_similar_relationships.append(
            (dummy_rel_data, 0.9 - (i*0.05)) # Tuple of (RelationshipData, score)
        )

    return RelationshipVDBSearchOutputs(similar_relationships=dummy_similar_relationships)


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
    # This assumes RelationshipData is the harmonized model
    example_rels = [
        RelationshipData(relationship_id=f"agg_rel_{i}", src_id=f"src_{i}", tgt_id=f"tgt_{i}", source_id="agg_tool", type="aggregated_score_type")
        for i in range(num_to_return)
    ]

    for i in range(num_to_return):
        dummy_scored_relationships.append(
            (example_rels[i], 0.8 - (i * 0.1)) # (RelationshipData, aggregated_score)
        )

    return RelationshipScoreAggregatorOutputs(scored_relationships=dummy_scored_relationships)
