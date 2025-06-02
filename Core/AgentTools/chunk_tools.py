# Core/AgentTools/chunk_tools.py

"""
Chunk Tools for GraphRAG

This module implements chunk-related operators for the GraphRAG system.
"""

from typing import List, Tuple, Optional, Any, Union, Dict
from pydantic import BaseModel, Field
import networkx as nx
from Core.Common.Logger import logger

from Core.AgentSchema.tool_contracts import (
    ChunkFromRelationshipsInputs, 
    ChunkFromRelationshipsOutputs,
    ChunkData
)
from Core.AgentSchema.context import GraphRAGContext

# --- Tool Implementation for: Chunks From Relationships ---
# tool_id: "Chunk.FromRelationships"

def chunk_from_relationships(
    input_data: Dict[str, Any],
    context: GraphRAGContext
) -> Dict[str, Any]:
    """
    Retrieves text chunks associated with specified relationships.
    
    Args:
        input_data: Dictionary containing ChunkFromRelationshipsInputs fields
        context: GraphRAGContext containing graph instances and chunk data
        
    Returns:
        Dictionary containing ChunkFromRelationshipsOutputs fields
    """
    try:
        # Validate input
        validated_input = ChunkFromRelationshipsInputs(**input_data)
    except Exception as e:
        logger.error(f"Failed to validate Chunk.FromRelationships input: {e}")
        return {
            "relevant_chunks": []
        }
    
    target_relationships = validated_input.target_relationships
    document_collection_id = validated_input.document_collection_id
    max_chunks_per_relationship = validated_input.max_chunks_per_relationship
    top_k_total = validated_input.top_k_total
    
    # Use document_collection_id as graph_id (they represent the same concept in this context)
    graph_id = document_collection_id
    
    # Get graph instance
    graph_instance = context.get_graph_instance(graph_id)
    if graph_instance is None:
        logger.warning(f"Graph '{graph_id}' not found in context")
        return {
            "relevant_chunks": []
        }
    
    # Extract NetworkX graph
    graph = None
    logger.debug(f"Graph instance type: {type(graph_instance)}")
    logger.debug(f"Graph instance attrs: {dir(graph_instance) if graph_instance else 'None'}")
    
    if hasattr(graph_instance, '_graph'):
        storage = graph_instance._graph
        logger.debug(f"Storage type from _graph: {type(storage)}")
        if isinstance(storage, nx.Graph):
            graph = storage
        elif hasattr(storage, 'graph') and isinstance(storage.graph, nx.Graph):
            graph = storage.graph
        elif hasattr(storage, '_graph'):
            graph = storage._graph
    elif hasattr(graph_instance, 'graph') and isinstance(graph_instance.graph, nx.Graph):
        graph = graph_instance.graph
    elif isinstance(graph_instance, nx.Graph):
        graph = graph_instance
    
    logger.debug(f"Extracted graph type: {type(graph)}")
    
    if graph is None:
        logger.error(f"Could not extract NetworkX graph from graph instance '{graph_id}'")
        return {
            "relevant_chunks": []
        }
    
    # Collect chunks from relationships
    all_chunks = []
    chunks_by_relationship = {}
    
    for rel_spec in target_relationships:
        relationship_id = None
        
        # Handle different relationship specification formats
        if isinstance(rel_spec, str):
            relationship_id = rel_spec
        elif isinstance(rel_spec, dict):
            # Could have keys like 'id', 'source->target', etc.
            if 'id' in rel_spec:
                relationship_id = rel_spec['id']
            elif 'relationship_id' in rel_spec:
                relationship_id = rel_spec['relationship_id']
            elif 'source' in rel_spec and 'target' in rel_spec:
                relationship_id = f"{rel_spec['source']}->{rel_spec['target']}"
        
        if not relationship_id:
            logger.warning(f"Could not extract relationship ID from: {rel_spec}")
            continue
        
        # Find chunks associated with this relationship
        relationship_chunks = []
        
        # First, try to find edges by relationship ID
        edge_found = False
        for u, v, edge_data in graph.edges(data=True):
            edge_id = None
            
            # Try different possible ID fields
            if 'id' in edge_data:
                edge_id = edge_data['id']
            elif 'relationship_id' in edge_data:
                edge_id = edge_data['relationship_id']
            elif 'rel_id' in edge_data:
                edge_id = edge_data['rel_id']
            else:
                # Use a composite key as fallback
                edge_id = f"{u}->{v}"
            
            if edge_id == relationship_id:
                edge_found = True
                
                # Check if edge has associated chunks
                if 'chunks' in edge_data:
                    # Chunks might be stored as list of chunk IDs or chunk data
                    chunks_data = edge_data['chunks']
                    
                    if isinstance(chunks_data, list):
                        for idx, chunk in enumerate(chunks_data):
                            if isinstance(chunk, str):
                                # It's a chunk ID - create basic chunk data
                                chunk_obj = {
                                    "chunk_id": chunk,
                                    "text": f"[Chunk content for {chunk}]",  # Placeholder
                                    "doc_id": f"doc_{relationship_id}",
                                    "index": idx,
                                    "tokens": 0,
                                    "metadata": {
                                        "relationship_id": relationship_id,
                                        "source": u,
                                        "target": v
                                    }
                                }
                            elif isinstance(chunk, dict):
                                # It's already chunk data
                                chunk_obj = chunk.copy()
                                chunk_obj.setdefault("metadata", {})["relationship_id"] = relationship_id
                            else:
                                continue
                            
                            relationship_chunks.append(chunk_obj)
                
                # Also check for chunk references in other edge attributes
                if 'text' in edge_data:
                    # Edge might have direct text content
                    chunk_obj = {
                        "chunk_id": f"{relationship_id}_text",
                        "text": edge_data['text'],
                        "doc_id": f"edge_{relationship_id}",
                        "index": 0,
                        "tokens": len(edge_data['text'].split()),
                        "metadata": {
                            "relationship_id": relationship_id,
                            "source": u,
                            "target": v
                        }
                    }
                    relationship_chunks.append(chunk_obj)
                
                # Check source and target nodes for associated chunks
                for node_id in [u, v]:
                    if node_id in graph:
                        node_data = graph.nodes[node_id]
                        if 'chunks' in node_data:
                            node_chunks = node_data['chunks']
                            if isinstance(node_chunks, list):
                                for idx, chunk in enumerate(node_chunks):
                                    if isinstance(chunk, str):
                                        chunk_obj = {
                                            "chunk_id": chunk,
                                            "text": f"[Chunk content for {chunk} from node {node_id}]",
                                            "doc_id": f"node_{node_id}",
                                            "index": idx,
                                            "tokens": 0,
                                            "metadata": {
                                                "relationship_id": relationship_id,
                                                "node_id": node_id,
                                                "source": u,
                                                "target": v
                                            }
                                        }
                                    elif isinstance(chunk, dict):
                                        chunk_obj = chunk.copy()
                                        chunk_obj.setdefault("metadata", {}).update({
                                            "relationship_id": relationship_id,
                                            "node_id": node_id
                                        })
                                    else:
                                        continue
                                    
                                    relationship_chunks.append(chunk_obj)
        
        # Apply per-relationship limit if specified
        if max_chunks_per_relationship and len(relationship_chunks) > max_chunks_per_relationship:
            relationship_chunks = relationship_chunks[:max_chunks_per_relationship]
        
        chunks_by_relationship[relationship_id] = relationship_chunks
        all_chunks.extend(relationship_chunks)
    
    # Apply global limit if specified
    if top_k_total and len(all_chunks) > top_k_total:
        # Sort by some relevance criteria if available
        # For now, just take the first top_k
        all_chunks = all_chunks[:top_k_total]
    
    # Convert to ChunkData objects
    chunk_data_list = []
    for chunk_dict in all_chunks:
        try:
            # ChunkData expects specific fields; map our internal representation
            chunk_data = ChunkData(
                tokens=chunk_dict.get("tokens", 0),
                chunk_id=chunk_dict.get("chunk_id", "unknown"),
                content=chunk_dict.get("text", ""),  # Map 'text' to 'content'
                doc_id=chunk_dict.get("doc_id", "unknown"),
                index=chunk_dict.get("index", 0),
                title=chunk_dict.get("title")
            )
            # Set metadata after creation
            if "metadata" in chunk_dict and chunk_dict["metadata"]:
                chunk_data.metadata = chunk_dict["metadata"]
            chunk_data_list.append(chunk_data)
        except Exception as e:
            logger.warning(f"Failed to create ChunkData: {e}")
            logger.debug(f"Chunk dict: {chunk_dict}")
    
    # Create result message
    if not chunk_data_list:
        message = f"No chunks found for relationships: {target_relationships}"
    else:
        message = f"Found {len(chunk_data_list)} chunks from {len(chunks_by_relationship)} relationships"
    
    logger.info(f"Chunk.FromRelationships: {message}")
    
    # Create output following the contract
    result = {
        "relevant_chunks": chunk_data_list
    }
    
    return result


# Async wrapper for compatibility with async orchestrator
async def chunk_from_relationships_tool(
    input_data: Dict[str, Any],
    context: GraphRAGContext
) -> Dict[str, Any]:
    """Async wrapper for chunk_from_relationships."""
    return chunk_from_relationships(input_data, context)


# --- Tool Implementation for: Chunk Operator - Occurrence ---
# tool_id: "Chunk.Occurrence"

from Core.AgentSchema.tool_contracts import ChunkOccurrenceInputs, ChunkOccurrenceOutputs, ChunkData
from Core.AgentSchema.context import GraphRAGContext

async def chunk_occurrence_tool(
    params: ChunkOccurrenceInputs,
    graphrag_context: Optional[Any] = None
) -> ChunkOccurrenceOutputs:
    """
    Ranks text chunks based on the co-occurrence of specified entity pairs (representing relationships).
    Wraps core GraphRAG logic.
    """
    print(f"Executing tool 'Chunk.Occurrence' with parameters: {params}")

    # 1. Extract parameters from 'params: ChunkOccurrenceInputs'
    #    - target_entity_pairs_in_relationship: List[Dict[str, str]]
    #    - document_collection_id: str
    #    - top_k_chunks: int

    # Placeholder: Access chunk data. For each chunk, check for co-occurrence of entity pairs.
    # Rank chunks based on these occurrences.
    print(f"Placeholder: Would rank chunks from '{params.document_collection_id}' for co-occurrence of {len(params.target_entity_pairs_in_relationship)} entity pairs.")

    # Dummy results
    dummy_ranked_chunks = []
    for i in range(params.top_k_chunks):
        pair_info = params.target_entity_pairs_in_relationship[0] if params.target_entity_pairs_in_relationship else {"entity1_id": "e1", "entity2_id": "e2"}
        dummy_ranked_chunks.append(
            ChunkData(
                chunk_id=f"occurrence_chunk_{i+1}",
                content=f"Chunk {i+1} discussing entities {pair_info.get('entity1_id')} and {pair_info.get('entity2_id')}",
                doc_id=f"doc_for_occurrence_{i+1}",
                index=i,
                tokens=0, # Placeholder
                relevance_score=0.9 - (i*0.05) # Example score
            )
        )

    return ChunkOccurrenceOutputs(ranked_occurrence_chunks=dummy_ranked_chunks)
