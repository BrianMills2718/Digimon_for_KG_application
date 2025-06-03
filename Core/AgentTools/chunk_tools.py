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
    ChunkData,
    ChunkGetTextForEntitiesInput,
    ChunkGetTextForEntitiesOutput,
    ChunkTextResultItem
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


# --- Tool Implementation for: Chunk.GetTextForEntities ---
# tool_id: "Chunk.GetTextForEntities"

async def chunk_get_text_for_entities_tool(
    params: Union[Dict[str, Any], ChunkGetTextForEntitiesInput],
    context: GraphRAGContext
) -> Dict[str, Any]:
    """
    Get text chunks associated with specific entities.
    
    This tool retrieves the original text chunks that mention or are associated
    with the given entity IDs.
    """
    logger.info(f"Executing Chunk.GetTextForEntities with params: {params}")
    
    try:
        # Handle both dict and Pydantic model inputs
        if isinstance(params, dict):
            validated_input = ChunkGetTextForEntitiesInput(**params)
        else:
            # params is already a Pydantic model instance
            validated_input = params
        
        # Get graph instance
        graph_instance = context.get_graph_instance(validated_input.graph_reference_id)
        if not graph_instance:
            error_msg = f"Graph '{validated_input.graph_reference_id}' not found in context"
            logger.error(error_msg)
            return {
                "retrieved_chunks": [],
                "status_message": f"Error: {error_msg}"
            }
        
        # Extract NetworkX graph - try different access patterns
        nx_graph = None
        try:
            # Try _graph.graph pattern (as seen in relationship_tools.py)
            if hasattr(graph_instance, '_graph') and hasattr(graph_instance._graph, 'graph'):
                nx_graph = graph_instance._graph.graph
                logger.debug(f"Accessed NetworkX graph via _graph.graph pattern")
            # Try direct .graph pattern
            elif hasattr(graph_instance, 'graph'):
                nx_graph = graph_instance.graph
                logger.debug(f"Accessed NetworkX graph via direct .graph pattern")
            else:
                raise AttributeError("Could not find NetworkX graph in graph instance")
        except Exception as e:
            logger.error(f"Failed to extract NetworkX graph: {e}")
            return {
                "retrieved_chunks": [],
                "status_message": f"Error extracting graph: {str(e)}"
            }
        
        retrieved_chunks_data = []
        chunks_per_entity = {}
        
        # Process each entity
        for entity_id in validated_input.entity_ids:
            if entity_id not in nx_graph:
                logger.warning(f"Entity '{entity_id}' not found in graph")
                continue
                
            # Get node data
            node_data = nx_graph.nodes[entity_id]
            logger.debug(f"Processing entity '{entity_id}' with node data keys: {list(node_data.keys())}")
            
            # Look for chunk associations in node data
            # Common patterns: 'chunk_id', 'source_chunk_id', 'chunk_ids', 'source_chunks', 'source_id'
            chunk_ids = []
            
            # Single chunk ID
            if 'chunk_id' in node_data:
                chunk_ids.append(node_data['chunk_id'])
            elif 'source_chunk_id' in node_data:
                chunk_ids.append(node_data['source_chunk_id'])
            elif 'source_id' in node_data:
                # Handle source_id which may contain chunk references
                source_id = node_data['source_id']
                if isinstance(source_id, str):
                    # May be a single chunk ID or multiple separated by <SEP>
                    if '<SEP>' in source_id:
                        chunk_ids.extend(source_id.split('<SEP>'))
                    else:
                        chunk_ids.append(source_id)
            
            # Multiple chunk IDs
            if 'chunk_ids' in node_data:
                if isinstance(node_data['chunk_ids'], list):
                    chunk_ids.extend(node_data['chunk_ids'])
                else:
                    chunk_ids.append(node_data['chunk_ids'])
            elif 'source_chunks' in node_data:
                if isinstance(node_data['source_chunks'], list):
                    chunk_ids.extend(node_data['source_chunks'])
                else:
                    chunk_ids.append(node_data['source_chunks'])
            
            # Also check edges for chunk associations
            # Some graphs store entity-chunk relationships as edges
            for neighbor in nx_graph.neighbors(entity_id):
                neighbor_data = nx_graph.nodes[neighbor]
                # Check if neighbor is a chunk node (common pattern: node_type='chunk')
                if neighbor_data.get('node_type') == 'chunk' or neighbor_data.get('type') == 'chunk':
                    chunk_ids.append(neighbor)
            
            # Limit chunks per entity
            if validated_input.max_chunks_per_entity and len(chunk_ids) > validated_input.max_chunks_per_entity:
                chunk_ids = chunk_ids[:validated_input.max_chunks_per_entity]
            
            chunks_per_entity[entity_id] = chunk_ids
            logger.debug(f"Entity '{entity_id}' associated with chunk IDs: {chunk_ids}")
        
        # If specific chunk_ids were provided, use those instead
        if validated_input.chunk_ids:
            chunk_ids_to_retrieve = validated_input.chunk_ids
        else:
            # Collect all unique chunk IDs from entity associations
            chunk_ids_to_retrieve = []
            for chunk_list in chunks_per_entity.values():
                chunk_ids_to_retrieve.extend(chunk_list)
            chunk_ids_to_retrieve = list(set(chunk_ids_to_retrieve))
        
        # Retrieve chunk content
        chunk_storage = context.chunk_storage_manager
        
        # First, get all chunks for the dataset
        all_chunks_dict = {}
        if chunk_storage:
            try:
                # Extract dataset name from graph_reference_id
                dataset_name = validated_input.graph_reference_id
                for suffix in ["_ERGraph", "_RKGraph", "_TreeGraph", "_PassageGraph"]:
                    if dataset_name.endswith(suffix):
                        dataset_name = dataset_name[:-len(suffix)]
                        break
                
                # Get all chunks for the dataset
                chunks_list = await chunk_storage.get_chunks_for_dataset(dataset_name)
                # Convert to dict for easy lookup - try multiple key formats
                all_chunks_dict = {}
                
                # Store chunks by their actual chunk_id
                for chunk_id, chunk in chunks_list:
                    all_chunks_dict[chunk_id] = chunk
                    
                    # Also store by doc_id-based keys for backward compatibility
                    doc_id_key = f"chunk_{chunk.doc_id}"
                    all_chunks_dict[doc_id_key] = chunk
                    
                logger.info(f"Loaded {len(chunks_list)} chunks for dataset '{dataset_name}' (indexed with {len(all_chunks_dict)} keys)")
                # Debug: show some chunk IDs
                logger.debug(f"Sample chunk IDs in storage: {list(all_chunks_dict.keys())[:10]}")
            except Exception as e:
                logger.warning(f"Failed to load chunks from storage: {e}")
        
        for chunk_id in chunk_ids_to_retrieve:
            try:
                # Try to get chunk from our loaded chunks
                chunk_data = None
                chunk_obj = None
                
                # Try direct lookup first
                if chunk_id in all_chunks_dict:
                    chunk_obj = all_chunks_dict[chunk_id]
                else:
                    # If chunk_id looks like a hash, try to find by matching content
                    # This handles the case where graph has hash-based IDs but corpus has simple IDs
                    if chunk_id.startswith('chunk-') and len(chunk_id) > 10:
                        # This might be a hash-based ID, look for chunks by index
                        # The graph seems to use hashes, but we can try to match by order
                        for stored_id, stored_chunk in all_chunks_dict.items():
                            if stored_id.startswith('chunk_') and stored_chunk.doc_id is not None:
                                # Try to match based on entity content being in chunk
                                if entity_id.lower() in stored_chunk.content.lower():
                                    chunk_obj = stored_chunk
                                    logger.debug(f"Matched entity '{entity_id}' to chunk via content search")
                                    break
                
                if chunk_obj:
                    # chunk_obj is a TextChunk object
                    chunk_data = {
                        'content': chunk_obj.content,
                        'doc_id': chunk_obj.doc_id,
                        'title': chunk_obj.title,
                        'index': chunk_obj.index,
                        'tokens': chunk_obj.tokens
                    }
                    logger.debug(f"Found chunk {chunk_id} with content length: {len(chunk_obj.content)}")
                
                if chunk_data:
                    # Find which entity this chunk is associated with
                    associated_entity = None
                    for entity_id, entity_chunks in chunks_per_entity.items():
                        if chunk_id in entity_chunks:
                            associated_entity = entity_id
                            break
                    
                    chunk_item = {
                        "entity_id": associated_entity,
                        "chunk_id": chunk_id,
                        "text_content": chunk_data.get('content', ''),
                        "metadata": {
                            k: v for k, v in chunk_data.items() 
                            if k not in ['content', 'text', 'chunk_id', 'id']
                        }
                    }
                    retrieved_chunks_data.append(chunk_item)
                else:
                    # Fallback: try to get chunk content from graph node
                    if chunk_id in nx_graph:
                        chunk_node = nx_graph.nodes[chunk_id]
                        content = chunk_node.get('content', chunk_node.get('text', ''))
                        if content:
                            # Find associated entity
                            associated_entity = None
                            for entity_id, entity_chunks in chunks_per_entity.items():
                                if chunk_id in entity_chunks:
                                    associated_entity = entity_id
                                    break
                            
                            chunk_item = {
                                "entity_id": associated_entity,
                                "chunk_id": chunk_id,
                                "text_content": content,
                                "metadata": {
                                    k: v for k, v in chunk_node.items()
                                    if k not in ['content', 'text'] and not k.startswith('_')
                                }
                            }
                            retrieved_chunks_data.append(chunk_item)
            except Exception as e:
                logger.warning(f"Failed to retrieve chunk {chunk_id}: {e}")
                continue
        
        status_msg = f"Retrieved {len(retrieved_chunks_data)} chunks for {len(validated_input.entity_ids)} entities"
        logger.info(status_msg)
        
        return {
            "retrieved_chunks": retrieved_chunks_data,
            "status_message": status_msg
        }
        
    except Exception as e:
        error_msg = f"Error in Chunk.GetTextForEntities: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "retrieved_chunks": [],
            "status_message": error_msg
        }
