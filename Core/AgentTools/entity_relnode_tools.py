# Core/AgentTools/entity_relnode_tools.py

"""
Entity RelNode Tool

This tool extracts entities that are connected by specific relationships.
Used in methods like LightRAG for finding entities involved in particular relationships.
"""

import asyncio
from typing import Dict, Any, List, Set, Tuple
import networkx as nx
from Core.Common.Logger import logger
from Core.AgentSchema.tool_contracts import EntityRelNodeInput, EntityRelNodeOutput
from Core.AgentSchema.context import GraphRAGContext


def entity_relnode_extract(input_data: Dict[str, Any], context: GraphRAGContext) -> Dict[str, Any]:
    """
    Extract entities connected by specific relationships.
    
    Args:
        input_data: Dictionary containing EntityRelNodeInput fields
        context: GraphRAGContext containing graph instances
        
    Returns:
        Dictionary containing EntityRelNodeOutput fields
    """
    try:
        # Validate input
        validated_input = EntityRelNodeInput(**input_data)
    except Exception as e:
        logger.error(f"Failed to validate Entity.RelNode input: {e}")
        return {
            "entities": [],
            "entity_count": 0,
            "relationship_entity_map": {},
            "message": f"Invalid input parameters: {str(e)}"
        }
    
    relationship_ids = validated_input.relationship_ids
    graph_id = validated_input.graph_id
    role_filter = validated_input.entity_role_filter
    type_filter = validated_input.entity_type_filter
    
    # Validate role filter
    if role_filter and role_filter not in ['source', 'target', 'both']:
        logger.warning(f"Invalid role filter '{role_filter}', using 'both'")
        role_filter = 'both'
    
    # Default to 'both' if not specified
    if not role_filter:
        role_filter = 'both'
    
    # Get graph instance
    graph_instance = context.get_graph_instance(graph_id)
    if graph_instance is None:
        logger.warning(f"Graph '{graph_id}' not found in context")
        return {
            "entities": [],
            "entity_count": 0,
            "relationship_entity_map": {},
            "message": f"Graph '{graph_id}' not found in context"
        }
    
    # Extract NetworkX graph
    graph = None
    if hasattr(graph_instance, '_graph'):
        storage = graph_instance._graph
        if hasattr(storage, 'graph'):
            graph = storage.graph
        elif hasattr(storage, '_graph'):
            graph = storage._graph
        elif isinstance(storage, nx.Graph):
            graph = storage
    elif hasattr(graph_instance, 'graph'):
        graph = graph_instance.graph
    elif isinstance(graph_instance, nx.Graph):
        graph = graph_instance
    
    if graph is None:
        logger.error(f"Could not extract NetworkX graph from graph instance '{graph_id}'")
        return {
            "entities": [],
            "entity_count": 0,
            "relationship_entity_map": {},
            "message": f"Could not extract graph data from '{graph_id}'"
        }
    
    # Extract entities from relationships
    entities = {}  # entity_id -> entity_data
    relationship_entity_map = {}
    
    # In NetworkX, relationships are typically represented as edges
    # We'll assume relationship_ids correspond to edge identifiers or edge attributes
    
    # First, try to find edges by relationship ID
    edges_to_process = []
    
    # Check if edges have 'id' or 'relationship_id' attribute
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
        
        if edge_id in relationship_ids:
            edges_to_process.append((u, v, edge_data, edge_id))
    
    # If no edges found by ID, try interpreting relationship_ids as edge patterns
    if not edges_to_process:
        for rel_id in relationship_ids:
            if '->' in rel_id:
                # Parse as "source->target" pattern
                parts = rel_id.split('->')
                if len(parts) == 2:
                    source, target = parts[0].strip(), parts[1].strip()
                    if graph.has_edge(source, target):
                        edge_data = graph[source][target]
                        edges_to_process.append((source, target, edge_data, rel_id))
    
    # Process found edges
    for source, target, edge_data, rel_id in edges_to_process:
        entities_in_rel = []
        
        # Extract source entity
        if role_filter in ['source', 'both'] and source in graph:
            node_data = graph.nodes[source]
            
            # Check type filter
            if type_filter is None or ('type' in node_data and node_data['type'] in type_filter):
                entity_data = {
                    "entity_id": source,
                    "role": "source",
                    "relationship_id": rel_id,
                    "attributes": dict(node_data)
                }
                entities[source] = entity_data
                entities_in_rel.append(source)
        
        # Extract target entity
        if role_filter in ['target', 'both'] and target in graph:
            node_data = graph.nodes[target]
            
            # Check type filter
            if type_filter is None or ('type' in node_data and node_data['type'] in type_filter):
                entity_data = {
                    "entity_id": target,
                    "role": "target",
                    "relationship_id": rel_id,
                    "attributes": dict(node_data)
                }
                # If entity already exists from another relationship, merge the info
                if target in entities:
                    # Add this relationship to existing entity
                    if isinstance(entities[target]["relationship_id"], list):
                        entities[target]["relationship_id"].append(rel_id)
                    else:
                        entities[target]["relationship_id"] = [entities[target]["relationship_id"], rel_id]
                    # Update role if needed
                    if entities[target]["role"] != "both" and entities[target]["role"] != entity_data["role"]:
                        entities[target]["role"] = "both"
                else:
                    entities[target] = entity_data
                entities_in_rel.append(target)
        
        relationship_entity_map[rel_id] = entities_in_rel
    
    # Convert entities dict to list
    entities_list = list(entities.values())
    
    # Handle case where no relationships were found
    if not edges_to_process:
        message = f"No relationships found matching IDs: {relationship_ids}"
        logger.warning(message)
    else:
        message = f"Successfully extracted {len(entities)} entities from {len(edges_to_process)} relationships"
    
    # Create output
    result = {
        "entities": entities_list,
        "entity_count": len(entities),
        "relationship_entity_map": relationship_entity_map,
        "message": message
    }
    
    logger.info(f"Entity.RelNode extracted {len(entities)} entities from {len(relationship_ids)} relationship IDs in graph '{graph_id}'")
    
    return result


# Async wrapper for compatibility with async orchestrator
async def entity_relnode_extract_tool(input_data: Dict[str, Any], context: GraphRAGContext) -> Dict[str, Any]:
    """Async wrapper for entity_relnode_extract."""
    return entity_relnode_extract(input_data, context)
