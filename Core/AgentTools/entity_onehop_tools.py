# Core/AgentTools/entity_onehop_tools.py

"""
Entity One-Hop Neighbors Tool

This tool extracts all one-hop neighbor entities for given seed entities.
Used in methods like LightRAG for expanding entity context.
"""

import asyncio
from typing import Dict, Any, List, Set
import networkx as nx
from Core.Common.Logger import logger
from Core.AgentSchema.tool_contracts import EntityOneHopInput, EntityOneHopOutput
from Core.AgentSchema.context import GraphRAGContext


def entity_onehop_neighbors(input_data: Dict[str, Any], context: GraphRAGContext) -> Dict[str, Any]:
    """
    Find one-hop neighbor entities for given entity IDs.
    
    Args:
        input_data: Dictionary containing EntityOneHopInput fields
        context: GraphRAGContext containing graph instances
        
    Returns:
        Dictionary containing EntityOneHopOutput fields
    """
    try:
        # Validate input
        validated_input = EntityOneHopInput(**input_data)
    except Exception as e:
        logger.error(f"Failed to validate Entity.Onehop input: {e}")
        return {
            "neighbors": {},
            "total_neighbors_found": 0,
            "message": f"Invalid input parameters: {str(e)}"
        }
    
    entity_ids = validated_input.entity_ids
    graph_id = validated_input.graph_id
    include_edge_attrs = validated_input.include_edge_attributes
    neighbor_limit = validated_input.neighbor_limit_per_entity
    
    # Get graph instance
    graph_instance = context.get_graph_instance(graph_id)
    if graph_instance is None:
        logger.warning(f"Graph '{graph_id}' not found in context")
        return {
            "neighbors": {},
            "total_neighbors_found": 0,
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
            "neighbors": {},
            "total_neighbors_found": 0,
            "message": f"Could not extract graph data from '{graph_id}'"
        }
    
    # Find one-hop neighbors for each entity
    neighbors_dict = {}
    all_neighbors = set()
    
    for entity_id in entity_ids:
        if entity_id not in graph:
            logger.warning(f"Entity '{entity_id}' not found in graph '{graph_id}'")
            neighbors_dict[entity_id] = []
            continue
        
        # Get neighbors
        try:
            if graph.is_directed():
                # For directed graphs, get both successors and predecessors
                successors = list(graph.successors(entity_id))
                predecessors = list(graph.predecessors(entity_id))
                neighbor_ids = list(set(successors + predecessors))
            else:
                # For undirected graphs
                neighbor_ids = list(graph.neighbors(entity_id))
            
            # Apply neighbor limit if specified
            if neighbor_limit is not None and len(neighbor_ids) > neighbor_limit:
                # Sort by degree centrality to get most connected neighbors
                neighbor_degrees = [(n, graph.degree(n)) for n in neighbor_ids]
                neighbor_degrees.sort(key=lambda x: x[1], reverse=True)
                neighbor_ids = [n[0] for n in neighbor_degrees[:neighbor_limit]]
            
            # Build neighbor information
            neighbor_info = []
            for neighbor_id in neighbor_ids:
                neighbor_data = {
                    "entity_id": neighbor_id,
                    "node_attributes": dict(graph.nodes[neighbor_id]) if neighbor_id in graph else {}
                }
                
                # Include edge attributes if requested
                if include_edge_attrs:
                    edge_attrs = []
                    
                    if graph.is_directed():
                        # Check both directions
                        if graph.has_edge(entity_id, neighbor_id):
                            edge_attrs.append({
                                "direction": "outgoing",
                                "attributes": dict(graph[entity_id][neighbor_id])
                            })
                        if graph.has_edge(neighbor_id, entity_id):
                            edge_attrs.append({
                                "direction": "incoming", 
                                "attributes": dict(graph[neighbor_id][entity_id])
                            })
                    else:
                        # Undirected graph
                        if graph.has_edge(entity_id, neighbor_id):
                            edge_attrs.append({
                                "attributes": dict(graph[entity_id][neighbor_id])
                            })
                    
                    neighbor_data["edge_attributes"] = edge_attrs
                
                neighbor_info.append(neighbor_data)
                all_neighbors.add(neighbor_id)
            
            neighbors_dict[entity_id] = neighbor_info
            
        except Exception as e:
            logger.error(f"Error finding neighbors for entity '{entity_id}': {e}")
            neighbors_dict[entity_id] = []
    
    # Create output
    result = {
        "neighbors": neighbors_dict,
        "total_neighbors_found": len(all_neighbors),
        "message": f"Successfully found neighbors for {len(neighbors_dict)} entities"
    }
    
    # Log summary
    logger.info(f"Entity.Onehop found {len(all_neighbors)} unique neighbors for {len(entity_ids)} entities in graph '{graph_id}'")
    
    return result


# Async wrapper for compatibility with async orchestrator
async def entity_onehop_neighbors_tool(input_data: Dict[str, Any], context: GraphRAGContext) -> Dict[str, Any]:
    """Async wrapper for entity_onehop_neighbors."""
    return entity_onehop_neighbors(input_data, context)
