# Core/AgentTools/graph_visualization_tools.py

import json
import logging
from typing import Dict, Any, Optional
import networkx as nx

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import GraphVisualizerInput, GraphVisualizerOutput
from Core.Common.Logger import logger


def visualize_graph(input_data: Dict[str, Any], context: GraphRAGContext) -> Dict[str, Any]:
    """
    Visualize a graph in various formats.
    
    Args:
        input_data: Dictionary containing graph_id and output_format
        context: GraphRAGContext containing workspace and namespace info
        
    Returns:
        Dictionary with graph representation, format used, and optional message
    """
    # Validate input data
    try:
        params = GraphVisualizerInput(**input_data)
    except Exception as e:
        logger.error(f"Failed to validate GraphVisualizer input: {e}")
        return {
            "graph_representation": "",
            "format_used": "",
            "message": f"Error: Invalid input parameters - {str(e)}"
        }
    
    graph_id = params.graph_id
    output_format = params.output_format
    
    if not graph_id:
        return {
            "graph_representation": "",
            "format_used": output_format,
            "message": "Error: graph_id is required."
        }
    
    try:
        # 1. Retrieve the graph from context
        graph_instance = context.get_graph_instance(graph_id)
        
        if graph_instance is None:
            return {
                "graph_representation": "",
                "format_used": output_format,
                "message": f"Error: Graph with ID '{graph_id}' not found in context."
            }
        
        # Get the actual NetworkX graph from the graph instance
        # The graph instance typically has a _graph attribute that is the storage
        # and the storage has a graph or _graph attribute that is the nx.Graph
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
            return {
                "graph_representation": "",
                "format_used": output_format,
                "message": f"Error: Could not extract NetworkX graph from graph instance."
            }
        
        # 2. Convert the graph to the desired output format
        graph_representation_str = ""
        
        if output_format == "GML":
            # Convert NetworkX graph to GML string
            lines = []
            for line in nx.generate_gml(graph):
                lines.append(line)
            graph_representation_str = '\n'.join(lines)
            
        elif output_format == "JSON_NODES_EDGES":
            # Convert graph to a JSON serializable dict of nodes and edges
            nodes = []
            for node_id in graph.nodes():
                node_data = {"id": node_id}
                # Add node attributes
                node_data.update(graph.nodes[node_id])
                nodes.append(node_data)
            
            edges = []
            for source, target in graph.edges():
                edge_data = {"source": source, "target": target}
                # Add edge attributes
                edge_data.update(graph.edges[source, target])
                edges.append(edge_data)
            
            graph_dict = {
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    "node_count": graph.number_of_nodes(),
                    "edge_count": graph.number_of_edges(),
                    "is_directed": graph.is_directed()
                }
            }
            graph_representation_str = json.dumps(graph_dict, indent=2)
            
        else:
            return {
                "graph_representation": "",
                "format_used": output_format,
                "message": f"Error: Unsupported output format '{output_format}'. Supported formats: 'GML', 'JSON_NODES_EDGES'"
            }
        
        # 3. Return the output
        logger.info(f"Successfully visualized graph '{graph_id}' in {output_format} format")
        return {
            "graph_representation": graph_representation_str,
            "format_used": output_format,
            "message": f"Graph visualized successfully. Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}"
        }
        
    except Exception as e:
        logger.error(f"Error visualizing graph '{graph_id}': {str(e)}")
        return {
            "graph_representation": "",
            "format_used": output_format,
            "message": f"Error: Failed to visualize graph - {str(e)}"
        }
