# Core/AgentTools/graph_analysis_tools.py

import logging
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import numpy as np

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import GraphAnalyzerInput, GraphAnalyzerOutput
from Core.Common.Logger import logger


def analyze_graph(input_data: Dict[str, Any], context: GraphRAGContext) -> Dict[str, Any]:
    """
    Analyze a graph and calculate various metrics.
    
    Args:
        input_data: Dictionary containing graph_id and analysis parameters
        context: GraphRAGContext containing graph instances
        
    Returns:
        Dictionary with calculated metrics and analysis results
    """
    # Validate input data
    try:
        params = GraphAnalyzerInput(**input_data)
    except Exception as e:
        logger.error(f"Failed to validate GraphAnalyzer input: {e}")
        return {
            "graph_id": input_data.get("graph_id", ""),
            "message": f"Error: Invalid input parameters - {str(e)}"
        }
    
    graph_id = params.graph_id
    metrics_to_calculate = params.metrics_to_calculate
    top_k_nodes = params.top_k_nodes
    calculate_expensive = params.calculate_expensive_metrics
    
    if not graph_id:
        return {
            "graph_id": "",
            "message": "Error: graph_id is required."
        }
    
    try:
        # 1. Retrieve the graph from context
        graph_instance = context.get_graph_instance(graph_id)
        
        if graph_instance is None:
            return {
                "graph_id": graph_id,
                "message": f"Error: Graph with ID '{graph_id}' not found in context."
            }
        
        # Get the actual NetworkX graph
        graph = _extract_networkx_graph(graph_instance)
        
        if graph is None:
            return {
                "graph_id": graph_id,
                "message": f"Error: Could not extract NetworkX graph from graph instance."
            }
        
        # 2. Determine which metrics to calculate
        all_metrics = ['basic_stats', 'centrality', 'clustering', 'connectivity', 'components', 'paths']
        if metrics_to_calculate is None:
            metrics_to_calculate = all_metrics
        else:
            # Validate requested metrics
            invalid_metrics = [m for m in metrics_to_calculate if m not in all_metrics]
            if invalid_metrics:
                return {
                    "graph_id": graph_id,
                    "message": f"Error: Invalid metrics requested: {invalid_metrics}. Valid options: {all_metrics}"
                }
        
        # 3. Calculate requested metrics
        result = {"graph_id": graph_id}
        warnings = []
        
        # Basic statistics
        if 'basic_stats' in metrics_to_calculate:
            result['basic_stats'] = _calculate_basic_stats(graph)
        
        # Centrality metrics
        if 'centrality' in metrics_to_calculate:
            centrality_result, centrality_warnings = _calculate_centrality_metrics(
                graph, top_k_nodes, calculate_expensive
            )
            result['centrality_metrics'] = centrality_result
            warnings.extend(centrality_warnings)
        
        # Clustering metrics
        if 'clustering' in metrics_to_calculate:
            result['clustering_metrics'] = _calculate_clustering_metrics(graph)
        
        # Connectivity metrics
        if 'connectivity' in metrics_to_calculate:
            result['connectivity_metrics'] = _calculate_connectivity_metrics(graph)
        
        # Component details
        if 'components' in metrics_to_calculate:
            result['component_details'] = _calculate_component_details(graph)
        
        # Path metrics
        if 'paths' in metrics_to_calculate:
            path_result, path_warnings = _calculate_path_metrics(graph, calculate_expensive)
            result['path_metrics'] = path_result
            warnings.extend(path_warnings)
        
        # 4. Compile status message
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        message = f"Successfully analyzed graph '{graph_id}' with {node_count} nodes and {edge_count} edges."
        if warnings:
            message += " Warnings: " + "; ".join(warnings)
        
        result['message'] = message
        logger.info(f"Successfully analyzed graph '{graph_id}'")
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing graph '{graph_id}': {str(e)}")
        return {
            "graph_id": graph_id,
            "message": f"Error: Failed to analyze graph - {str(e)}"
        }


def _extract_networkx_graph(graph_instance) -> Optional[nx.Graph]:
    """Extract NetworkX graph from graph instance."""
    if hasattr(graph_instance, '_graph'):
        storage = graph_instance._graph
        if hasattr(storage, 'graph'):
            return storage.graph
        elif hasattr(storage, '_graph'):
            return storage._graph
        elif isinstance(storage, nx.Graph):
            return storage
    elif hasattr(graph_instance, 'graph'):
        return graph_instance.graph
    elif isinstance(graph_instance, nx.Graph):
        return graph_instance
    return None


def _calculate_basic_stats(graph: nx.Graph) -> Dict[str, Any]:
    """Calculate basic graph statistics."""
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "density": nx.density(graph),
        "is_directed": graph.is_directed(),
        "is_multigraph": graph.is_multigraph(),
        "self_loops": nx.number_of_selfloops(graph),
        "average_degree": 2.0 * graph.number_of_edges() / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
    }


def _calculate_centrality_metrics(
    graph: nx.Graph, 
    top_k: int, 
    calculate_expensive: bool
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Calculate various centrality metrics."""
    warnings = []
    centrality_metrics = {}
    
    # Degree centrality (fast)
    degree_centrality = nx.degree_centrality(graph)
    centrality_metrics['degree'] = _get_top_k_nodes(degree_centrality, top_k)
    
    # PageRank (relatively fast)
    try:
        pagerank = nx.pagerank(graph)
        centrality_metrics['pagerank'] = _get_top_k_nodes(pagerank, top_k)
    except:
        warnings.append("PageRank calculation failed (graph might not be suitable)")
    
    # Calculate expensive metrics only if requested or graph is small
    if calculate_expensive or graph.number_of_nodes() < 100:
        # Betweenness centrality (expensive)
        try:
            betweenness = nx.betweenness_centrality(graph)
            centrality_metrics['betweenness'] = _get_top_k_nodes(betweenness, top_k)
        except:
            warnings.append("Betweenness centrality calculation failed")
        
        # Closeness centrality (expensive for disconnected graphs)
        try:
            closeness = nx.closeness_centrality(graph)
            centrality_metrics['closeness'] = _get_top_k_nodes(closeness, top_k)
        except:
            warnings.append("Closeness centrality calculation failed")
        
        # Eigenvector centrality (can be expensive)
        try:
            eigenvector = nx.eigenvector_centrality(graph, max_iter=100)
            centrality_metrics['eigenvector'] = _get_top_k_nodes(eigenvector, top_k)
        except:
            warnings.append("Eigenvector centrality calculation failed (graph might not be suitable)")
    else:
        warnings.append(f"Skipped expensive centrality metrics (graph has {graph.number_of_nodes()} nodes). Set calculate_expensive_metrics=True to calculate them.")
    
    return centrality_metrics, warnings


def _get_top_k_nodes(scores: Dict[str, float], k: int) -> Dict[str, float]:
    """Get top k nodes by score."""
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_nodes[:k])


def _calculate_clustering_metrics(graph: nx.Graph) -> Dict[str, float]:
    """Calculate clustering-related metrics."""
    metrics = {}
    
    # Convert to undirected for clustering calculations
    undirected_graph = graph.to_undirected() if graph.is_directed() else graph
    
    try:
        metrics['average_clustering'] = nx.average_clustering(undirected_graph)
    except:
        metrics['average_clustering'] = None
    
    try:
        metrics['transitivity'] = nx.transitivity(undirected_graph)
    except:
        metrics['transitivity'] = None
    
    try:
        # Count triangles
        triangles = sum(nx.triangles(undirected_graph).values()) // 3
        metrics['number_of_triangles'] = triangles
    except:
        metrics['number_of_triangles'] = None
    
    return metrics


def _calculate_connectivity_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """Calculate connectivity-related metrics."""
    metrics = {}
    
    if graph.is_directed():
        metrics['is_strongly_connected'] = nx.is_strongly_connected(graph)
        metrics['is_weakly_connected'] = nx.is_weakly_connected(graph)
        metrics['number_strongly_connected_components'] = nx.number_strongly_connected_components(graph)
        metrics['number_weakly_connected_components'] = nx.number_weakly_connected_components(graph)
    else:
        metrics['is_connected'] = nx.is_connected(graph)
        metrics['number_connected_components'] = nx.number_connected_components(graph)
    
    # Get size of largest component
    if graph.is_directed():
        components = list(nx.strongly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))
    
    if components:
        metrics['largest_component_size'] = max(len(c) for c in components)
        metrics['smallest_component_size'] = min(len(c) for c in components)
    
    return metrics


def _calculate_component_details(graph: nx.Graph) -> List[Dict[str, Any]]:
    """Get details about connected components."""
    if graph.is_directed():
        components = list(nx.strongly_connected_components(graph))
        component_type = "strongly_connected"
    else:
        components = list(nx.connected_components(graph))
        component_type = "connected"
    
    # Sort components by size (descending)
    components = sorted(components, key=len, reverse=True)
    
    component_details = []
    for i, component in enumerate(components[:10]):  # Limit to top 10 components
        subgraph = graph.subgraph(component)
        details = {
            "component_id": i,
            "size": len(component),
            "density": nx.density(subgraph),
            "type": component_type
        }
        
        # Sample nodes if component is large
        if len(component) > 10:
            details["sample_nodes"] = list(component)[:10]
        else:
            details["nodes"] = list(component)
        
        component_details.append(details)
    
    return component_details


def _calculate_path_metrics(
    graph: nx.Graph, 
    calculate_expensive: bool
) -> Tuple[Dict[str, Any], List[str]]:
    """Calculate path-related metrics."""
    metrics = {}
    warnings = []
    
    # For directed graphs, use the largest strongly connected component
    if graph.is_directed():
        components = list(nx.strongly_connected_components(graph))
        if components:
            largest_component = max(components, key=len)
            subgraph = graph.subgraph(largest_component)
            warnings.append("Path metrics calculated on largest strongly connected component only")
        else:
            return {}, ["No strongly connected components found"]
    else:
        # For undirected graphs, check if connected
        if nx.is_connected(graph):
            subgraph = graph
        else:
            components = list(nx.connected_components(graph))
            largest_component = max(components, key=len)
            subgraph = graph.subgraph(largest_component)
            warnings.append("Path metrics calculated on largest connected component only")
    
    # Calculate metrics on the (sub)graph
    if calculate_expensive or subgraph.number_of_nodes() < 100:
        try:
            metrics['diameter'] = nx.diameter(subgraph)
            metrics['radius'] = nx.radius(subgraph)
        except:
            warnings.append("Could not calculate diameter/radius")
        
        try:
            metrics['average_shortest_path_length'] = nx.average_shortest_path_length(subgraph)
        except:
            warnings.append("Could not calculate average shortest path length")
    else:
        warnings.append(f"Skipped expensive path metrics (component has {subgraph.number_of_nodes()} nodes). Set calculate_expensive_metrics=True to calculate them.")
    
    metrics['component_analyzed_size'] = subgraph.number_of_nodes()
    
    return metrics, warnings
