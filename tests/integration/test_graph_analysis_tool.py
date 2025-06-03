# testing/test_graph_analysis_tool.py

import json
import os
import sys
import pytest
import networkx as nx
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.AgentTools.graph_analysis_tools import analyze_graph
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from Option.Config2 import Config  # For main_config


def create_test_context(workspace: str, namespace: str) -> GraphRAGContext:
    """Create a valid GraphRAGContext for testing."""
    main_config = Config.default()  # Get default config
    
    # Set workspace in config output dir
    if hasattr(main_config, 'output') and hasattr(main_config.output, 'dir'):
        main_config.output.dir = workspace
    
    # Use namespace as target_dataset_name since that's what the tool will use
    context = GraphRAGContext(
        target_dataset_name=namespace,  # This will be used as namespace
        main_config=main_config
    )
    
    return context


# Create mock classes to wrap NetworkX graphs
class MockGraphInstance:
    def __init__(self, graph):
        self._graph = MockStorage(graph)
        
        
class MockStorage:
    """Mock storage class that holds a NetworkX graph."""
    def __init__(self, graph):
        self.graph = graph  # This is the actual NetworkX graph


def create_small_test_graph():
    """Create a small test graph for testing."""
    G = nx.DiGraph()
    
    # Create a simple graph with clear structure
    # Hub node (A) connected to others
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("A", "C", weight=2.0)
    G.add_edge("A", "D", weight=1.5)
    
    # Create a triangle
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "B", weight=1.0)
    
    # Add some additional connections
    G.add_edge("D", "E", weight=1.0)
    G.add_edge("E", "F", weight=1.0)
    G.add_edge("F", "D", weight=2.0)  # Create a cycle
    
    # Add an isolated component
    G.add_edge("X", "Y", weight=1.0)
    G.add_edge("Y", "Z", weight=1.0)
    
    return G


def create_large_test_graph():
    """Create a larger test graph for performance testing."""
    # Create a scale-free graph (common in real-world networks)
    # Note: barabasi_albert_graph creates undirected graphs, so we'll convert
    G_undirected = nx.barabasi_albert_graph(150, 3)
    
    # Convert to directed graph
    G = G_undirected.to_directed()
    
    # Convert node labels to strings
    mapping = {i: f"node_{i}" for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    return G


def test_analyze_graph_basic_stats():
    """Test basic statistics calculation."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_small_test_graph()
    graph_id = "test_graph_basic"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test analysis with only basic stats
    input_data = {
        "graph_id": graph_id,
        "metrics_to_calculate": ["basic_stats"]
    }
    
    result = analyze_graph(input_data, context)
    
    # Assertions
    assert result["graph_id"] == graph_id
    assert "basic_stats" in result
    assert result["basic_stats"]["node_count"] == 9  # A,B,C,D,E,F,X,Y,Z
    assert result["basic_stats"]["edge_count"] == 10
    assert result["basic_stats"]["is_directed"] == True
    assert "density" in result["basic_stats"]
    assert "average_degree" in result["basic_stats"]
    
    logger.info("test_analyze_graph_basic_stats passed!")


def test_analyze_graph_centrality():
    """Test centrality metrics calculation."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_small_test_graph()
    graph_id = "test_graph_centrality"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test analysis with centrality metrics
    input_data = {
        "graph_id": graph_id,
        "metrics_to_calculate": ["centrality"],
        "top_k_nodes": 5,
        "calculate_expensive_metrics": True  # Small graph, so calculate all
    }
    
    result = analyze_graph(input_data, context)
    
    # Assertions
    assert result["graph_id"] == graph_id
    assert "centrality_metrics" in result
    assert "degree" in result["centrality_metrics"]
    assert "pagerank" in result["centrality_metrics"]
    
    # Node A should have high degree centrality (hub node)
    degree_centrality = result["centrality_metrics"]["degree"]
    assert "A" in degree_centrality
    
    # Should have at most 5 nodes per metric (top_k_nodes)
    assert len(degree_centrality) <= 5
    
    logger.info("test_analyze_graph_centrality passed!")


def test_analyze_graph_clustering():
    """Test clustering metrics calculation."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_small_test_graph()
    graph_id = "test_graph_clustering"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test analysis with clustering metrics
    input_data = {
        "graph_id": graph_id,
        "metrics_to_calculate": ["clustering"]
    }
    
    result = analyze_graph(input_data, context)
    
    # Assertions
    assert result["graph_id"] == graph_id
    assert "clustering_metrics" in result
    assert "average_clustering" in result["clustering_metrics"]
    assert "transitivity" in result["clustering_metrics"]
    assert "number_of_triangles" in result["clustering_metrics"]
    
    # Should find at least one triangle (B-C-A)
    assert result["clustering_metrics"]["number_of_triangles"] >= 1
    
    logger.info("test_analyze_graph_clustering passed!")


def test_analyze_graph_connectivity():
    """Test connectivity metrics calculation."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_small_test_graph()
    graph_id = "test_graph_connectivity"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test analysis with connectivity metrics
    input_data = {
        "graph_id": graph_id,
        "metrics_to_calculate": ["connectivity", "components"]
    }
    
    result = analyze_graph(input_data, context)
    
    # Assertions
    assert result["graph_id"] == graph_id
    assert "connectivity_metrics" in result
    
    # For directed graph, should have strongly/weakly connected info
    assert "is_strongly_connected" in result["connectivity_metrics"]
    assert "is_weakly_connected" in result["connectivity_metrics"]
    assert result["connectivity_metrics"]["is_strongly_connected"] == False  # Has multiple components
    
    # Component details
    assert "component_details" in result
    assert len(result["component_details"]) > 0
    assert "size" in result["component_details"][0]
    assert "density" in result["component_details"][0]
    
    logger.info("test_analyze_graph_connectivity passed!")


def test_analyze_large_graph_performance():
    """Test performance settings with larger graph."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create larger test graph and add to context
    test_graph = create_large_test_graph()
    graph_id = "test_graph_large"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test analysis WITHOUT expensive metrics
    input_data = {
        "graph_id": graph_id,
        "metrics_to_calculate": ["basic_stats", "centrality"],
        "calculate_expensive_metrics": False  # Should skip expensive calculations
    }
    
    result = analyze_graph(input_data, context)
    
    # Assertions
    assert result["graph_id"] == graph_id
    assert "basic_stats" in result
    assert result["basic_stats"]["node_count"] == 150
    
    # Should have warning about skipping expensive metrics
    assert "Skipped expensive" in result["message"]
    
    # Should still have degree centrality (fast)
    assert "centrality_metrics" in result
    assert "degree" in result["centrality_metrics"]
    
    logger.info("test_analyze_large_graph_performance passed!")


def test_analyze_all_metrics():
    """Test analyzing with all metrics (default behavior)."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_small_test_graph()
    graph_id = "test_graph_all"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test analysis without specifying metrics (should calculate all)
    input_data = {
        "graph_id": graph_id,
        "calculate_expensive_metrics": True
    }
    
    result = analyze_graph(input_data, context)
    
    # Assertions - should have all metric types
    assert result["graph_id"] == graph_id
    assert "basic_stats" in result
    assert "centrality_metrics" in result
    assert "clustering_metrics" in result
    assert "connectivity_metrics" in result
    assert "component_details" in result
    assert "path_metrics" in result
    
    logger.info("test_analyze_all_metrics passed!")


def test_analyze_nonexistent_graph():
    """Test analyzing a graph that doesn't exist."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Test analysis of non-existent graph
    input_data = {
        "graph_id": "nonexistent_graph",
        "metrics_to_calculate": ["basic_stats"]
    }
    
    result = analyze_graph(input_data, context)
    
    # Assertions
    assert result["graph_id"] == "nonexistent_graph"
    assert "not found" in result["message"]
    assert "basic_stats" not in result
    
    logger.info("test_analyze_nonexistent_graph passed!")


def test_analyze_invalid_metrics():
    """Test requesting invalid metrics."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_small_test_graph()
    graph_id = "test_graph_invalid"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test with invalid metrics
    input_data = {
        "graph_id": graph_id,
        "metrics_to_calculate": ["invalid_metric", "basic_stats"]
    }
    
    result = analyze_graph(input_data, context)
    
    # Assertions
    assert result["graph_id"] == graph_id
    assert "Invalid metrics" in result["message"]
    assert "basic_stats" not in result  # Should not calculate any metrics
    
    logger.info("test_analyze_invalid_metrics passed!")


def test_analyze_missing_graph_id():
    """Test analyzing without providing graph_id."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Test analysis without graph_id
    input_data = {
        "metrics_to_calculate": ["basic_stats"]
    }
    
    result = analyze_graph(input_data, context)
    
    # Assertions
    assert "Invalid input parameters" in result["message"]
    
    logger.info("test_analyze_missing_graph_id passed!")


if __name__ == "__main__":
    # Run all tests
    test_analyze_graph_basic_stats()
    test_analyze_graph_centrality()
    test_analyze_graph_clustering()
    test_analyze_graph_connectivity()
    test_analyze_large_graph_performance()
    test_analyze_all_metrics()
    test_analyze_nonexistent_graph()
    test_analyze_invalid_metrics()
    test_analyze_missing_graph_id()
    
    print("\nAll GraphAnalyzer tests passed successfully!")
