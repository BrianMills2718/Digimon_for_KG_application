# testing/test_graph_visualization_tool.py

import json
import os
import sys
import pytest
import networkx as nx
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.AgentTools.graph_visualization_tools import visualize_graph
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


def create_test_graph():
    """Create a simple test graph for visualization."""
    G = nx.DiGraph()
    
    # Add nodes with attributes
    G.add_node("american_revolution", 
               entity_type="EVENT", 
               description="Revolutionary War in America")
    G.add_node("george_washington", 
               entity_type="PERSON", 
               description="First President of the United States")
    G.add_node("united_states", 
               entity_type="COUNTRY", 
               description="Country formed after the American Revolution")
    
    # Add edges with attributes
    G.add_edge("george_washington", "american_revolution", 
               relationship_type="PARTICIPATED_IN", 
               description="Led Continental Army")
    G.add_edge("american_revolution", "united_states", 
               relationship_type="RESULTED_IN", 
               description="Led to formation of the United States")
    
    return G


# Create a mock graph instance class to wrap the NetworkX graph
class MockGraphInstance:
    def __init__(self, graph):
        # Create a mock storage object that has the graph
        self._graph = MockStorage(graph)


class MockStorage:
    """Mock storage class that holds a NetworkX graph."""
    def __init__(self, graph):
        self.graph = graph  # This is the actual NetworkX graph


def test_visualize_graph_json_format():
    """Test visualizing a graph in JSON_NODES_EDGES format."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph()
    graph_id = "test_graph_json"
    
    # Create a mock graph instance and add to context
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test visualization
    input_data = {
        "graph_id": graph_id,
        "output_format": "JSON_NODES_EDGES"
    }
    
    result = visualize_graph(input_data, context)
    
    # Assertions
    assert result["format_used"] == "JSON_NODES_EDGES"
    assert result["message"].startswith("Graph visualized successfully")
    assert result["graph_representation"] != ""
    
    # Parse and validate JSON
    graph_data = json.loads(result["graph_representation"])
    assert "nodes" in graph_data
    assert "edges" in graph_data
    assert "metadata" in graph_data
    assert len(graph_data["nodes"]) == 3
    assert len(graph_data["edges"]) == 2
    assert graph_data["metadata"]["node_count"] == 3
    assert graph_data["metadata"]["edge_count"] == 2
    assert graph_data["metadata"]["is_directed"] == True
    
    logger.info("test_visualize_graph_json_format passed!")


def test_visualize_graph_gml_format():
    """Test visualizing a graph in GML format."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph()
    graph_id = "test_graph_gml"
    
    # Create a mock graph instance and add to context
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test visualization
    input_data = {
        "graph_id": graph_id,
        "output_format": "GML"
    }
    
    result = visualize_graph(input_data, context)
    
    # Assertions
    assert result["format_used"] == "GML"
    assert result["message"].startswith("Graph visualized successfully")
    assert result["graph_representation"] != ""
    assert "graph [" in result["graph_representation"]
    assert "node [" in result["graph_representation"]
    assert "edge [" in result["graph_representation"]
    
    logger.info("test_visualize_graph_gml_format passed!")


def test_visualize_nonexistent_graph():
    """Test visualizing a graph that doesn't exist."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Test visualization of non-existent graph
    input_data = {
        "graph_id": "nonexistent_graph",
        "output_format": "JSON_NODES_EDGES"
    }
    
    result = visualize_graph(input_data, context)
    
    # Assertions
    assert result["format_used"] == "JSON_NODES_EDGES"
    assert "not found" in result["message"]
    assert result["graph_representation"] == ""
    
    logger.info("test_visualize_nonexistent_graph passed!")


def test_visualize_unsupported_format():
    """Test visualizing with an unsupported output format."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph()
    graph_id = "test_graph_unsupported"
    
    # Create a mock graph instance and add to context
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test visualization with unsupported format
    input_data = {
        "graph_id": graph_id,
        "output_format": "UNSUPPORTED_FORMAT"
    }
    
    result = visualize_graph(input_data, context)
    
    # Assertions
    assert result["format_used"] == "UNSUPPORTED_FORMAT"
    assert "Unsupported output format" in result["message"]
    assert result["graph_representation"] == ""
    
    logger.info("test_visualize_unsupported_format passed!")


def test_visualize_default_format():
    """Test visualizing with default format (JSON_NODES_EDGES)."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph()
    graph_id = "test_graph_default"
    
    # Create a mock graph instance and add to context
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test visualization without specifying format (should use default)
    input_data = {
        "graph_id": graph_id
    }
    
    result = visualize_graph(input_data, context)
    
    # Assertions
    assert result["format_used"] == "JSON_NODES_EDGES"
    assert result["message"].startswith("Graph visualized successfully")
    assert result["graph_representation"] != ""
    
    logger.info("test_visualize_default_format passed!")


def test_visualize_missing_graph_id():
    """Test visualizing without providing graph_id."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Test visualization without graph_id
    input_data = {
        "output_format": "JSON_NODES_EDGES"
    }
    
    result = visualize_graph(input_data, context)
    
    # Assertions
    assert "Invalid input parameters" in result["message"]
    assert result["graph_representation"] == ""
    
    logger.info("test_visualize_missing_graph_id passed!")


if __name__ == "__main__":
    # Run all tests
    test_visualize_graph_json_format()
    test_visualize_graph_gml_format()
    test_visualize_nonexistent_graph()
    test_visualize_unsupported_format()
    test_visualize_default_format()
    test_visualize_missing_graph_id()
    
    print("\nAll GraphVisualizer tests passed successfully!")
