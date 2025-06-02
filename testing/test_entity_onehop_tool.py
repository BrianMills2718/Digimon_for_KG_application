# testing/test_entity_onehop_tool.py

import json
import os
import sys
import pytest
import networkx as nx
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.AgentTools.entity_onehop_tools import entity_onehop_neighbors
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


def create_test_graph():
    """Create a test graph with various entity relationships."""
    G = nx.DiGraph()
    
    # Create entities with attributes
    entities = {
        "person_1": {"type": "Person", "name": "Alice"},
        "person_2": {"type": "Person", "name": "Bob"},
        "person_3": {"type": "Person", "name": "Charlie"},
        "company_1": {"type": "Company", "name": "TechCorp"},
        "company_2": {"type": "Company", "name": "DataInc"},
        "location_1": {"type": "Location", "name": "San Francisco"},
        "location_2": {"type": "Location", "name": "New York"},
        "isolated_entity": {"type": "Other", "name": "Isolated"}
    }
    
    for entity_id, attrs in entities.items():
        G.add_node(entity_id, **attrs)
    
    # Create relationships with attributes
    edges = [
        ("person_1", "company_1", {"relation": "works_at", "since": "2020"}),
        ("person_2", "company_1", {"relation": "works_at", "since": "2019"}),
        ("person_3", "company_2", {"relation": "works_at", "since": "2021"}),
        ("person_1", "person_2", {"relation": "knows", "context": "colleagues"}),
        ("person_2", "person_3", {"relation": "knows", "context": "friends"}),
        ("company_1", "location_1", {"relation": "headquartered_in"}),
        ("company_2", "location_2", {"relation": "headquartered_in"}),
        ("person_1", "location_1", {"relation": "lives_in"}),
        ("person_3", "location_2", {"relation": "lives_in"})
    ]
    
    G.add_edges_from(edges)
    
    return G


def test_entity_onehop_basic():
    """Test basic one-hop neighbor extraction."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph()
    graph_id = "test_graph_onehop"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test finding neighbors for person_1
    input_data = {
        "entity_ids": ["person_1"],
        "graph_id": graph_id,
        "include_edge_attributes": False
    }
    
    result = entity_onehop_neighbors(input_data, context)
    
    # Assertions
    assert "neighbors" in result
    assert "person_1" in result["neighbors"]
    
    # person_1 should have neighbors: company_1, person_2, location_1
    person_1_neighbors = result["neighbors"]["person_1"]
    neighbor_ids = [n["entity_id"] for n in person_1_neighbors]
    
    assert "company_1" in neighbor_ids
    assert "person_2" in neighbor_ids
    assert "location_1" in neighbor_ids
    assert len(neighbor_ids) == 3
    
    assert result["total_neighbors_found"] == 3
    
    logger.info("test_entity_onehop_basic passed!")


def test_entity_onehop_multiple_entities():
    """Test finding neighbors for multiple entities."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph()
    graph_id = "test_graph_onehop_multi"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test finding neighbors for person_1 and company_1
    input_data = {
        "entity_ids": ["person_1", "company_1"],
        "graph_id": graph_id,
        "include_edge_attributes": False
    }
    
    result = entity_onehop_neighbors(input_data, context)
    
    # Assertions
    assert "neighbors" in result
    assert "person_1" in result["neighbors"]
    assert "company_1" in result["neighbors"]
    
    # Check unique neighbors
    all_neighbor_ids = set()
    for entity_neighbors in result["neighbors"].values():
        for n in entity_neighbors:
            all_neighbor_ids.add(n["entity_id"])
    
    # Should include person_2, location_1 (from person_1) and person_1, person_2, location_1 (from company_1)
    assert result["total_neighbors_found"] == len(all_neighbor_ids)
    
    logger.info("test_entity_onehop_multiple_entities passed!")


def test_entity_onehop_with_edge_attributes():
    """Test including edge attributes in results."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph()
    graph_id = "test_graph_onehop_edges"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test with edge attributes
    input_data = {
        "entity_ids": ["person_1"],
        "graph_id": graph_id,
        "include_edge_attributes": True
    }
    
    result = entity_onehop_neighbors(input_data, context)
    
    # Check that edge attributes are included
    person_1_neighbors = result["neighbors"]["person_1"]
    
    # Find the company_1 neighbor
    company_neighbor = None
    for n in person_1_neighbors:
        if n["entity_id"] == "company_1":
            company_neighbor = n
            break
    
    assert company_neighbor is not None
    assert "edge_attributes" in company_neighbor
    assert len(company_neighbor["edge_attributes"]) > 0
    
    # Check edge attribute content
    edge_attr = company_neighbor["edge_attributes"][0]
    assert edge_attr["direction"] == "outgoing"
    assert edge_attr["attributes"]["relation"] == "works_at"
    assert edge_attr["attributes"]["since"] == "2020"
    
    logger.info("test_entity_onehop_with_edge_attributes passed!")


def test_entity_onehop_with_limit():
    """Test limiting number of neighbors per entity."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph with hub node
    G = nx.DiGraph()
    G.add_node("hub")
    
    # Add many neighbors
    for i in range(10):
        neighbor_id = f"neighbor_{i}"
        G.add_node(neighbor_id)
        G.add_edge("hub", neighbor_id)
    
    graph_id = "test_graph_limit"
    mock_instance = MockGraphInstance(G)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test with limit
    input_data = {
        "entity_ids": ["hub"],
        "graph_id": graph_id,
        "neighbor_limit_per_entity": 5
    }
    
    result = entity_onehop_neighbors(input_data, context)
    
    # Should only return 5 neighbors
    hub_neighbors = result["neighbors"]["hub"]
    assert len(hub_neighbors) == 5
    assert result["total_neighbors_found"] == 5
    
    logger.info("test_entity_onehop_with_limit passed!")


def test_entity_onehop_nonexistent_entity():
    """Test handling of non-existent entities."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph()
    graph_id = "test_graph_nonexistent"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test with non-existent entity
    input_data = {
        "entity_ids": ["nonexistent_entity", "person_1"],
        "graph_id": graph_id
    }
    
    result = entity_onehop_neighbors(input_data, context)
    
    # Should have empty list for non-existent entity
    assert "nonexistent_entity" in result["neighbors"]
    assert result["neighbors"]["nonexistent_entity"] == []
    
    # Should still have neighbors for existing entity
    assert "person_1" in result["neighbors"]
    assert len(result["neighbors"]["person_1"]) > 0
    
    logger.info("test_entity_onehop_nonexistent_entity passed!")


def test_entity_onehop_undirected_graph():
    """Test with undirected graph."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create undirected test graph
    G = nx.Graph()  # Undirected
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "D")
    G.add_edge("B", "D")
    
    graph_id = "test_graph_undirected"
    mock_instance = MockGraphInstance(G)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test finding neighbors
    input_data = {
        "entity_ids": ["B"],
        "graph_id": graph_id
    }
    
    result = entity_onehop_neighbors(input_data, context)
    
    # B should have neighbors: A, C, D
    b_neighbors = result["neighbors"]["B"]
    neighbor_ids = [n["entity_id"] for n in b_neighbors]
    
    assert set(neighbor_ids) == {"A", "C", "D"}
    assert result["total_neighbors_found"] == 3
    
    logger.info("test_entity_onehop_undirected_graph passed!")


def test_entity_onehop_isolated_entity():
    """Test with isolated entity (no neighbors)."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph()
    graph_id = "test_graph_isolated"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test with isolated entity
    input_data = {
        "entity_ids": ["isolated_entity"],
        "graph_id": graph_id
    }
    
    result = entity_onehop_neighbors(input_data, context)
    
    # Should have empty neighbor list
    assert "isolated_entity" in result["neighbors"]
    assert result["neighbors"]["isolated_entity"] == []
    assert result["total_neighbors_found"] == 0
    
    logger.info("test_entity_onehop_isolated_entity passed!")


def test_entity_onehop_invalid_graph():
    """Test with invalid graph ID."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Test with non-existent graph
    input_data = {
        "entity_ids": ["person_1"],
        "graph_id": "nonexistent_graph"
    }
    
    result = entity_onehop_neighbors(input_data, context)
    
    # Should return error
    assert result["neighbors"] == {}
    assert result["total_neighbors_found"] == 0
    assert "not found" in result["message"]
    
    logger.info("test_entity_onehop_invalid_graph passed!")


def test_entity_onehop_invalid_input():
    """Test with invalid input parameters."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Test with missing required field
    input_data = {
        "entity_ids": ["person_1"]
        # Missing graph_id
    }
    
    result = entity_onehop_neighbors(input_data, context)
    
    # Should return error
    assert result["neighbors"] == {}
    assert result["total_neighbors_found"] == 0
    assert "Invalid input parameters" in result["message"]
    
    logger.info("test_entity_onehop_invalid_input passed!")


if __name__ == "__main__":
    # Run all tests
    test_entity_onehop_basic()
    test_entity_onehop_multiple_entities()
    test_entity_onehop_with_edge_attributes()
    test_entity_onehop_with_limit()
    test_entity_onehop_nonexistent_entity()
    test_entity_onehop_undirected_graph()
    test_entity_onehop_isolated_entity()
    test_entity_onehop_invalid_graph()
    test_entity_onehop_invalid_input()
    
    print("\nAll Entity.Onehop tests passed successfully!")
