# testing/test_entity_relnode_tool.py

import json
import os
import sys
import pytest
import networkx as nx
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.AgentTools.entity_relnode_tools import entity_relnode_extract
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


def create_test_graph_with_relationships():
    """Create a test graph with explicit relationship IDs."""
    G = nx.DiGraph()
    
    # Create entities
    entities = {
        "person_1": {"type": "Person", "name": "Alice"},
        "person_2": {"type": "Person", "name": "Bob"},
        "person_3": {"type": "Person", "name": "Charlie"},
        "company_1": {"type": "Company", "name": "TechCorp"},
        "company_2": {"type": "Company", "name": "DataInc"},
        "location_1": {"type": "Location", "name": "San Francisco"},
        "project_1": {"type": "Project", "name": "AI Research"}
    }
    
    for entity_id, attrs in entities.items():
        G.add_node(entity_id, **attrs)
    
    # Create relationships with explicit IDs
    relationships = [
        ("person_1", "company_1", {"id": "rel_001", "relation": "works_at", "since": "2020"}),
        ("person_2", "company_1", {"id": "rel_002", "relation": "works_at", "since": "2019"}),
        ("person_3", "company_2", {"id": "rel_003", "relation": "works_at", "since": "2021"}),
        ("person_1", "person_2", {"id": "rel_004", "relation": "collaborates_with"}),
        ("person_2", "person_3", {"id": "rel_005", "relation": "knows"}),
        ("company_1", "location_1", {"id": "rel_006", "relation": "headquartered_in"}),
        ("person_1", "project_1", {"id": "rel_007", "relation": "leads"}),
        ("person_2", "project_1", {"id": "rel_008", "relation": "contributes_to"}),
        ("person_3", "project_1", {"id": "rel_009", "relation": "contributes_to"})
    ]
    
    G.add_edges_from(relationships)
    
    return G


def test_entity_relnode_basic():
    """Test basic entity extraction from relationships."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph_with_relationships()
    graph_id = "test_graph_relnode"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test extracting entities from specific relationships
    input_data = {
        "relationship_ids": ["rel_001", "rel_002"],
        "graph_id": graph_id
    }
    
    result = entity_relnode_extract(input_data, context)
    
    # Assertions
    assert "entities" in result
    assert len(result["entities"]) > 0
    
    # Should have person_1, person_2, and company_1
    entity_ids = [e["entity_id"] for e in result["entities"]]
    assert "person_1" in entity_ids
    assert "person_2" in entity_ids
    assert "company_1" in entity_ids
    assert result["entity_count"] == 3
    
    # Check relationship mapping
    assert "rel_001" in result["relationship_entity_map"]
    assert "rel_002" in result["relationship_entity_map"]
    assert set(result["relationship_entity_map"]["rel_001"]) == {"person_1", "company_1"}
    assert set(result["relationship_entity_map"]["rel_002"]) == {"person_2", "company_1"}
    
    logger.info("test_entity_relnode_basic passed!")


def test_entity_relnode_with_role_filter():
    """Test entity extraction with role filtering."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph_with_relationships()
    graph_id = "test_graph_relnode_role"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test extracting only source entities
    input_data = {
        "relationship_ids": ["rel_001", "rel_002", "rel_003"],
        "graph_id": graph_id,
        "entity_role_filter": "source"
    }
    
    result = entity_relnode_extract(input_data, context)
    
    # Should only have person entities (sources)
    entity_ids = [e["entity_id"] for e in result["entities"]]
    assert "person_1" in entity_ids
    assert "person_2" in entity_ids
    assert "person_3" in entity_ids
    assert "company_1" not in entity_ids
    assert "company_2" not in entity_ids
    
    # Test extracting only target entities
    input_data["entity_role_filter"] = "target"
    result = entity_relnode_extract(input_data, context)
    
    # Should only have company entities (targets)
    entity_ids = [e["entity_id"] for e in result["entities"]]
    assert "company_1" in entity_ids
    assert "company_2" in entity_ids
    assert "person_1" not in entity_ids
    assert "person_2" not in entity_ids
    assert "person_3" not in entity_ids
    
    logger.info("test_entity_relnode_with_role_filter passed!")


def test_entity_relnode_with_type_filter():
    """Test entity extraction with type filtering."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph_with_relationships()
    graph_id = "test_graph_relnode_type"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test extracting only Person entities
    input_data = {
        "relationship_ids": ["rel_001", "rel_002", "rel_004", "rel_006"],
        "graph_id": graph_id,
        "entity_type_filter": ["Person"]
    }
    
    result = entity_relnode_extract(input_data, context)
    
    # Should only have Person entities
    for entity in result["entities"]:
        assert entity["attributes"]["type"] == "Person"
    
    # Test extracting Company and Location entities
    input_data["entity_type_filter"] = ["Company", "Location"]
    result = entity_relnode_extract(input_data, context)
    
    # Should only have Company and Location entities
    for entity in result["entities"]:
        assert entity["attributes"]["type"] in ["Company", "Location"]
    
    logger.info("test_entity_relnode_with_type_filter passed!")


def test_entity_relnode_multiple_relationships():
    """Test entity appearing in multiple relationships."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph_with_relationships()
    graph_id = "test_graph_relnode_multi"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test with relationships involving person_1
    input_data = {
        "relationship_ids": ["rel_001", "rel_004", "rel_007"],
        "graph_id": graph_id
    }
    
    result = entity_relnode_extract(input_data, context)
    
    # Find person_1 in results
    person_1_entity = None
    for entity in result["entities"]:
        if entity["entity_id"] == "person_1":
            person_1_entity = entity
            break
    
    assert person_1_entity is not None
    # Check that person_1 is associated with multiple relationships
    rel_ids = person_1_entity["relationship_id"]
    if isinstance(rel_ids, list):
        assert len(rel_ids) >= 2  # Should be in at least 2 relationships
    
    logger.info("test_entity_relnode_multiple_relationships passed!")


def test_entity_relnode_edge_pattern():
    """Test using edge patterns (source->target) as relationship IDs."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph_with_relationships()
    graph_id = "test_graph_relnode_pattern"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test using edge patterns
    input_data = {
        "relationship_ids": ["person_1->company_1", "person_2->person_3"],
        "graph_id": graph_id
    }
    
    result = entity_relnode_extract(input_data, context)
    
    # Should find entities from these edges
    entity_ids = [e["entity_id"] for e in result["entities"]]
    assert "person_1" in entity_ids
    assert "company_1" in entity_ids
    assert "person_2" in entity_ids
    assert "person_3" in entity_ids
    
    logger.info("test_entity_relnode_edge_pattern passed!")


def test_entity_relnode_nonexistent_relationships():
    """Test handling of non-existent relationships."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph_with_relationships()
    graph_id = "test_graph_relnode_nonexist"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test with non-existent relationship IDs
    input_data = {
        "relationship_ids": ["rel_999", "rel_888"],
        "graph_id": graph_id
    }
    
    result = entity_relnode_extract(input_data, context)
    
    # Should return empty results
    assert result["entities"] == []
    assert result["entity_count"] == 0
    assert result["relationship_entity_map"] == {}
    assert "No relationships found" in result["message"]
    
    logger.info("test_entity_relnode_nonexistent_relationships passed!")


def test_entity_relnode_mixed_relationships():
    """Test with mix of existing and non-existing relationships."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Create test graph and add to context
    test_graph = create_test_graph_with_relationships()
    graph_id = "test_graph_relnode_mixed"
    
    mock_instance = MockGraphInstance(test_graph)
    context.add_graph_instance(graph_id, mock_instance)
    
    # Test with mix of valid and invalid IDs
    input_data = {
        "relationship_ids": ["rel_001", "rel_999", "rel_002"],
        "graph_id": graph_id
    }
    
    result = entity_relnode_extract(input_data, context)
    
    # Should only process valid relationships
    assert "rel_001" in result["relationship_entity_map"]
    assert "rel_002" in result["relationship_entity_map"]
    assert "rel_999" not in result["relationship_entity_map"]
    assert result["entity_count"] == 3  # person_1, person_2, company_1
    
    logger.info("test_entity_relnode_mixed_relationships passed!")


def test_entity_relnode_invalid_graph():
    """Test with invalid graph ID."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Test with non-existent graph
    input_data = {
        "relationship_ids": ["rel_001"],
        "graph_id": "nonexistent_graph"
    }
    
    result = entity_relnode_extract(input_data, context)
    
    # Should return error
    assert result["entities"] == []
    assert result["entity_count"] == 0
    assert "not found" in result["message"]
    
    logger.info("test_entity_relnode_invalid_graph passed!")


def test_entity_relnode_invalid_input():
    """Test with invalid input parameters."""
    # Setup
    workspace = "/tmp/test_workspace"
    namespace = "test_namespace"
    context = create_test_context(workspace, namespace)
    
    # Test with missing required field
    input_data = {
        "relationship_ids": ["rel_001"]
        # Missing graph_id
    }
    
    result = entity_relnode_extract(input_data, context)
    
    # Should return error
    assert result["entities"] == []
    assert result["entity_count"] == 0
    assert "Invalid input parameters" in result["message"]
    
    logger.info("test_entity_relnode_invalid_input passed!")


if __name__ == "__main__":
    # Run all tests
    test_entity_relnode_basic()
    test_entity_relnode_with_role_filter()
    test_entity_relnode_with_type_filter()
    test_entity_relnode_multiple_relationships()
    test_entity_relnode_edge_pattern()
    test_entity_relnode_nonexistent_relationships()
    test_entity_relnode_mixed_relationships()
    test_entity_relnode_invalid_graph()
    test_entity_relnode_invalid_input()
    
    print("\nAll Entity.RelNode tests passed successfully!")
