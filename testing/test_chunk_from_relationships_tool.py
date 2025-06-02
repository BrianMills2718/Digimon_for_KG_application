"""
Unit tests for the Chunk.FromRelationships tool
"""

import pytest
import networkx as nx
from unittest.mock import Mock, MagicMock
from Core.AgentTools.chunk_tools import chunk_from_relationships, chunk_from_relationships_tool
from Core.AgentSchema.context import GraphRAGContext


class MockGraph:
    """Mock graph wrapper that simulates the graph storage interface."""
    def __init__(self, graph):
        self._graph = graph  # Could be the storage object
        self.graph = graph   # Or direct graph access


def create_test_graph_with_chunks():
    """Create a test graph with relationships and associated chunks."""
    G = nx.DiGraph()
    
    # Add nodes with chunk data
    G.add_node("entity1", type="person", chunks=[
        {"chunk_id": "chunk_e1_1", "text": "Entity 1 description", "doc_id": "doc1", "index": 0, "tokens": 3},
        {"chunk_id": "chunk_e1_2", "text": "Entity 1 description 2", "doc_id": "doc1", "index": 1, "tokens": 4}
    ])
    G.add_node("entity2", type="organization", chunks=[
        {"chunk_id": "chunk_e2_1", "text": "Entity 2 description", "doc_id": "doc1", "index": 0, "tokens": 3}
    ])
    G.add_node("entity3", type="location")
    
    # Add edges with different relationship formats and chunk associations
    G.add_edge("entity1", "entity2", 
               id="rel1", 
               type="works_for",
               chunks=[
                   {"chunk_id": "chunk_r1_1", "text": "Entity1 works for Entity2 since 2020", "doc_id": "doc1", "index": 0, "tokens": 5},
                   {"chunk_id": "chunk_r1_2", "text": "Entity1 works for Entity2 since 2020 2", "doc_id": "doc1", "index": 1, "tokens": 6}
               ],
               text="Entity1 works for Entity2 since 2020")
    
    G.add_edge("entity2", "entity3", 
               relationship_id="rel2",
               type="located_in",
               chunks=[
                   {"chunk_id": "chunk_r2_1", "text": "Located in the downtown area", "doc_id": "doc2", "index": 0, "tokens": 5}
               ])
    
    G.add_edge("entity1", "entity3",
               rel_id="rel3",
               type="lives_in")
    
    # Edge without explicit ID
    G.add_edge("entity3", "entity1",
               type="contains",
               chunks=[
                   {"chunk_id": "chunk_r4_1", "text": "Entity3 contains Entity1", "doc_id": "doc2", "index": 0, "tokens": 4}
               ])
    
    return G


def test_basic_chunk_extraction():
    """Test basic extraction of chunks from a single relationship."""
    context = Mock(spec=GraphRAGContext)
    graph = create_test_graph_with_chunks()
    mock_graph = MockGraph(graph)
    
    # Mock the get_graph_instance method
    context.get_graph_instance = MagicMock(return_value=mock_graph)
    
    input_data = {
        "target_relationships": ["rel1"],
        "document_collection_id": "test_graph"
    }
    
    result = chunk_from_relationships(input_data, context)
    
    assert "relevant_chunks" in result
    chunks = result["relevant_chunks"]
    
    # Should have chunks from edge, plus potentially from connected nodes
    assert len(chunks) >= 3  # 2 from edge + 1 text chunk
    
    # Check that we have the edge text chunk
    text_chunks = [c for c in chunks if c.chunk_id == "rel1_text"]
    assert len(text_chunks) == 1
    assert "works for" in text_chunks[0].content
    
    # Check metadata
    for chunk in chunks:
        if hasattr(chunk, 'metadata') and chunk.metadata:
            assert chunk.metadata.get("relationship_id") == "rel1"


def test_multiple_relationships():
    """Test extraction from multiple relationships."""
    context = Mock(spec=GraphRAGContext)
    graph = create_test_graph_with_chunks()
    mock_graph = MockGraph(graph)
    
    context.get_graph_instance = MagicMock(return_value=mock_graph)
    
    input_data = {
        "target_relationships": ["rel1", "rel2"],
        "document_collection_id": "test_graph"
    }
    
    result = chunk_from_relationships(input_data, context)
    
    chunks = result["relevant_chunks"]
    assert len(chunks) > 0
    
    # Check we have chunks from both relationships
    rel_ids = set()
    for chunk in chunks:
        if hasattr(chunk, 'metadata') and chunk.metadata:
            rel_id = chunk.metadata.get("relationship_id")
            if rel_id:
                rel_ids.add(rel_id)
    
    assert "rel1" in rel_ids
    assert "rel2" in rel_ids


def test_relationship_dict_format():
    """Test handling of relationship specified as dictionary."""
    context = Mock(spec=GraphRAGContext)
    graph = create_test_graph_with_chunks()
    mock_graph = MockGraph(graph)
    
    context.get_graph_instance = MagicMock(return_value=mock_graph)
    
    input_data = {
        "target_relationships": [
            {"id": "rel1"},
            {"relationship_id": "rel2"},
            {"source": "entity3", "target": "entity1"}  # Will match entity3->entity1
        ],
        "document_collection_id": "test_graph"
    }
    
    result = chunk_from_relationships(input_data, context)
    
    chunks = result["relevant_chunks"]
    assert len(chunks) > 0


def test_max_chunks_per_relationship():
    """Test limiting chunks per relationship."""
    context = Mock(spec=GraphRAGContext)
    graph = create_test_graph_with_chunks()
    mock_graph = MockGraph(graph)
    
    context.get_graph_instance = MagicMock(return_value=mock_graph)
    
    input_data = {
        "target_relationships": ["rel1"],
        "document_collection_id": "test_graph",
        "max_chunks_per_relationship": 2
    }
    
    result = chunk_from_relationships(input_data, context)
    
    chunks = result["relevant_chunks"]
    
    # Count chunks from rel1
    rel1_chunks = [c for c in chunks if hasattr(c, 'metadata') and 
                   c.metadata and c.metadata.get("relationship_id") == "rel1"]
    
    # Should be limited to 2 per relationship
    assert len(rel1_chunks) <= 2


def test_top_k_total_limit():
    """Test overall limit on total chunks returned."""
    context = Mock(spec=GraphRAGContext)
    graph = create_test_graph_with_chunks()
    mock_graph = MockGraph(graph)
    
    context.get_graph_instance = MagicMock(return_value=mock_graph)
    
    input_data = {
        "target_relationships": ["rel1", "rel2", "rel3"],
        "document_collection_id": "test_graph",
        "top_k_total": 3
    }
    
    result = chunk_from_relationships(input_data, context)
    
    chunks = result["relevant_chunks"]
    assert len(chunks) <= 3


def test_non_existent_relationship():
    """Test handling of non-existent relationship IDs."""
    context = Mock(spec=GraphRAGContext)
    graph = create_test_graph_with_chunks()
    mock_graph = MockGraph(graph)
    
    context.get_graph_instance = MagicMock(return_value=mock_graph)
    
    input_data = {
        "target_relationships": ["non_existent_rel"],
        "document_collection_id": "test_graph"
    }
    
    result = chunk_from_relationships(input_data, context)
    
    chunks = result["relevant_chunks"]
    assert len(chunks) == 0  # No chunks should be found


def test_mixed_chunk_formats():
    """Test handling edges with different chunk storage formats."""
    context = Mock(spec=GraphRAGContext)
    graph = create_test_graph_with_chunks()
    mock_graph = MockGraph(graph)
    
    context.get_graph_instance = MagicMock(return_value=mock_graph)
    
    input_data = {
        "target_relationships": ["rel2"],  # Has dict-format chunks
        "document_collection_id": "test_graph"
    }
    
    result = chunk_from_relationships(input_data, context)
    
    chunks = result["relevant_chunks"]
    assert len(chunks) > 0
    
    # Check that dict-format chunks were properly converted
    dict_chunks = [c for c in chunks if c.chunk_id == "chunk_r2_1"]
    assert len(dict_chunks) > 0
    assert dict_chunks[0].content == "Located in the downtown area"
    assert dict_chunks[0].tokens == 5


def test_composite_edge_key():
    """Test using composite edge key (source->target) for edges without explicit ID."""
    context = Mock(spec=GraphRAGContext)
    graph = create_test_graph_with_chunks()
    mock_graph = MockGraph(graph)
    
    context.get_graph_instance = MagicMock(return_value=mock_graph)
    
    input_data = {
        "target_relationships": ["entity3->entity1"],  # Edge without explicit ID
        "document_collection_id": "test_graph"
    }
    
    result = chunk_from_relationships(input_data, context)
    
    chunks = result["relevant_chunks"]
    assert len(chunks) > 0  # Should find chunks from entity3->entity1 edge


def test_invalid_graph_id():
    """Test handling of invalid graph ID."""
    context = Mock(spec=GraphRAGContext)
    context.get_graph_instance = MagicMock(return_value=None)
    
    input_data = {
        "target_relationships": ["rel1"],
        "document_collection_id": "non_existent_graph"
    }
    
    result = chunk_from_relationships(input_data, context)
    
    assert "relevant_chunks" in result
    assert len(result["relevant_chunks"]) == 0


def test_invalid_input_parameters():
    """Test handling of invalid input parameters."""
    context = Mock(spec=GraphRAGContext)
    
    # Missing required field
    input_data = {
        "target_relationships": ["rel1"]
        # Missing document_collection_id
    }
    
    result = chunk_from_relationships(input_data, context)
    
    assert "relevant_chunks" in result
    assert len(result["relevant_chunks"]) == 0


async def test_async_wrapper():
    """Test the async wrapper function."""
    context = Mock(spec=GraphRAGContext)
    graph = create_test_graph_with_chunks()
    mock_graph = MockGraph(graph)
    
    context.get_graph_instance = MagicMock(return_value=mock_graph)
    
    input_data = {
        "target_relationships": ["rel1"],
        "document_collection_id": "test_graph"
    }
    
    result = await chunk_from_relationships_tool(input_data, context)
    
    assert "relevant_chunks" in result
    assert len(result["relevant_chunks"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
