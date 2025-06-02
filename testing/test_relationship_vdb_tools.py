"""
Unit tests for Relationship VDB Build and Search tools.
"""

import pytest
import networkx as nx
from unittest.mock import MagicMock, AsyncMock, patch

from Core.AgentTools.relationship_tools import (
    relationship_vdb_build_tool,
    relationship_vdb_search_tool
)
from Core.AgentSchema.tool_contracts import (
    RelationshipVDBBuildInputs,
    RelationshipVDBBuildOutputs,
    RelationshipVDBSearchInputs,
    RelationshipVDBSearchOutputs
)
from Core.AgentSchema.context import GraphRAGContext


class TestRelationshipVDBBuildTool:
    """Test suite for relationship_vdb_build_tool"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_context = MagicMock(spec=GraphRAGContext)
        self.mock_embedding_provider = MagicMock()
        self.mock_context.embedding_provider = self.mock_embedding_provider
        
        # Create test graph
        self.test_graph = nx.DiGraph()
        self.test_graph.add_edge(
            "entity1", "entity2",
            id="rel1",
            type="works_for",
            description="Entity1 works for Entity2",
            weight=0.8
        )
        self.test_graph.add_edge(
            "entity2", "entity3",
            id="rel2",
            type="located_in",
            description="Entity2 is located in Entity3",
            weight=0.9
        )
        self.test_graph.add_edge(
            "entity1", "entity3",
            relation_name="collaborates_with",  # Alternative field name
            description="Entity1 collaborates with Entity3"
        )
    
    @pytest.mark.asyncio
    async def test_build_vdb_success(self):
        """Test successful VDB build"""
        # Setup
        graph_wrapper = MagicMock()
        graph_wrapper._graph = MagicMock()
        graph_wrapper._graph.graph = self.test_graph
        
        self.mock_context.get_graph_instance.return_value = graph_wrapper
        self.mock_context.get_vdb_instance.return_value = None
        
        # Mock VDB creation
        mock_vdb = AsyncMock()
        mock_vdb.build_index = AsyncMock()
        
        with patch('Core.AgentTools.relationship_tools.FaissIndex', return_value=mock_vdb):
            with patch('Core.AgentTools.relationship_tools.PickleBlobStorage'):
                # Execute
                params = RelationshipVDBBuildInputs(
                    graph_reference_id="test_graph",
                    vdb_collection_name="test_rel_vdb",
                    embedding_fields=["type", "description"],
                    include_metadata=True,
                    force_rebuild=False
                )
                
                result = await relationship_vdb_build_tool(params, self.mock_context)
        
        # Verify
        assert isinstance(result, RelationshipVDBBuildOutputs)
        assert result.vdb_reference_id == "test_rel_vdb_relationships"
        assert result.num_relationships_indexed == 3
        assert "Successfully built VDB with 3 relationships" in result.status
        
        # Verify VDB was registered
        self.mock_context.add_vdb_instance.assert_called_once_with(
            "test_rel_vdb_relationships",
            mock_vdb
        )
        
        # Verify build_index was called with correct data
        mock_vdb.build_index.assert_called_once()
        call_args = mock_vdb.build_index.call_args
        elements = call_args[1]['elements']
        assert len(elements) == 3
        
        # Check first relationship
        rel1 = next(e for e in elements if e['id'] == 'rel1')
        assert 'type: works_for' in rel1['content']  # Type is in content, not a separate field
        assert 'description: Entity1 works for Entity2' in rel1['content']
        assert rel1['source'] == 'entity1'
        assert rel1['target'] == 'entity2'
        # Since include_metadata=True, type should be in the document as metadata
        assert rel1.get('type') == 'works_for' or 'type: works_for' in rel1['content']
    
    @pytest.mark.asyncio
    async def test_build_vdb_graph_not_found(self):
        """Test VDB build when graph is not found"""
        self.mock_context.get_graph_instance.return_value = None
        
        params = RelationshipVDBBuildInputs(
            graph_reference_id="nonexistent_graph",
            vdb_collection_name="test_vdb",
            embedding_fields=["type"],
            include_metadata=False,
            force_rebuild=False
        )
        
        result = await relationship_vdb_build_tool(params, self.mock_context)
        
        assert result.vdb_reference_id == ""
        assert result.num_relationships_indexed == 0
        assert "Graph 'nonexistent_graph' not found" in result.status
    
    @pytest.mark.asyncio
    async def test_build_vdb_already_exists(self):
        """Test VDB build when VDB already exists and force_rebuild=False"""
        graph_wrapper = MagicMock()
        graph_wrapper._graph = MagicMock()
        graph_wrapper._graph.graph = self.test_graph
        
        self.mock_context.get_graph_instance.return_value = graph_wrapper
        self.mock_context.get_vdb_instance.return_value = MagicMock()  # Existing VDB
        
        params = RelationshipVDBBuildInputs(
            graph_reference_id="test_graph",
            vdb_collection_name="existing_vdb",
            embedding_fields=["type"],
            include_metadata=False,
            force_rebuild=False
        )
        
        result = await relationship_vdb_build_tool(params, self.mock_context)
        
        assert result.vdb_reference_id == "existing_vdb_relationships"
        assert result.num_relationships_indexed == 3  # Should show number of edges even if not rebuilt
        assert "VDB already exists" in result.status
    
    @pytest.mark.asyncio
    async def test_build_vdb_no_relationships(self):
        """Test VDB build when graph has no relationships"""
        empty_graph = nx.DiGraph()
        empty_graph.add_node("entity1")
        
        graph_wrapper = MagicMock()
        graph_wrapper._graph = MagicMock()
        graph_wrapper._graph.graph = empty_graph
        
        self.mock_context.get_graph_instance.return_value = graph_wrapper
        self.mock_context.get_vdb_instance.return_value = None
        
        params = RelationshipVDBBuildInputs(
            graph_reference_id="empty_graph",
            vdb_collection_name="empty_vdb",
            embedding_fields=["type"],
            include_metadata=False,
            force_rebuild=False
        )
        
        result = await relationship_vdb_build_tool(params, self.mock_context)
        
        assert result.vdb_reference_id == ""
        assert result.num_relationships_indexed == 0
        assert "No relationships found" in result.status
    
    @pytest.mark.asyncio
    async def test_build_vdb_no_embedding_provider(self):
        """Test VDB build when no embedding provider is available"""
        graph_wrapper = MagicMock()
        graph_wrapper._graph = MagicMock()
        graph_wrapper._graph.graph = self.test_graph
        
        self.mock_context.get_graph_instance.return_value = graph_wrapper
        self.mock_context.get_vdb_instance.return_value = None
        self.mock_context.embedding_provider = None
        
        params = RelationshipVDBBuildInputs(
            graph_reference_id="test_graph",
            vdb_collection_name="test_vdb",
            embedding_fields=["type"],
            include_metadata=False,
            force_rebuild=False
        )
        
        result = await relationship_vdb_build_tool(params, self.mock_context)
        
        assert result.vdb_reference_id == ""
        assert result.num_relationships_indexed == 0
        assert "No embedding provider" in result.status


class TestRelationshipVDBSearchTool:
    """Test suite for relationship_vdb_search_tool"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_context = MagicMock(spec=GraphRAGContext)
        self.mock_vdb = AsyncMock()
        
    @pytest.mark.asyncio
    async def test_search_with_text_query(self):
        """Test search with text query"""
        self.mock_context.get_vdb_instance.return_value = self.mock_vdb
        
        # Mock search results
        self.mock_vdb.search.return_value = [
            {"id": "rel1", "content": "Entity1 works for Entity2", "score": 0.95},
            {"id": "rel2", "content": "Entity2 manages Entity3", "score": 0.85},
            {"id": "rel3", "content": "Entity1 reports to Entity2", "score": 0.75}
        ]
        
        params = RelationshipVDBSearchInputs(
            vdb_reference_id="test_vdb",
            query_text="employment relationships",
            query_embedding=None,
            top_k=5,
            score_threshold=None
        )
        
        result = await relationship_vdb_search_tool(params, self.mock_context)
        
        assert isinstance(result, RelationshipVDBSearchOutputs)
        assert len(result.similar_relationships) == 3
        
        # Check results are sorted by score
        assert result.similar_relationships[0][0] == "rel1"
        assert result.similar_relationships[0][2] == 0.95
        assert result.similar_relationships[1][0] == "rel2"
        assert result.similar_relationships[1][2] == 0.85
        
        # Check metadata
        assert result.metadata["num_results"] == 3
        assert result.metadata["query_type"] == "text"
        
        # Verify VDB search was called correctly
        self.mock_vdb.search.assert_called_once_with(
            query="employment relationships",
            k=5
        )
    
    @pytest.mark.asyncio
    async def test_search_with_embedding(self):
        """Test search with pre-computed embedding"""
        self.mock_context.get_vdb_instance.return_value = self.mock_vdb
        
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        self.mock_vdb.search_by_embedding.return_value = [
            {"id": "rel1", "content": "Similar rel 1", "score": 0.9}
        ]
        
        params = RelationshipVDBSearchInputs(
            vdb_reference_id="test_vdb",
            query_text=None,
            query_embedding=test_embedding,
            top_k=3,
            score_threshold=None
        )
        
        result = await relationship_vdb_search_tool(params, self.mock_context)
        
        assert len(result.similar_relationships) == 1
        assert result.metadata["query_type"] == "embedding"
        
        self.mock_vdb.search_by_embedding.assert_called_once_with(
            embedding=test_embedding,
            k=3
        )
    
    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self):
        """Test search with score threshold filtering"""
        self.mock_context.get_vdb_instance.return_value = self.mock_vdb
        
        self.mock_vdb.search.return_value = [
            {"id": "rel1", "content": "High score", "score": 0.95},
            {"id": "rel2", "content": "Medium score", "score": 0.75},
            {"id": "rel3", "content": "Low score", "score": 0.5}
        ]
        
        params = RelationshipVDBSearchInputs(
            vdb_reference_id="test_vdb",
            query_text="test query",
            query_embedding=None,
            top_k=10,
            score_threshold=0.7
        )
        
        result = await relationship_vdb_search_tool(params, self.mock_context)
        
        # Only relationships with score >= 0.7 should be returned
        assert len(result.similar_relationships) == 2
        assert all(score >= 0.7 for _, _, score in result.similar_relationships)
    
    @pytest.mark.asyncio
    async def test_search_vdb_not_found(self):
        """Test search when VDB is not found"""
        self.mock_context.get_vdb_instance.return_value = None
        
        params = RelationshipVDBSearchInputs(
            vdb_reference_id="nonexistent_vdb",
            query_text="test query",
            query_embedding=None,
            top_k=5,
            score_threshold=None
        )
        
        result = await relationship_vdb_search_tool(params, self.mock_context)
        
        assert len(result.similar_relationships) == 0
        assert "error" in result.metadata
        assert "not found" in result.metadata["error"]
    
    @pytest.mark.asyncio
    async def test_search_no_query_provided(self):
        """Test search when neither text nor embedding is provided"""
        params = RelationshipVDBSearchInputs(
            vdb_reference_id="test_vdb",
            query_text=None,
            query_embedding=None,
            top_k=5,
            score_threshold=None
        )
        
        result = await relationship_vdb_search_tool(params, self.mock_context)
        
        assert len(result.similar_relationships) == 0
        assert "error" in result.metadata
        assert "query_text or query_embedding must be provided" in result.metadata["error"]
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Test error handling during search"""
        self.mock_context.get_vdb_instance.return_value = self.mock_vdb
        self.mock_vdb.search.side_effect = Exception("Search failed")
        
        params = RelationshipVDBSearchInputs(
            vdb_reference_id="test_vdb",
            query_text="test query",
            query_embedding=None,
            top_k=5,
            score_threshold=None
        )
        
        result = await relationship_vdb_search_tool(params, self.mock_context)
        
        assert len(result.similar_relationships) == 0
        assert "error" in result.metadata
        assert "Search failed" in result.metadata["error"]
