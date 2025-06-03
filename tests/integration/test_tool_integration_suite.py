"""
Comprehensive integration test suite for all DIGIMON agent tools.
"""

import pytest
import asyncio
import uuid
from pathlib import Path
from typing import Dict, List, Any

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    # Entity tools
    EntityVDBSearchInputs, EntityPPRInputs, EntityOneHopInput, EntityRelNodeInput,
    # Relationship tools  
    RelationshipOneHopNeighborsInputs, RelationshipVDBBuildInputs, RelationshipVDBSearchInputs,
    # Chunk tools
    ChunkFromRelationshipsInputs,
    # Community tools
    CommunityRetrievalInputs,
    # Graph tools
    GraphBuildInputs, GraphAnalysisInputs, GraphVisualizationInputs,
    # Corpus tools
    CorpusPrepareInputs,
    # Entity VDB tools
    EntityVDBBuildInputs,
    # Query expansion
    QueryExpansionInputs,
    # Subgraph tools
    SubgraphExtractInputs
)

# Import all tools
from Core.AgentTools.entity_tools import (
    entity_vdb_search_tool, entity_ppr_tool, entity_onehop_tool, entity_relnode_tool
)
from Core.AgentTools.entity_vdb_tools import entity_vdb_build_tool
from Core.AgentTools.relationship_tools import (
    relationship_one_hop_neighbors_tool, relationship_vdb_build_tool, relationship_vdb_search_tool
)
from Core.AgentTools.chunk_tools import chunk_from_relationships_tool
from Core.AgentTools.community_tools import community_retrieval_tool
from Core.AgentTools.graph_construction_tools import (
    graph_build_er_tool, graph_build_rk_tool, graph_build_tree_tool,
    graph_build_tree_balanced_tool, graph_build_passage_tool
)
from Core.AgentTools.graph_analysis_tools import graph_analysis_tool
from Core.AgentTools.graph_visualization_tools import graph_visualization_tool
from Core.AgentTools.corpus_tools import corpus_prepare_tool
from Core.AgentTools.query_expansion import query_expansion_tool
from Core.AgentTools.subgraph_tools import subgraph_extract_tool

from Core.Storage.PickleBlobStorage import PickleBlobStorage
from Option.Config2 import Config
from Core.Index.EmbeddingFactory import get_rag_embedding
import networkx as nx


class TestToolIntegrationSuite:
    """Comprehensive integration tests for all tools."""
    
    @pytest.fixture
    async def test_context(self, tmp_path):
        """Create a test GraphRAG context."""
        # Create config
        config = Config.default()
        
        # Create embedding provider
        embedding_provider = get_rag_embedding(config=config)
        
        # Create storage
        storage = PickleBlobStorage(root_path=str(tmp_path))
        
        # Create context
        context = GraphRAGContext(
            project_root=str(tmp_path),
            storage=storage,
            embedding_provider=embedding_provider,
            llm_provider=None  # Not needed for most tests
        )
        
        return context
    
    @pytest.fixture
    def sample_corpus_path(self, tmp_path):
        """Create sample corpus files."""
        corpus_dir = tmp_path / "test_corpus"
        corpus_dir.mkdir()
        
        # Create test documents
        docs = [
            ("doc1.txt", "The quick brown fox jumps over the lazy dog. This is a test document about animals."),
            ("doc2.txt", "Machine learning is transforming artificial intelligence. Deep learning models require data."),
            ("doc3.txt", "Natural language processing enables computers to understand human language and text.")
        ]
        
        for filename, content in docs:
            (corpus_dir / filename).write_text(content)
        
        return str(corpus_dir)
    
    @pytest.mark.asyncio
    async def test_corpus_to_graph_pipeline(self, test_context, sample_corpus_path):
        """Test the full pipeline from corpus preparation to graph construction."""
        # Step 1: Prepare corpus
        corpus_params = CorpusPrepareInputs(
            corpus_directory=sample_corpus_path,
            output_reference_id="test_corpus",
            recursive=False,
            supported_extensions=[".txt"],
            encoding="utf-8"
        )
        
        corpus_result = await corpus_prepare_tool(corpus_params, test_context)
        assert corpus_result.corpus_reference_id == "test_corpus"
        assert corpus_result.num_documents == 3
        assert corpus_result.status == "Corpus prepared successfully"
        
        # Step 2: Build ER Graph
        graph_params = GraphBuildInputs(
            corpus_reference_id="test_corpus",
            graph_reference_id="test_er_graph",
            graph_type="er",
            force_rebuild=True
        )
        
        graph_result = await graph_build_er_tool(graph_params, test_context)
        assert graph_result.graph_reference_id == "test_er_graph"
        assert graph_result.num_nodes > 0
        assert graph_result.num_edges >= 0
        assert "Successfully" in graph_result.status
        
        # Verify graph is in context
        graph = test_context.get_graph_instance("test_er_graph")
        assert graph is not None
    
    @pytest.mark.asyncio
    async def test_entity_vdb_pipeline(self, test_context):
        """Test entity VDB build and search pipeline."""
        # Create a simple graph manually
        graph_id = "test_graph_for_vdb"
        nx_graph = nx.Graph()
        nx_graph.add_node("entity1", entity_name="Python", description="A programming language")
        nx_graph.add_node("entity2", entity_name="Machine Learning", description="A field of AI")
        nx_graph.add_node("entity3", entity_name="Data Science", description="Analysis of data")
        nx_graph.add_edge("entity1", "entity2", relation_name="used_in")
        
        # Create mock graph wrapper
        class MockGraph:
            def __init__(self, graph):
                self._graph = self
                self.graph = graph
                self.entity_metakey = "entity_name"
                self.node_num = len(graph.nodes)
            
            async def nodes_data(self):
                return [
                    {**data, "id": node_id} 
                    for node_id, data in self.graph.nodes(data=True)
                ]
            
            async def get_node_index(self, entity_name):
                for i, (node_id, data) in enumerate(self.graph.nodes(data=True)):
                    if data.get("entity_name") == entity_name:
                        return i
                return None
        
        mock_graph = MockGraph(nx_graph)
        test_context.add_graph_instance(graph_id, mock_graph)
        
        # Build entity VDB
        vdb_params = EntityVDBBuildInputs(
            graph_reference_id=graph_id,
            vdb_collection_name="test_entity_vdb",
            force_rebuild=True,
            entity_types=None,
            include_metadata=True
        )
        
        vdb_result = await entity_vdb_build_tool(vdb_params, test_context)
        assert vdb_result.vdb_reference_id == "test_entity_vdb"
        assert vdb_result.num_entities_indexed == 3
        assert "Successfully" in vdb_result.status
        
        # Search entity VDB
        search_params = EntityVDBSearchInputs(
            vdb_reference_id="test_entity_vdb",
            query_text="programming",
            top_k_results=2
        )
        
        search_result = await entity_vdb_search_tool(search_params, test_context)
        assert len(search_result.similar_entities) > 0
        assert any("Python" in entity.entity_name for entity in search_result.similar_entities)
    
    @pytest.mark.asyncio
    async def test_relationship_tools(self, test_context):
        """Test relationship extraction and VDB tools."""
        # Create test graph
        graph_id = "test_rel_graph"
        nx_graph = nx.DiGraph()
        nx_graph.add_node("A", entity_name="Node A")
        nx_graph.add_node("B", entity_name="Node B")
        nx_graph.add_node("C", entity_name="Node C")
        nx_graph.add_edge("A", "B", relation_name="connects_to", description="A connects to B")
        nx_graph.add_edge("B", "C", relation_name="links_to", description="B links to C")
        nx_graph.add_edge("A", "C", relation_name="relates_to", description="A relates to C")
        
        # Add to context with proper wrapper
        class MockGraph:
            def __init__(self, graph):
                self._graph = self
                self.graph = graph
                self.relation_metakey = "relation_name"
            
            async def edges_data(self):
                return [
                    {
                        "src_id": src,
                        "tgt_id": tgt,
                        **data
                    }
                    for src, tgt, data in self.graph.edges(data=True)
                ]
        
        mock_graph = MockGraph(nx_graph)
        test_context.add_graph_instance(graph_id, mock_graph)
        
        # Test one-hop neighbors
        onehop_params = RelationshipOneHopNeighborsInputs(
            entity_ids=["A"],
            graph_reference_id=graph_id,
            direction="both",
            relationship_types_to_include=None
        )
        
        onehop_result = await relationship_one_hop_neighbors_tool(onehop_params, test_context)
        assert len(onehop_result.one_hop_relationships) == 2  # A->B and A->C
        
        # Build relationship VDB
        rel_vdb_params = RelationshipVDBBuildInputs(
            graph_reference_id=graph_id,
            vdb_collection_name="test_rel_vdb",
            force_rebuild=True,
            relationship_types=None,
            include_metadata=True
        )
        
        rel_vdb_result = await relationship_vdb_build_tool(rel_vdb_params, test_context)
        assert rel_vdb_result.vdb_reference_id == "test_rel_vdb"
        assert rel_vdb_result.num_relationships_indexed == 3
    
    @pytest.mark.asyncio
    async def test_query_expansion(self, test_context):
        """Test query expansion tool."""
        params = QueryExpansionInputs(
            original_query="machine learning",
            expansion_method="synonyms",
            num_expansions=5
        )
        
        result = await query_expansion_tool(params, test_context)
        assert len(result.expanded_queries) > 0
        assert "machine learning" in result.expanded_queries
        # Should include related terms
        assert any(term in ' '.join(result.expanded_queries).lower() 
                  for term in ["ml", "ai", "artificial", "deep"])
    
    @pytest.mark.asyncio
    async def test_graph_analysis(self, test_context):
        """Test graph analysis tool."""
        # Create test graph
        graph_id = "test_analysis_graph"
        nx_graph = nx.karate_club_graph()  # Well-known test graph
        
        class MockGraph:
            def __init__(self, graph):
                self._graph = self
                self.graph = graph
                self.node_num = len(graph.nodes)
                self.edge_num = len(graph.edges)
        
        mock_graph = MockGraph(nx_graph)
        test_context.add_graph_instance(graph_id, mock_graph)
        
        # Analyze graph
        params = GraphAnalysisInputs(
            graph_reference_id=graph_id,
            metrics_to_compute=["basic", "centrality", "clustering"],
            top_k_central_nodes=5
        )
        
        result = await graph_analysis_tool(params, test_context)
        assert result.num_nodes == 34  # Karate club has 34 nodes
        assert result.num_edges == 78  # Karate club has 78 edges
        assert result.density > 0
        assert len(result.top_nodes_by_degree) <= 5
        assert result.clustering_coefficient >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_context):
        """Test error handling in tools."""
        # Test with non-existent graph
        params = EntityVDBSearchInputs(
            vdb_reference_id="non_existent_vdb",
            query_text="test",
            top_k_results=5
        )
        
        result = await entity_vdb_search_tool(params, test_context)
        assert len(result.similar_entities) == 0  # Should return empty, not crash
        
        # Test with invalid corpus path
        corpus_params = CorpusPrepareInputs(
            corpus_directory="/non/existent/path",
            output_reference_id="test",
            recursive=False
        )
        
        corpus_result = await corpus_prepare_tool(corpus_params, test_context)
        assert "Error" in corpus_result.status
        assert corpus_result.num_documents == 0
    
    @pytest.mark.asyncio
    async def test_tool_chaining(self, test_context, sample_corpus_path):
        """Test chaining multiple tools together."""
        # 1. Prepare corpus
        corpus_result = await corpus_prepare_tool(
            CorpusPrepareInputs(
                corpus_directory=sample_corpus_path,
                output_reference_id="chain_corpus"
            ),
            test_context
        )
        
        # 2. Build graph
        graph_result = await graph_build_er_tool(
            GraphBuildInputs(
                corpus_reference_id="chain_corpus",
                graph_reference_id="chain_graph",
                graph_type="er"
            ),
            test_context
        )
        
        # 3. Build entity VDB
        vdb_result = await entity_vdb_build_tool(
            EntityVDBBuildInputs(
                graph_reference_id="chain_graph",
                vdb_collection_name="chain_vdb"
            ),
            test_context
        )
        
        # 4. Search VDB
        search_result = await entity_vdb_search_tool(
            EntityVDBSearchInputs(
                vdb_reference_id="chain_vdb",
                query_text="learning",
                top_k_results=3
            ),
            test_context
        )
        
        # Verify chain worked
        assert corpus_result.num_documents > 0
        assert graph_result.num_nodes > 0
        assert vdb_result.num_entities_indexed > 0
        # May or may not find results depending on graph extraction


@pytest.mark.integration
class TestToolPerformance:
    """Performance tests for tools."""
    
    @pytest.mark.asyncio
    async def test_large_corpus_handling(self, test_context, tmp_path):
        """Test handling of larger corpus."""
        # Create 100 documents
        corpus_dir = tmp_path / "large_corpus"
        corpus_dir.mkdir()
        
        for i in range(100):
            content = f"Document {i}: " + " ".join([f"word{j}" for j in range(100)])
            (corpus_dir / f"doc{i}.txt").write_text(content)
        
        # Time corpus preparation
        import time
        start = time.time()
        
        result = await corpus_prepare_tool(
            CorpusPrepareInputs(
                corpus_directory=str(corpus_dir),
                output_reference_id="large_corpus"
            ),
            test_context
        )
        
        duration = time.time() - start
        
        assert result.num_documents == 100
        assert duration < 10  # Should complete in reasonable time
        print(f"Processed 100 documents in {duration:.2f}s")
    
    @pytest.mark.asyncio 
    async def test_concurrent_tool_execution(self, test_context):
        """Test concurrent execution of multiple tools."""
        # Create multiple VDB search tasks
        tasks = []
        
        # Assume we have a VDB set up (would need to set up in real test)
        for i in range(10):
            params = QueryExpansionInputs(
                original_query=f"query {i}",
                expansion_method="synonyms",
                num_expansions=3
            )
            task = query_expansion_tool(params, test_context)
            tasks.append(task)
        
        # Execute concurrently
        start = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start
        
        assert len(results) == 10
        assert all(len(r.expanded_queries) > 0 for r in results)
        print(f"Executed 10 concurrent expansions in {duration:.2f}s")