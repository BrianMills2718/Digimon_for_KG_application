"""
Test DIGIMON MCP Tool Server - Integration of 5 Core Tools
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from Core.MCP.digimon_mcp_server import DigimonToolServer
from Core.MCP.mcp_server import MCPRequest, MCPResponse
from Core.AgentSchema.context import GraphRAGContext


class TestDigimonMCPTools:
    """Test DIGIMON MCP Tool Server with 5 core tools"""
    
    @pytest.fixture
    async def tool_server(self):
        """Create a test tool server instance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal config file
            config_path = Path(tmpdir) / "test_config.yaml"
            config_content = """
llm:
  api_type: "openai"
  model: "gpt-3.5-turbo"
  api_key: "test-key"
  temperature: 0.1

embedding:
  api_type: "openai"
  model: "text-embedding-ada-002"
  api_key: "test-key"

chunk:
  overlap_token_size: 10
  chunk_token_size: 300
  chunk_type: "text"

graph:
  type: "networkx"
  enabled_entities: ["entity"]
  enabled_relations: ["relations"]

query:
  response_type: "default"
  tree_search: false
"""
            config_path.write_text(config_content)
            
            server = DigimonToolServer(config_path=str(config_path))
            
            # Mock the core components
            server.config = MagicMock()
            server.llm_provider = AsyncMock()
            server.embedding_provider = AsyncMock()
            server.chunk_factory = AsyncMock()
            
            yield server
    
    @pytest.mark.asyncio
    async def test_entity_vdb_search_tool(self, tool_server):
        """Test Entity.VDBSearch tool through MCP"""
        # Setup mock context and VDB
        mock_context = MagicMock(spec=GraphRAGContext)
        mock_vdb = AsyncMock()
        mock_vdb.retrieval.return_value = [
            MagicMock(
                node=MagicMock(
                    metadata={"name": "French Revolution", "id": "e1"},
                    node_id="node1"
                ),
                score=0.95
            ),
            MagicMock(
                node=MagicMock(
                    metadata={"name": "American Revolution", "id": "e2"},
                    node_id="node2"
                ),
                score=0.85
            )
        ]
        mock_context.get_vdb_instance.return_value = mock_vdb
        
        with patch.object(tool_server, 'get_or_create_context', return_value=mock_context):
            result = await tool_server.entity_vdb_search_wrapper(
                vdb_reference_id="test_vdb",
                query_text="revolution",
                top_k_results=5,
                session_id="test-session",
                dataset_name="test-dataset"
            )
        
        assert isinstance(result, dict)
        assert "similar_entities" in result
        assert len(result["similar_entities"]) == 2
        assert result["similar_entities"][0]["entity_name"] == "French Revolution"
        assert result["similar_entities"][0]["score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_corpus_prepare_tool(self, tool_server):
        """Test Corpus.Prepare tool through MCP"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test input files
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            (input_dir / "revolution1.txt").write_text("The French Revolution began in 1789")
            (input_dir / "revolution2.txt").write_text("The American Revolution preceded the French")
            
            output_dir = Path(tmpdir) / "output"
            
            result = await tool_server.corpus_prepare_wrapper(
                input_directory_path=str(input_dir),
                output_directory_path=str(output_dir),
                target_corpus_name="revolutions_corpus",
                session_id="test-session"
            )
        
        assert result["status"] == "success"
        assert result["document_count"] == 2
        assert "corpus_json_path" in result
        
        # Verify corpus file was created
        corpus_path = Path(result["corpus_json_path"])
        assert corpus_path.exists()
        
        # Verify corpus content
        with open(corpus_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            doc1 = json.loads(lines[0])
            assert doc1["title"] == "revolution1"
            assert "French Revolution" in doc1["content"]
    
    @pytest.mark.asyncio
    async def test_graph_build_tool(self, tool_server):
        """Test Graph.Build tool through MCP"""
        # Mock the build_er_graph function
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.graph_id = "revolutions_ERGraph"
        mock_result.node_count = 25
        mock_result.edge_count = 40
        mock_result.artifact_path = "/test/revolutions/graph"
        mock_result.model_dump.return_value = {
            "graph_id": "revolutions_ERGraph",
            "status": "success",
            "message": "ERGraph built successfully for revolutions",
            "node_count": 25,
            "edge_count": 40,
            "artifact_path": "/test/revolutions/graph"
        }
        
        with patch('Core.MCP.digimon_mcp_server.build_er_graph', return_value=mock_result):
            result = await tool_server.graph_build_wrapper(
                target_dataset_name="revolutions",
                force_rebuild=False,
                session_id="test-session"
            )
        
        assert result["status"] == "success"
        assert result["graph_id"] == "revolutions_ERGraph"
        assert result["node_count"] == 25
        assert result["edge_count"] == 40
        assert "ERGraph built successfully" in result["message"]
    
    @pytest.mark.asyncio
    async def test_chunk_retrieve_tool(self, tool_server):
        """Test Chunk.Retrieve tool through MCP"""
        # Setup mock context
        mock_context = MagicMock(spec=GraphRAGContext)
        mock_context.chunk_storage_manager = AsyncMock()
        
        expected_result = {
            "retrieved_chunks": [
                {
                    "entity_id": "French Revolution",
                    "chunk_id": "chunk_fr_1",
                    "text_content": "The French Revolution was a period of radical political and societal change in France that began with the Estates-General of 1789.",
                    "metadata": {"doc_id": "doc1", "index": 0}
                },
                {
                    "entity_id": "American Revolution",
                    "chunk_id": "chunk_am_1", 
                    "text_content": "The American Revolution was a colonial revolt that occurred between 1765 and 1783.",
                    "metadata": {"doc_id": "doc2", "index": 0}
                }
            ],
            "status_message": "Retrieved 2 chunks for 2 entities"
        }
        
        with patch.object(tool_server, 'get_or_create_context', return_value=mock_context):
            with patch('Core.MCP.digimon_mcp_server.chunk_get_text_for_entities_tool', return_value=expected_result):
                result = await tool_server.chunk_retrieve_wrapper(
                    entity_ids=["French Revolution", "American Revolution"],
                    graph_reference_id="revolutions_graph",
                    session_id="test-session",
                    dataset_name="revolutions"
                )
        
        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) == 2
        assert result["retrieved_chunks"][0]["entity_id"] == "French Revolution"
        assert "radical political" in result["retrieved_chunks"][0]["text_content"]
    
    @pytest.mark.asyncio
    async def test_answer_generate_tool(self, tool_server):
        """Test Answer.Generate tool through MCP"""
        # Mock the LLM response
        tool_server.llm_provider.aask = AsyncMock(
            return_value="The French Revolution and American Revolution were both significant political upheavals that challenged existing monarchical systems. The American Revolution (1765-1783) inspired the French Revolution (1789-1799) with its ideas of liberty and democracy."
        )
        
        context = """
        Entities: French Revolution (1789-1799), American Revolution (1765-1783)
        Relationships: American Revolution influenced French Revolution
        Text: Both revolutions challenged monarchical rule and promoted democratic ideals.
        """
        
        result = await tool_server.answer_generate_wrapper(
            query="How were the French and American Revolutions related?",
            context=context,
            response_type="default",
            use_tree_search=False
        )
        
        assert result["status"] == "success"
        assert "French Revolution" in result["answer"]
        assert "American Revolution" in result["answer"]
        assert result["query"] == "How were the French and American Revolutions related?"
    
    @pytest.mark.asyncio
    async def test_tool_integration_workflow(self, tool_server):
        """Test a complete workflow using multiple tools"""
        # This test simulates a realistic workflow:
        # 1. Prepare corpus
        # 2. Build graph
        # 3. Search entities
        # 4. Retrieve chunks
        # 5. Generate answer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Prepare corpus
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            (input_dir / "doc1.txt").write_text("The French Revolution transformed France.")
            
            output_dir = Path(tmpdir) / "output"
            
            corpus_result = await tool_server.corpus_prepare_wrapper(
                input_directory_path=str(input_dir),
                output_directory_path=str(output_dir),
                target_corpus_name="test_corpus",
                session_id="workflow-session"
            )
            
            assert corpus_result["status"] == "success"
            
            # Step 2: Build graph (mocked)
            mock_graph_result = MagicMock()
            mock_graph_result.status = "success"
            mock_graph_result.graph_id = "test_ERGraph"
            mock_graph_result.model_dump.return_value = {
                "graph_id": "test_ERGraph",
                "status": "success",
                "message": "Graph built",
                "node_count": 10,
                "edge_count": 5
            }
            
            with patch('Core.MCP.digimon_mcp_server.build_er_graph', return_value=mock_graph_result):
                graph_result = await tool_server.graph_build_wrapper(
                    target_dataset_name="test",
                    session_id="workflow-session"
                )
            
            assert graph_result["status"] == "success"
            
            # Steps 3-5 would follow similar pattern...
    
    @pytest.mark.asyncio 
    async def test_error_handling_all_tools(self, tool_server):
        """Test error handling for all 5 tools"""
        
        # Test Entity.VDBSearch error
        with patch.object(tool_server, 'get_or_create_context', side_effect=Exception("VDB error")):
            result = await tool_server.entity_vdb_search_wrapper(
                vdb_reference_id="bad_vdb",
                query_text="test",
                session_id="error-test"
            )
        assert "error" in result
        assert result["similar_entities"] == []
        
        # Test Corpus.Prepare error - invalid directory
        result = await tool_server.corpus_prepare_wrapper(
            input_directory_path="/nonexistent/path",
            output_directory_path="/tmp/out",
            session_id="error-test"
        )
        assert result["status"] == "failure"
        assert result["document_count"] == 0
        
        # Test Graph.Build error
        with patch('Core.MCP.digimon_mcp_server.build_er_graph', side_effect=Exception("Graph build failed")):
            result = await tool_server.graph_build_wrapper(
                target_dataset_name="bad_dataset",
                session_id="error-test"
            )
        assert result["status"] == "failure"
        assert "Graph build failed" in result["message"]
        
        # Test Chunk.Retrieve error
        with patch.object(tool_server, 'get_or_create_context', side_effect=Exception("Chunk error")):
            result = await tool_server.chunk_retrieve_wrapper(
                entity_ids=["bad_entity"],
                graph_reference_id="bad_graph",
                session_id="error-test"
            )
        assert result["retrieved_chunks"] == []
        assert "Error:" in result["status_message"]
        
        # Test Answer.Generate error
        with patch.object(tool_server, 'llm_provider', None):
            result = await tool_server.answer_generate_wrapper(
                query="test query",
                context="test context"
            )
        assert result["status"] == "error"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_session_context_isolation(self, tool_server):
        """Test that different sessions maintain isolated contexts"""
        await tool_server.initialize()
        
        # Create contexts for different sessions
        context1 = await tool_server.get_or_create_context("session1", "dataset1")
        context2 = await tool_server.get_or_create_context("session2", "dataset2")
        
        # Verify isolation
        assert context1 is not context2
        assert context1.target_dataset_name == "dataset1"
        assert context2.target_dataset_name == "dataset2"
        
        # Verify persistence within session
        context1_again = await tool_server.get_or_create_context("session1", "dataset1")
        assert context1 is context1_again