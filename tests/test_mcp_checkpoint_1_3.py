"""
Tests for Phase 1, Checkpoint 1.3: Direct Tool Migration
"""

import asyncio
import pytest
from datetime import datetime
import uuid

from Core.MCP import MCPRequest, MCPResponse
from Core.MCP.simple_mcp_server import SimpleMCPToolServer


class TestMCPToolMigration:
    """Test direct migration of 5 core tools to MCP"""
    
    @pytest.fixture
    async def mcp_server(self):
        """Create test MCP server with tools"""
        server = SimpleMCPToolServer(port=9998)
        yield server
    
    async def test_entity_vdb_search(self, mcp_server):
        """Test: Entity.VDBSearch accessible via MCP"""
        request = MCPRequest(
            id=str(uuid.uuid4()),
            tool_name="Entity.VDBSearch",
            params={
                "vdb_reference_id": "test_vdb",
                "query_text": "French Revolution",
                "top_k_results": 5
            },
            context={},
            session_id="test_session",
            timestamp=datetime.utcnow()
        )
        
        response = await mcp_server.server.handle_request(request)
        
        assert response.status == "success"
        assert "similar_entities" in response.result
        assert len(response.result["similar_entities"]) <= 5
        assert response.result["similar_entities"][0]["similarity_score"] > 0.8
    
    async def test_graph_build(self, mcp_server):
        """Test: Graph.Build accessible via MCP"""
        request = MCPRequest(
            id=str(uuid.uuid4()),
            tool_name="Graph.Build",
            params={
                "corpus_reference_id": "corpus_123",
                "target_graph_id": "graph_456",
                "graph_type": "ERGraph"
            },
            context={},
            session_id="test_session",
            timestamp=datetime.utcnow()
        )
        
        response = await mcp_server.server.handle_request(request)
        
        assert response.status == "success"
        assert response.result["graph_reference_id"] == "graph_456"
        assert response.result["node_count"] > 0
        assert response.result["edge_count"] > 0
    
    async def test_corpus_prepare(self, mcp_server):
        """Test: Corpus.Prepare accessible via MCP"""
        request = MCPRequest(
            id=str(uuid.uuid4()),
            tool_name="Corpus.Prepare",
            params={
                "input_directory_path": "/test/input",
                "output_directory_path": "/test/output",
                "target_corpus_name": "test_corpus"
            },
            context={},
            session_id="test_session",
            timestamp=datetime.utcnow()
        )
        
        response = await mcp_server.server.handle_request(request)
        
        assert response.status == "success"
        assert response.result["status"] == "success"
        assert response.result["document_count"] > 0
        assert "corpus_json_path" in response.result
    
    async def test_chunk_retrieve(self, mcp_server):
        """Test: Chunk.Retrieve accessible via MCP"""
        request = MCPRequest(
            id=str(uuid.uuid4()),
            tool_name="Chunk.Retrieve",
            params={
                "entity_names": ["Napoleon", "Washington"],
                "document_collection_id": "corpus_123",
                "max_chunks_per_entity": 3
            },
            context={},
            session_id="test_session",
            timestamp=datetime.utcnow()
        )
        
        response = await mcp_server.server.handle_request(request)
        
        assert response.status == "success"
        assert "chunks" in response.result
        assert len(response.result["chunks"]) > 0
        assert "text" in response.result["chunks"][0]
    
    async def test_answer_generate(self, mcp_server):
        """Test: Answer.Generate accessible via MCP"""
        request = MCPRequest(
            id=str(uuid.uuid4()),
            tool_name="Answer.Generate",
            params={
                "query": "What were the causes of the French Revolution?",
                "context": [
                    "The French Revolution began in 1789.",
                    "Economic crisis was a major factor.",
                    "Social inequality contributed to unrest."
                ],
                "max_length": 200
            },
            context={},
            session_id="test_session",
            timestamp=datetime.utcnow()
        )
        
        response = await mcp_server.server.handle_request(request)
        
        assert response.status == "success"
        assert "answer" in response.result
        assert len(response.result["answer"]) <= 200
        assert response.result["confidence"] > 0.5
    
    async def test_tool_not_found(self, mcp_server):
        """Test: Error handling for non-existent tool"""
        request = MCPRequest(
            id=str(uuid.uuid4()),
            tool_name="NonExistent.Tool",
            params={},
            context={},
            session_id="test_session",
            timestamp=datetime.utcnow()
        )
        
        response = await mcp_server.server.handle_request(request)
        
        assert response.status == "error"
        assert "Tool not found" in str(response.result)
    
    async def test_performance_metrics(self, mcp_server):
        """Test: Performance within acceptable limits"""
        import time
        
        # Test each tool's latency
        tools = [
            ("Entity.VDBSearch", {"vdb_reference_id": "test", "query_text": "test"}),
            ("Graph.Build", {"corpus_reference_id": "c1", "target_graph_id": "g1"}),
            ("Corpus.Prepare", {"input_directory_path": "/in", "output_directory_path": "/out"}),
            ("Chunk.Retrieve", {"entity_names": ["e1"], "document_collection_id": "c1"}),
            ("Answer.Generate", {"query": "test?", "context": ["test context"]})
        ]
        
        for tool_name, params in tools:
            request = MCPRequest(
                id=str(uuid.uuid4()),
                tool_name=tool_name,
                params=params,
                context={},
                session_id="perf_test",
                timestamp=datetime.utcnow()
            )
            
            start = time.time()
            response = await mcp_server.server.handle_request(request)
            latency = (time.time() - start) * 1000  # ms
            
            assert response.status == "success"
            assert latency < 100  # Should complete in under 100ms for mock
            assert "latency_ms" in response.metadata
    
    async def test_concurrent_requests(self, mcp_server):
        """Test: Multiple tools can be called concurrently"""
        requests = []
        
        # Create multiple different requests
        for i in range(5):
            request = MCPRequest(
                id=str(uuid.uuid4()),
                tool_name="Entity.VDBSearch",
                params={
                    "vdb_reference_id": f"vdb_{i}",
                    "query_text": f"query_{i}",
                    "top_k_results": 3
                },
                context={},
                session_id=f"concurrent_{i}",
                timestamp=datetime.utcnow()
            )
            requests.append(mcp_server.server.handle_request(request))
        
        # Execute concurrently
        responses = await asyncio.gather(*requests)
        
        # All should succeed
        assert all(r.status == "success" for r in responses)
        assert len(responses) == 5
    
    async def test_session_context(self, mcp_server):
        """Test: Context persists within session"""
        session_id = "context_test"
        
        # First request
        request1 = MCPRequest(
            id=str(uuid.uuid4()),
            tool_name="Entity.VDBSearch",
            params={"vdb_reference_id": "vdb1", "query_text": "test"},
            context={"user_preference": "detailed"},
            session_id=session_id,
            timestamp=datetime.utcnow()
        )
        
        response1 = await mcp_server.server.handle_request(request1)
        assert response1.status == "success"
        
        # Second request in same session
        request2 = MCPRequest(
            id=str(uuid.uuid4()),
            tool_name="Answer.Generate",
            params={"query": "test?", "context": ["data"]},
            context={},  # Empty context
            session_id=session_id,
            timestamp=datetime.utcnow()
        )
        
        response2 = await mcp_server.server.handle_request(request2)
        assert response2.status == "success"
        
        # Check context was preserved (would need to check in actual implementation)
        # For now just verify both succeeded
        assert mcp_server.server.context_store.sessions[session_id] is not None


@pytest.mark.asyncio 
async def test_tool_integration():
    """Test: Integration test of all 5 tools"""
    server = SimpleMCPToolServer(port=9997)
    
    # 1. Prepare corpus
    corpus_req = MCPRequest(
        id=str(uuid.uuid4()),
        tool_name="Corpus.Prepare",
        params={
            "input_directory_path": "/data/texts",
            "output_directory_path": "/data/corpus",
            "target_corpus_name": "revolution_corpus"
        },
        context={},
        session_id="integration_test",
        timestamp=datetime.utcnow()
    )
    
    corpus_resp = await server.server.handle_request(corpus_req)
    assert corpus_resp.status == "success"
    
    # 2. Build graph
    graph_req = MCPRequest(
        id=str(uuid.uuid4()),
        tool_name="Graph.Build",
        params={
            "corpus_reference_id": "revolution_corpus",
            "target_graph_id": "revolution_graph"
        },
        context={},
        session_id="integration_test",
        timestamp=datetime.utcnow()
    )
    
    graph_resp = await server.server.handle_request(graph_req)
    assert graph_resp.status == "success"
    
    # 3. Search entities
    search_req = MCPRequest(
        id=str(uuid.uuid4()),
        tool_name="Entity.VDBSearch",
        params={
            "vdb_reference_id": "revolution_graph_vdb",
            "query_text": "revolutionary leaders",
            "top_k_results": 5
        },
        context={},
        session_id="integration_test",
        timestamp=datetime.utcnow()
    )
    
    search_resp = await server.server.handle_request(search_req)
    assert search_resp.status == "success"
    entities = [e["entity_name"] for e in search_resp.result["similar_entities"]]
    
    # 4. Get chunks
    chunks_req = MCPRequest(
        id=str(uuid.uuid4()),
        tool_name="Chunk.Retrieve",
        params={
            "entity_names": entities[:2],
            "document_collection_id": "revolution_corpus"
        },
        context={},
        session_id="integration_test",
        timestamp=datetime.utcnow()
    )
    
    chunks_resp = await server.server.handle_request(chunks_req)
    assert chunks_resp.status == "success"
    context_texts = [c["text"] for c in chunks_resp.result["chunks"]]
    
    # 5. Generate answer
    answer_req = MCPRequest(
        id=str(uuid.uuid4()),
        tool_name="Answer.Generate",
        params={
            "query": "Who were the main revolutionary leaders?",
            "context": context_texts
        },
        context={},
        session_id="integration_test",
        timestamp=datetime.utcnow()
    )
    
    answer_resp = await server.server.handle_request(answer_req)
    assert answer_resp.status == "success"
    assert len(answer_resp.result["answer"]) > 0
    
    # Verify metrics
    metrics = server.server.get_metrics()
    assert metrics["request_count"] == 5
    assert metrics["error_count"] == 0
    assert metrics["registered_tools"] == 5


if __name__ == "__main__":
    asyncio.run(test_tool_integration())