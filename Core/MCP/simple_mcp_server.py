"""
Simple MCP Server wrapper for DIGIMON tools - focuses on MCP protocol
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
import uuid

from Core.MCP.mcp_server import DigimonMCPServer, MCPTool, MCPRequest, MCPResponse

logger = logging.getLogger(__name__)


class SimpleMCPToolServer:
    """
    Simplified MCP Server that demonstrates tool wrapping
    """
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.server = DigimonMCPServer(
            server_name="digimon-tools",
            capabilities=[
                "Entity.VDBSearch",
                "Graph.Build",
                "Corpus.Prepare",
                "Chunk.Retrieve",
                "Answer.Generate"
            ],
            port=port
        )
        self._register_tools()
    
    def _register_tools(self):
        """Register all DIGIMON tools with the MCP server"""
        
        # Entity.VDBSearch
        self.server.register_tool(MCPTool(
            name="Entity.VDBSearch",
            handler=self._entity_vdb_search,
            schema={
                "description": "Search for entities in a vector database",
                "parameters": {
                    "vdb_reference_id": {"type": "string", "description": "VDB identifier"},
                    "query_text": {"type": "string", "description": "Search query"},
                    "top_k_results": {"type": "integer", "default": 5}
                }
            }
        ))
        
        # Graph.Build
        self.server.register_tool(MCPTool(
            name="Graph.Build",
            handler=self._graph_build,
            schema={
                "description": "Build a knowledge graph from corpus",
                "parameters": {
                    "corpus_reference_id": {"type": "string"},
                    "target_graph_id": {"type": "string"},
                    "graph_type": {"type": "string", "default": "ERGraph"}
                }
            }
        ))
        
        # Corpus.Prepare
        self.server.register_tool(MCPTool(
            name="Corpus.Prepare",
            handler=self._corpus_prepare,
            schema={
                "description": "Prepare corpus from text files",
                "parameters": {
                    "input_directory_path": {"type": "string"},
                    "output_directory_path": {"type": "string"},
                    "target_corpus_name": {"type": "string", "optional": True}
                }
            }
        ))
        
        # Chunk.Retrieve
        self.server.register_tool(MCPTool(
            name="Chunk.Retrieve",
            handler=self._chunk_retrieve,
            schema={
                "description": "Retrieve text chunks for entities",
                "parameters": {
                    "entity_names": {"type": "array", "items": {"type": "string"}},
                    "document_collection_id": {"type": "string"},
                    "max_chunks_per_entity": {"type": "integer", "default": 5}
                }
            }
        ))
        
        # Answer.Generate
        self.server.register_tool(MCPTool(
            name="Answer.Generate",
            handler=self._answer_generate,
            schema={
                "description": "Generate answer from query and context",
                "parameters": {
                    "query": {"type": "string"},
                    "context": {"type": "array", "items": {"type": "string"}},
                    "max_length": {"type": "integer", "default": 500}
                }
            }
        ))
    
    async def _entity_vdb_search(self, vdb_reference_id=None, query_text=None, 
                                   top_k_results=5, context=None, **kwargs):
        """Mock implementation of Entity.VDBSearch"""
        vdb_id = vdb_reference_id
        query = query_text
        top_k = top_k_results
        
        # Mock search results
        mock_results = []
        for i in range(min(top_k, 3)):
            mock_results.append({
                "entity_name": f"Entity_{i+1}",
                "entity_id": f"e{i+1}",
                "similarity_score": 0.95 - (i * 0.1),
                "entity_metadata": {"type": "Person", "source": "test"}
            })
        
        return {
            "similar_entities": mock_results,
            "search_metadata": {
                "vdb_id": vdb_id,
                "query": query,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    async def _graph_build(self, corpus_reference_id=None, target_graph_id=None,
                           graph_type='ERGraph', context=None, **kwargs):
        """Mock implementation of Graph.Build"""
        corpus_id = corpus_reference_id
        graph_id = target_graph_id
        
        # Mock graph building
        return {
            "graph_reference_id": graph_id,
            "node_count": 150,
            "edge_count": 300,
            "graph_artifact_path": f"/mock/graphs/{graph_id}",
            "status": "Graph built successfully"
        }
    
    async def _corpus_prepare(self, input_directory_path=None, output_directory_path=None,
                              target_corpus_name='corpus', context=None, **kwargs):
        """Mock implementation of Corpus.Prepare"""
        input_dir = input_directory_path
        output_dir = output_directory_path
        corpus_name = target_corpus_name
        
        # Mock corpus preparation
        return {
            "status": "success",
            "message": f"Successfully processed 10 documents",
            "document_count": 10,
            "corpus_json_path": f"{output_dir}/Corpus.json",
            "corpus_name": corpus_name
        }
    
    async def _chunk_retrieve(self, entity_names=None, document_collection_id=None,
                              max_chunks_per_entity=5, context=None, **kwargs):
        """Mock implementation of Chunk.Retrieve"""
        entities = entity_names or []
        collection_id = document_collection_id
        max_chunks = max_chunks_per_entity
        
        # Mock chunk retrieval
        chunks = []
        for entity in entities[:3]:  # Limit to 3 entities
            for i in range(min(2, max_chunks)):
                chunks.append({
                    "entity": entity,
                    "chunk_id": f"chunk_{entity}_{i}",
                    "text": f"This is a text chunk about {entity}. It contains relevant information.",
                    "metadata": {
                        "doc_id": f"doc_{i}",
                        "position": i
                    }
                })
        
        return {
            "chunks": chunks,
            "total_chunks": len(chunks),
            "collection_id": collection_id
        }
    
    async def _answer_generate(self, query=None, context=None,
                               max_length=500, **kwargs):
        """Mock implementation of Answer.Generate"""
        # Note: There's a collision between MCP context and the 'context' parameter for the query
        # If context is a list, it's the query context. If it's a dict, it's MCP context
        if isinstance(context, list):
            query_context = context
            mcp_context = kwargs.get('mcp_context', {})
        elif isinstance(context, dict) and 'mcp_context' not in kwargs:
            # This is MCP context, check if there's query context in kwargs
            mcp_context = context
            query_context = []
        else:
            query_context = []
            mcp_context = kwargs.get('mcp_context', {})
        
        # Mock answer generation
        context_text = " ".join(query_context[:3]) if query_context else "No context provided."
        
        answer = f"Based on the query '{query}' and the provided context, here is a generated answer. {context_text[:100]}..."
        
        return {
            "answer": answer[:max_length],
            "confidence": 0.85,
            "sources_used": len(query_context),
            "metadata": {
                "model": "mock-llm",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    async def start(self):
        """Start the MCP server"""
        logger.info(f"Starting SimpleMCPToolServer on port {self.port}")
        await self.server.start()
    
    def run(self):
        """Run the server (blocking)"""
        asyncio.run(self.start())


async def test_simple_server():
    """Test the simple MCP server"""
    server = SimpleMCPToolServer(port=9999)
    
    # Create test request
    request = MCPRequest(
        id=str(uuid.uuid4()),
        tool_name="Entity.VDBSearch",
        params={
            "vdb_reference_id": "test_vdb",
            "query_text": "revolution",
            "top_k_results": 3
        },
        context={},
        session_id="test_session",
        timestamp=datetime.utcnow()
    )
    
    # Test direct handler
    response = await server.server.handle_request(request)
    print(f"Response: {response.to_json()}")
    
    # Test all tools
    tools = ["Entity.VDBSearch", "Graph.Build", "Corpus.Prepare", "Chunk.Retrieve", "Answer.Generate"]
    for tool_name in tools:
        request.tool_name = tool_name
        request.id = str(uuid.uuid4())
        
        if tool_name == "Answer.Generate":
            request.params = {"query": "What is revolution?", "context": ["Revolution is change"]}
        elif tool_name == "Chunk.Retrieve":
            request.params = {"entity_names": ["Entity1", "Entity2"], "document_collection_id": "test"}
        
        response = await server.server.handle_request(request)
        print(f"\n{tool_name}: {response.status}")
        if response.status == "success":
            print(f"Result keys: {list(response.result.keys())}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_simple_server())
    
    # Or run server
    # server = SimpleMCPToolServer()
    # server.run()