"""
DIGIMON MCP Server - Wraps core DIGIMON tools for MCP access
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from Core.MCP.mcp_server import DigimonMCPServer, MCPTool
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    EntityVDBSearchInputs, EntityVDBSearchOutputs,
    ChunkFromRelationshipsInputs, ChunkFromRelationshipsOutputs,
    ChunkGetTextForEntitiesInput, ChunkGetTextForEntitiesOutput
)
from Core.AgentSchema.graph_construction_tool_contracts import (
    BuildERGraphInputs, BuildERGraphOutputs
)
from Core.AgentSchema.corpus_tool_contracts import (
    PrepareCorpusInputs, PrepareCorpusOutputs
)

# Import tool implementations
from Core.AgentTools.entity_tools import entity_vdb_search_tool
from Core.AgentTools.graph_construction_tools import build_er_graph
from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
from Core.AgentTools.chunk_tools import chunk_get_text_for_entities_tool

# Import core dependencies
from Option.Config2 import Config
from Core.Provider.LiteLLMProvider import LiteLLMProvider
# from Core.Index.EmbeddingFactory import EmbeddingFactory  # Removed for simplicity
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger

# Configure logging
logging.basicConfig(level=logging.INFO)


class DigimonToolServer:
    """
    MCP Server that wraps core DIGIMON tools
    """
    
    def __init__(self, config_path: str = "Option/Config2.yaml"):
        self.config_path = config_path
        self.config = None
        self.llm_provider = None
        self.embedding_provider = None
        self.chunk_factory = None
        self.context_store = {}  # Store GraphRAGContext instances by session
        self.server = None
        
    async def initialize(self):
        """Initialize core components"""
        logger.info("Initializing DIGIMON MCP Server components...")
        
        # Load configuration
        self.config = Config.load_yaml(self.config_path)
        
        # Initialize LLM provider
        self.llm_provider = LiteLLMProvider.create(config=self.config.llm)
        
        # Initialize embedding provider
        # For now, use a mock embedding provider to simplify testing
        from unittest.mock import Mock
        self.embedding_provider = Mock()
        self.embedding_provider.get_text_embedding_batch = Mock(return_value=[[0.1] * 768] * 10)
        
        # Initialize chunk factory
        self.chunk_factory = ChunkFactory(config=self.config.chunk)
        
        logger.info("DIGIMON components initialized successfully")
        
    async def get_or_create_context(self, session_id: str, dataset_name: str) -> GraphRAGContext:
        """Get or create a GraphRAGContext for the session"""
        if session_id not in self.context_store:
            self.context_store[session_id] = GraphRAGContext(
                target_dataset_name=dataset_name,
                main_config=self.config,
                llm_provider=self.llm_provider,
                embedding_provider=self.embedding_provider,
                chunk_storage_manager=self.chunk_factory
            )
        return self.context_store[session_id]
    
    # Tool wrapper: Entity.VDBSearch
    async def entity_vdb_search_wrapper(self, **params):
        """Wrapper for Entity.VDBSearch tool"""
        try:
            # Extract session info
            session_id = params.pop('session_id', 'default')
            dataset_name = params.pop('dataset_name', 'default')
            
            # Create input model
            tool_input = EntityVDBSearchInputs(**params)
            
            # Get context
            context = await self.get_or_create_context(session_id, dataset_name)
            
            # Execute tool
            result = await entity_vdb_search_tool(tool_input, context)
            
            # Convert result to dict
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Error in entity_vdb_search_wrapper: {e}", exc_info=True)
            return {
                "similar_entities": [],
                "error": str(e)
            }
    
    # Tool wrapper: Graph.Build
    async def graph_build_wrapper(self, **params):
        """Wrapper for Graph.Build tool (using ERGraph as default)"""
        try:
            # Extract session info
            session_id = params.pop('session_id', 'default')
            dataset_name = params.get('target_dataset_name', 'default')
            
            # Create input model
            tool_input = BuildERGraphInputs(**params)
            
            # Get context (not used directly by graph build, but for consistency)
            context = await self.get_or_create_context(session_id, dataset_name)
            
            # Execute tool
            result = await build_er_graph(
                tool_input,
                self.config,
                self.llm_provider,
                self.embedding_provider,
                self.chunk_factory
            )
            
            # Store graph instance in context if successful
            if result.status == "success" and hasattr(result, 'graph_instance'):
                context.add_graph_instance(result.graph_id, result.graph_instance)
            
            # Convert result to dict (exclude graph_instance)
            result_dict = result.model_dump(exclude={'graph_instance'})
            return result_dict
            
        except Exception as e:
            logger.error(f"Error in graph_build_wrapper: {e}", exc_info=True)
            return {
                "graph_id": "",
                "status": "failure",
                "message": str(e),
                "artifact_path": None
            }
    
    # Tool wrapper: Corpus.Prepare
    async def corpus_prepare_wrapper(self, **params):
        """Wrapper for Corpus.Prepare tool"""
        try:
            # Extract session info
            session_id = params.pop('session_id', 'default')
            
            # Create input model
            tool_input = PrepareCorpusInputs(**params)
            
            # Execute tool
            result = await prepare_corpus_from_directory(tool_input, self.config)
            
            # Convert result to dict
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Error in corpus_prepare_wrapper: {e}", exc_info=True)
            return {
                "status": "failure",
                "message": str(e),
                "document_count": 0,
                "corpus_json_path": None
            }
    
    # Tool wrapper: Chunk.Retrieve
    async def chunk_retrieve_wrapper(self, **params):
        """Wrapper for Chunk.Retrieve tool (using GetTextForEntities)"""
        try:
            # Extract session info
            session_id = params.pop('session_id', 'default')
            dataset_name = params.pop('dataset_name', 'default')
            
            # Create input model
            tool_input = ChunkGetTextForEntitiesInput(**params)
            
            # Get context
            context = await self.get_or_create_context(session_id, dataset_name)
            
            # Execute tool
            result = await chunk_get_text_for_entities_tool(tool_input, context)
            
            # Return result (already a dict)
            return result
            
        except Exception as e:
            logger.error(f"Error in chunk_retrieve_wrapper: {e}", exc_info=True)
            return {
                "retrieved_chunks": [],
                "status_message": f"Error: {str(e)}"
            }
    
    # Tool wrapper: Answer.Generate
    async def answer_generate_wrapper(self, **params):
        """Wrapper for Answer.Generate tool"""
        try:
            # Extract parameters
            query = params.get('query', '')
            context_data = params.get('context', '')
            response_type = params.get('response_type', 'default')
            use_tree_search = params.get('use_tree_search', False)
            
            # Create a basic query instance for answer generation
            query_config = self.config.query.model_copy()
            query_config.response_type = response_type
            query_config.tree_search = use_tree_search
            
            # Create query instance
            basic_query = BaseQuery(config=query_config, retriever_context=None)
            basic_query.llm = self.llm_provider
            
            # Generate answer
            answer = await basic_query.generation_qa(query, context_data)
            
            return {
                "answer": answer,
                "query": query,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in answer_generate_wrapper: {e}", exc_info=True)
            return {
                "answer": "",
                "query": query if 'query' in locals() else "",
                "status": "error",
                "error": str(e)
            }
    
    async def start_server(self, port: int = 8765):
        """Start the MCP server"""
        # Initialize components
        await self.initialize()
        
        # Create MCP server
        self.server = DigimonMCPServer(
            server_name="DIGIMON-Core",
            capabilities=["entity_search", "graph_build", "corpus_prepare", "chunk_retrieve", "answer_generate"],
            port=port
        )
        
        # Register tools
        self.server.register_tool(MCPTool(
            name="Entity.VDBSearch",
            handler=self.entity_vdb_search_wrapper,
            schema={
                "description": "Search for entities in a vector database using text queries",
                "parameters": {
                    "vdb_reference_id": {"type": "string", "description": "VDB identifier"},
                    "query_text": {"type": "string", "description": "Search query"},
                    "top_k_results": {"type": "integer", "default": 5},
                    "session_id": {"type": "string", "default": "default"},
                    "dataset_name": {"type": "string", "default": "default"}
                }
            }
        ))
        
        self.server.register_tool(MCPTool(
            name="Graph.Build",
            handler=self.graph_build_wrapper,
            schema={
                "description": "Build an entity-relationship graph from a corpus",
                "parameters": {
                    "target_dataset_name": {"type": "string", "description": "Dataset name"},
                    "force_rebuild": {"type": "boolean", "default": False},
                    "session_id": {"type": "string", "default": "default"}
                }
            }
        ))
        
        self.server.register_tool(MCPTool(
            name="Corpus.Prepare",
            handler=self.corpus_prepare_wrapper,
            schema={
                "description": "Prepare a corpus from a directory of text files",
                "parameters": {
                    "input_directory_path": {"type": "string", "description": "Input directory path"},
                    "output_directory_path": {"type": "string", "description": "Output directory path"},
                    "target_corpus_name": {"type": "string", "description": "Corpus name", "optional": True},
                    "session_id": {"type": "string", "default": "default"}
                }
            }
        ))
        
        self.server.register_tool(MCPTool(
            name="Chunk.Retrieve",
            handler=self.chunk_retrieve_wrapper,
            schema={
                "description": "Retrieve text chunks associated with entities",
                "parameters": {
                    "entity_ids": {"type": "array", "items": {"type": "string"}, "description": "Entity IDs"},
                    "graph_reference_id": {"type": "string", "description": "Graph ID"},
                    "max_chunks_per_entity": {"type": "integer", "optional": True},
                    "session_id": {"type": "string", "default": "default"},
                    "dataset_name": {"type": "string", "default": "default"}
                }
            }
        ))
        
        self.server.register_tool(MCPTool(
            name="Answer.Generate",
            handler=self.answer_generate_wrapper,
            schema={
                "description": "Generate an answer based on query and context",
                "parameters": {
                    "query": {"type": "string", "description": "User question"},
                    "context": {"type": "string", "description": "Retrieved context"},
                    "response_type": {"type": "string", "default": "default"},
                    "use_tree_search": {"type": "boolean", "default": False}
                }
            }
        ))
        
        logger.info(f"Starting DIGIMON MCP Server on port {port}...")
        
        # Start server
        await self.server.start()


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DIGIMON MCP Server")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--config", type=str, default="Option/Config2.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    server = DigimonToolServer(config_path=args.config)
    
    try:
        await server.start_server(port=args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())