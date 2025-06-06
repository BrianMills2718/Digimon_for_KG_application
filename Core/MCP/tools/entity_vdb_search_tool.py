"""
MCP Wrapper for Entity VDB Search Tool

This module wraps the Entity.VDBSearch tool for MCP protocol communication.
"""

import time
import logging
from typing import Dict, Any, Optional
from Core.AgentSchema.tool_contracts import (
    EntityVDBSearchInputs,
    EntityVDBSearchOutputs,
    VDBSearchResultItem
)
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentTools.entity_tools import entity_vdb_search_tool
from Core.MCP.shared_context import get_shared_context

logger = logging.getLogger(__name__)


class EntityVDBSearchMCPTool:
    """MCP wrapper for Entity VDB Search tool"""
    
    TOOL_NAME = "Entity.VDBSearch"
    TOOL_DESCRIPTION = "Search for entities in a vector database using text queries"
    
    def __init__(self):
        self.shared_context = get_shared_context()
        
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition"""
        return {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "vdb_reference_id": {
                        "type": "string",
                        "description": "Reference ID of the VDB to search"
                    },
                    "query_text": {
                        "type": "string",
                        "description": "Text query to search for similar entities"
                    },
                    "query_embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional: Pre-computed query embedding"
                    },
                    "top_k_results": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of results to return"
                    }
                },
                "required": ["vdb_reference_id"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "similar_entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_id": {"type": "string"},
                                "entity_name": {"type": "string"},
                                "score": {"type": "number"}
                            }
                        }
                    }
                }
            }
        }
        
    async def execute(
        self,
        params: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the tool via MCP protocol"""
        start_time = time.time()
        
        try:
            # Validate and convert parameters
            inputs = EntityVDBSearchInputs(
                vdb_reference_id=params["vdb_reference_id"],
                query_text=params.get("query_text"),
                query_embedding=params.get("query_embedding"),
                top_k_results=params.get("top_k_results", 10)
            )
            
            # Get GraphRAG context from shared context or session
            graphrag_context = None
            if session_id:
                graphrag_context = self.shared_context.get(
                    "graphrag_context",
                    session_id=session_id
                )
            
            if not graphrag_context:
                # Try global context
                graphrag_context = self.shared_context.get("graphrag_context")
                
            if not graphrag_context:
                # Create new context if needed
                graphrag_context = GraphRAGContext()
                self.shared_context.set("graphrag_context", graphrag_context)
            
            # Execute the actual tool
            result = await entity_vdb_search_tool(inputs, graphrag_context)
            
            # Convert result to MCP format
            output = {
                "similar_entities": [
                    {
                        "node_id": item.node_id,
                        "entity_name": item.entity_name,
                        "score": item.score
                    }
                    for item in result.similar_entities
                ]
            }
            
            # Log performance
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Entity.VDBSearch executed in {elapsed_ms:.1f}ms, "
                f"found {len(result.similar_entities)} entities"
            )
            
            return {
                "status": "success",
                "result": output,
                "metadata": {
                    "execution_time_ms": elapsed_ms,
                    "result_count": len(result.similar_entities)
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing Entity.VDBSearch: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
            
    async def execute_direct(
        self,
        inputs: EntityVDBSearchInputs,
        graphrag_context: GraphRAGContext
    ) -> EntityVDBSearchOutputs:
        """Direct execution for backward compatibility"""
        return await entity_vdb_search_tool(inputs, graphrag_context)


# Singleton instance
entity_vdb_search_mcp_tool = EntityVDBSearchMCPTool()