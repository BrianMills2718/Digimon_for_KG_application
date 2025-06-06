"""
MCP wrapper for Entity.VDB.Build tool

Provides MCP-compatible interface for building entity vector databases.
"""

import asyncio
from typing import Dict, Any, Optional
from Core.MCP.shared_context import get_shared_context
from Core.AgentSchema.tool_contracts import EntityVDBBuildInputs
from Core.AgentTools.entity_vdb_tools import entity_vdb_build_tool
from Core.Common.Logger import logger
import time


class EntityVDBBuildMCPTool:
    """MCP wrapper for Entity.VDB.Build tool"""
    
    def __init__(self):
        self.tool_id = "Entity.VDB.Build"
        self.name = "Entity.VDB.Build"
        self.description = "Build vector database index for entities"
        
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return MCP tool definition"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_reference_id": {
                        "type": "string",
                        "description": "ID of the graph to build VDB from"
                    },
                    "vdb_reference_id": {
                        "type": "string",
                        "description": "ID to store the built VDB"
                    },
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific entity IDs to include (optional)"
                    },
                    "force_rebuild": {
                        "type": "boolean",
                        "description": "Force rebuild even if VDB exists",
                        "default": False
                    }
                },
                "required": ["graph_reference_id", "vdb_reference_id"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success", "failure"]},
                    "message": {"type": "string"},
                    "vdb_reference_id": {"type": "string"},
                    "entities_indexed": {"type": "integer"}
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute tool via MCP"""
        start_time = time.time()
        
        try:
            # Get shared context
            shared_context = get_shared_context()
            
            # Get GraphRAG context from shared context
            graphrag_context = shared_context.get("graphrag_context", session_id)
            if not graphrag_context:
                raise ValueError("GraphRAG context not found in shared context")
            
            # Create input object
            inputs = EntityVDBBuildInputs(
                graph_reference_id=params["graph_reference_id"],
                vdb_reference_id=params["vdb_reference_id"],
                entity_ids=params.get("entity_ids"),
                force_rebuild=params.get("force_rebuild", False)
            )
            
            # Execute tool
            logger.info(f"MCP: Executing entity_vdb_build_tool")
            
            result = await entity_vdb_build_tool(inputs, graphrag_context)
            
            # Convert result to dict
            result_dict = result.model_dump()
            
            execution_time = (time.time() - start_time) * 1000
            logger.info(f"MCP: entity_vdb_build_tool completed in {execution_time:.1f}ms")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"MCP: entity_vdb_build_tool failed: {e}")
            
            return {
                "status": "failure",
                "message": str(e),
                "vdb_reference_id": params.get("vdb_reference_id", ""),
                "entities_indexed": 0
            }


# Create singleton instance
entity_vdb_build_mcp_tool = EntityVDBBuildMCPTool()