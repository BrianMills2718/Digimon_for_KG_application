"""
MCP wrapper for Entity.PPR tool

Provides MCP-compatible interface for entity Personalized PageRank.
"""

import asyncio
from typing import Dict, Any, Optional
from Core.MCP.shared_context import get_shared_context
from Core.AgentSchema.tool_contracts import EntityPPRInputs
from Core.AgentTools.entity_tools import entity_ppr_tool
from Core.Common.Logger import logger
import time


class EntityPPRMCPTool:
    """MCP wrapper for Entity.PPR tool"""
    
    def __init__(self):
        self.tool_id = "Entity.PPR"
        self.name = "Entity.PPR"
        self.description = "Rank entities using Personalized PageRank algorithm"
        
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
                        "description": "ID of the graph to run PPR on"
                    },
                    "seed_entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Seed entity IDs for personalization"
                    },
                    "damping_factor": {
                        "type": "number",
                        "description": "PageRank damping factor",
                        "default": 0.85,
                        "minimum": 0,
                        "maximum": 1
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum iterations",
                        "default": 100
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top entities to return",
                        "default": 10
                    }
                },
                "required": ["graph_reference_id", "seed_entity_ids"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success", "failure"]},
                    "message": {"type": "string"},
                    "ranked_entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity_id": {"type": "string"},
                                "entity_name": {"type": "string"},
                                "score": {"type": "number"}
                            }
                        }
                    }
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
            inputs = EntityPPRInputs(
                graph_reference_id=params["graph_reference_id"],
                seed_entity_ids=params["seed_entity_ids"],
                damping_factor=params.get("damping_factor", 0.85),
                max_iterations=params.get("max_iterations", 100),
                top_k=params.get("top_k", 10)
            )
            
            # Execute tool
            logger.info(f"MCP: Executing entity_ppr_tool")
            
            result = await entity_ppr_tool(inputs, graphrag_context)
            
            # Convert result to dict
            result_dict = result.model_dump()
            
            execution_time = (time.time() - start_time) * 1000
            logger.info(f"MCP: entity_ppr_tool completed in {execution_time:.1f}ms")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"MCP: entity_ppr_tool failed: {e}")
            
            return {
                "status": "failure",
                "message": str(e),
                "ranked_entities": []
            }


# Create singleton instance
entity_ppr_mcp_tool = EntityPPRMCPTool()