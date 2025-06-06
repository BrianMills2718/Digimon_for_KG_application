"""
MCP wrapper for BuildERGraph tool

Provides MCP-compatible interface for building entity-relationship graphs.
"""

import asyncio
from typing import Dict, Any, Optional
from Core.MCP.shared_context import get_shared_context
from Core.AgentSchema.graph_construction_tool_contracts import (
    BuildERGraphInputs, BuildERGraphOutputs, ERGraphConfigOverrides
)
from Core.AgentTools.graph_construction_tools import build_er_graph
from Core.Common.Logger import logger
import time


class BuildERGraphMCPTool:
    """MCP wrapper for BuildERGraph tool"""
    
    def __init__(self):
        self.tool_id = "graph.BuildERGraph"
        self.name = "graph.BuildERGraph"
        self.description = "Build an entity-relationship graph from a dataset"
        
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return MCP tool definition"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset to build graph from"
                    },
                    "force_rebuild": {
                        "type": "boolean",
                        "description": "Force rebuild even if graph exists",
                        "default": False
                    },
                    "config_overrides": {
                        "type": "object",
                        "description": "Optional configuration overrides",
                        "properties": {
                            "max_knowledge_entities": {"type": "integer"},
                            "entity_extract_max_gleaning": {"type": "integer"},
                            "entity_summary_config": {
                                "type": "object",
                                "properties": {
                                    "max_tokens": {"type": "integer"}
                                }
                            }
                        }
                    }
                },
                "required": ["target_dataset_name"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["success", "failure"]},
                    "message": {"type": "string"},
                    "node_count": {"type": ["integer", "null"]},
                    "edge_count": {"type": ["integer", "null"]},
                    "layer_count": {"type": ["integer", "null"]},
                    "artifact_path": {"type": ["string", "null"]}
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
            config_overrides = None
            if params.get("config_overrides"):
                config_overrides = ERGraphConfigOverrides(**params["config_overrides"])
                
            inputs = BuildERGraphInputs(
                target_dataset_name=params["target_dataset_name"],
                force_rebuild=params.get("force_rebuild", False),
                config_overrides=config_overrides
            )
            
            # Report progress
            progress_callback = params.get("progress_callback")
            if progress_callback:
                await progress_callback({
                    "status": "started",
                    "message": f"Building ER graph for {inputs.target_dataset_name}",
                    "progress": 0
                })
            
            # Execute tool
            logger.info(f"MCP: Executing build_er_graph for {inputs.target_dataset_name}")
            
            result = await build_er_graph(
                tool_input=inputs,
                main_config=graphrag_context.main_config,
                llm_instance=graphrag_context.llm_provider,
                encoder_instance=graphrag_context.embedding_provider,
                chunk_factory=graphrag_context.chunk_storage_manager
            )
            
            # Add graph instance to context if successful
            if result.status == "success" and hasattr(result, 'graph_instance') and result.graph_instance:
                graphrag_context.add_graph_instance(result.graph_id, result.graph_instance)
                logger.info(f"MCP: Added graph {result.graph_id} to context")
            
            # Report completion
            if progress_callback:
                await progress_callback({
                    "status": "completed",
                    "message": result.message,
                    "progress": 100,
                    "node_count": result.node_count,
                    "edge_count": result.edge_count
                })
            
            # Convert result to dict
            result_dict = result.model_dump(exclude={"graph_instance"})
            
            execution_time = (time.time() - start_time) * 1000
            logger.info(f"MCP: build_er_graph completed in {execution_time:.1f}ms")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"MCP: build_er_graph failed: {e}")
            
            # Report error
            if "progress_callback" in params and params["progress_callback"]:
                await params["progress_callback"]({
                    "status": "failed",
                    "message": str(e),
                    "progress": 0
                })
            
            return {
                "graph_id": f"{params.get('target_dataset_name', 'unknown')}_ERGraph",
                "status": "failure",
                "message": str(e),
                "node_count": None,
                "edge_count": None,
                "layer_count": None,
                "artifact_path": None
            }


# Create singleton instance
build_er_graph_mcp_tool = BuildERGraphMCPTool()