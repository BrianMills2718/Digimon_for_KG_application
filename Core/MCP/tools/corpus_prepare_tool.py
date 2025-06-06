"""
MCP wrapper for corpus.PrepareFromDirectory tool

Provides MCP-compatible interface for preparing corpus from directory.
"""

import asyncio
from typing import Dict, Any, Optional
from Core.MCP.shared_context import get_shared_context
from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs
from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
from Core.Common.Logger import logger
import time


class CorpusPrepareFromDirectoryMCPTool:
    """MCP wrapper for corpus.PrepareFromDirectory tool"""
    
    def __init__(self):
        self.tool_id = "corpus.PrepareFromDirectory"
        self.name = "corpus.PrepareFromDirectory"
        self.description = "Process text files in directory into corpus format"
        
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return MCP tool definition"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "Path to directory containing text files"
                    },
                    "target_dataset_name": {
                        "type": "string",
                        "description": "Name for the output dataset"
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File extensions to include",
                        "default": [".txt", ".md", ".doc", ".docx", ".pdf"]
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Process subdirectories recursively",
                        "default": True
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "Maximum number of files to process",
                        "default": 1000
                    }
                },
                "required": ["directory_path", "target_dataset_name"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "corpus_path": {"type": "string"},
                    "status": {"type": "string", "enum": ["success", "failure"]},
                    "documents_processed": {"type": "integer"},
                    "message": {"type": "string"}
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute tool via MCP"""
        start_time = time.time()
        
        try:
            # Note: This tool doesn't require GraphRAG context
            # It operates independently on the file system
            
            # Create input object
            inputs = PrepareCorpusInputs(
                directory_path=params["directory_path"],
                target_dataset_name=params["target_dataset_name"],
                file_extensions=params.get("file_extensions", [".txt", ".md", ".doc", ".docx", ".pdf"]),
                recursive=params.get("recursive", True),
                max_files=params.get("max_files", 1000)
            )
            
            # Execute tool - it requires main_config
            from Option.Config2 import Config as FullConfig
            
            # Get config from shared context if available
            shared_context = get_shared_context()
            config_data = shared_context.get("main_config", session_id)
            
            if not config_data:
                # Create minimal config
                config_data = {
                    "llm": {
                        "api_type": "openai",
                        "base_url": "https://api.openai.com/v1",
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key"
                    },
                    "embedding": {
                        "api_type": "openai",
                        "api_key": "test-key",
                        "model": "text-embedding-3-small"
                    },
                    "data_root": "./Data",
                    "working_dir": "./results"
                }
            
            config = FullConfig(**config_data) if isinstance(config_data, dict) else config_data
            
            logger.info(f"MCP: Executing prepare_corpus_from_directory")
            
            result = await prepare_corpus_from_directory(inputs, config)
            
            # Convert result to dict
            result_dict = result.model_dump()
            
            execution_time = (time.time() - start_time) * 1000
            logger.info(f"MCP: prepare_corpus_from_directory completed in {execution_time:.1f}ms")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"MCP: prepare_corpus_from_directory failed: {e}")
            
            return {
                "corpus_path": "",
                "status": "failure",
                "documents_processed": 0,
                "message": str(e)
            }


# Create singleton instance
corpus_prepare_mcp_tool = CorpusPrepareFromDirectoryMCPTool()