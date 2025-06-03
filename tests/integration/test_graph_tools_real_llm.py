#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for testing graph construction tools with real LLM calls,
focusing on a true end-to-end integration test with actual API calls.
"""

import asyncio
import os
import sys
import json
import networkx as nx
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple, Set
import logging
from copy import deepcopy

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Core.Common.Logger import logger
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.NameSpace import NameSpace, Workspace
from Option.Config2 import Config
from Core.AgentSchema.graph_construction_tool_contracts import (
    BuildERGraphInputs, BuildERGraphOutputs,
    ERGraphConfigOverrides
)
from Core.AgentTools.graph_construction_tools import (
    build_er_graph, get_graph_counts, get_artifact_path
)
from Core.Graph.GraphFactory import get_graph
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Provider.LiteLLMProvider import LiteLLMProvider

class ChunkFactory:
    """A real ChunkFactory implementation for the test that works with a small test dataset"""
    
    def __init__(self, config: Config): 
        self.main_config = config
        self.workspaces: Dict[str, Workspace] = {}
        logger.info(f"TestChunkFactory initialized with working_dir: {self.main_config.working_dir}")

    def get_namespace(self, dataset_name: str, graph_type: str = "er_graph") -> NameSpace:
        # Ensure dataset_name is not empty
        if not dataset_name:
            raise ValueError("dataset_name cannot be empty for creating a namespace.")
            
        # Use working_dir from the main_config
        base_dataset_path = Path(self.main_config.working_dir) / dataset_name
        
        if dataset_name not in self.workspaces:
            self.workspaces[dataset_name] = Workspace(working_dir=str(base_dataset_path.parent), exp_name=dataset_name)
        
        namespace = self.workspaces[dataset_name].make_for(graph_type)
        # Ensure the namespace path exists (using get_save_path method)
        namespace_path = namespace.get_save_path()
        Path(namespace_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"TestChunkFactory: Created/ensured namespace for dataset '{dataset_name}', type '{graph_type}' at {namespace_path}")
        return namespace

    async def get_chunks_for_dataset(self, dataset_name: str) -> List[Tuple[str, TextChunk]]:
        logger.info(f"TestChunkFactory: Getting chunks for dataset '{dataset_name}'")
        if dataset_name == "american_revolution_doc":
            # Using content from american_revolution document for testing
            # Create chunks with token counts for each chunk's content
            content1 = "The American Revolution (1765–1783) was an ideological and political movement in the Thirteen Colonies in what was then British America. The revolution culminated in the American Revolutionary War, which was launched on April 19, 1775, in the Battles of Lexington and Concord."
            content2 = "Leaders of the American Revolution were colonial separatist leaders who, as British subjects, initially sought incremental levels of autonomy but came to embrace the cause of full independence and the necessity of prevailing in the Revolutionary War to obtain it."
            
            chunks = [
                ("chunk_0_1", TextChunk(
                    chunk_id="chunk_0_1",
                    doc_id="american_revolution_doc",
                    index=0,
                    content=content1,
                    title="american_revolution",
                    tokens=len(content1.split())  # Estimate token count based on word count
                )),
                ("chunk_0_2", TextChunk(
                    chunk_id="chunk_0_2",
                    doc_id="american_revolution_doc",
                    index=1,
                    content=content2,
                    title="american_revolution",
                    tokens=len(content2.split())  # Estimate token count based on word count
                ))
            ]
            logger.info(f"TestChunkFactory: Returning {len(chunks)} chunks for '{dataset_name}'")
            return chunks
        logger.warning(f"TestChunkFactory: No chunks defined for dataset '{dataset_name}'")
        return []


async def test_build_er_graph_real_llm():
    logger.info("Starting test_build_er_graph_real_llm (with REAL LLM calls for extraction)...")
    try:
        # Load the main configuration using Config.default()
        main_config = Config.default()
        
        # Override specific config values if necessary for the test
        main_config.data_root = str(Path(project_root) / "Data")
        main_config.working_dir = str(Path(project_root) / "results")
        
        # Create the working directory if it doesn't exist
        Path(main_config.working_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Loaded main_config. LLM model: {main_config.llm.model}, Embedding model: {main_config.embedding.model}")
        logger.info(f"Data root: {main_config.data_root}, Working directory: {main_config.working_dir}")
        
        if main_config.llm.api_key:
            logger.info(f"LLM API key found, starting with: {main_config.llm.api_key[:8]}...")
        else:
            logger.error("LLM API key not found in config. Test will likely fail.")
            raise ValueError("LLM API Key not configured in Option/Config2.yaml")

        # Initialize LLM Provider
        llm_provider = LiteLLMProvider(config=main_config.llm)
        
        # Initialize LLM semaphore if not already set
        if not hasattr(llm_provider, 'semaphore'):
            if hasattr(main_config.llm, 'concurrent_requests') and main_config.llm.concurrent_requests is not None:
                llm_provider.semaphore = asyncio.Semaphore(main_config.llm.concurrent_requests)
            else:
                llm_provider.semaphore = asyncio.Semaphore(1)
                
        logger.info(f"Initialized LLM provider: {type(llm_provider).__name__} for model {llm_provider.model}")
        
        # Initialize the embedding factory and encoder
        embedding_factory = RAGEmbeddingFactory()
        encoder = embedding_factory.get_rag_embedding(config=main_config)
        logger.info(f"Initialized Encoder: {type(encoder).__name__}")

        # Initialize the test chunk factory
        test_chunk_factory = ChunkFactory(config=main_config)
        logger.info(f"Initialized ChunkFactory for real extraction test")
        
        # Set up the target dataset and prepare tool inputs
        target_dataset_name = "american_revolution_doc"
        
        tool_input = BuildERGraphInputs(
            target_dataset_name=target_dataset_name,
            force_rebuild=True,
            config_overrides=ERGraphConfigOverrides(
                extract_two_step=True, 
                enable_entity_description=True,
                enable_entity_type=True
            )
        )
        logger.info(f"Prepared BuildERGraphInputs: {tool_input}")

        # Call the actual build_er_graph tool
        logger.info("Calling actual build_er_graph tool with REAL LLM extraction...")
        result = await build_er_graph(
            tool_input=tool_input,
            main_config=main_config,
            llm_instance=llm_provider,
            encoder_instance=encoder,
            chunk_factory=test_chunk_factory 
        )

        logger.info(f"REAL LLM ER Graph build result: {result}")
        assert result.status == "success", f"Build failed: {result.message}"
        assert result.graph_id == f"{target_dataset_name}_ERGraph"
        assert result.node_count is not None and result.node_count >= 0, "Node count should be available"
        assert result.edge_count is not None and result.edge_count >= 0, "Edge count should be available"
        
        # Even if artifact_path is None, we can construct the expected path
        namespace_path = test_chunk_factory.get_namespace(target_dataset_name).get_save_path()
        expected_graph_file = Path(namespace_path) / "nx_data.graphml"
        
        # Check if the graph file exists at the expected location
        assert expected_graph_file.exists(), f"Graph file not found at expected location: {expected_graph_file}"
        logger.info(f"Graph artifact verified at: {expected_graph_file}")

        logger.info("test_build_er_graph_real_llm PASSED!")
        return result
    except Exception as e:
        logger.error(f"Error in test_build_er_graph_real_llm: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Setup for tests - ensure working directory exists
    main_config = Config.default()
    if not Path(main_config.working_dir).exists():
        Path(main_config.working_dir).mkdir(parents=True, exist_ok=True)
    
    # Run the real LLM test
    asyncio.run(test_build_er_graph_real_llm())
