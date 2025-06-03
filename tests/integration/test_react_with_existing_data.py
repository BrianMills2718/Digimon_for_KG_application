#!/usr/bin/env python3
"""
Test ReACT implementation with existing corpus and graph data.
This test assumes the Fictional_Test corpus is already prepared and graphs are built.
"""

import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from Option.Config2 import Config
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.Graph.GraphFactory import GraphFactory
from Core.Index.FaissIndex import FaissIndex
from Core.AgentTools.graph_construction_tools import build_er_graph
from Core.AgentTools.entity_vdb_tools import entity_vdb_build_tool
from Core.AgentSchema.graph_construction_tool_contracts import BuildERGraphInputs
from Core.AgentSchema.tool_contracts import EntityVDBBuildInputs


async def prepare_existing_data(graphrag_context, corpus_name):
    """Build the graph and VDB if they don't exist."""
    logger.info(f"Checking and preparing data for corpus: {corpus_name}")
    
    # Check if ER graph exists
    er_graph_id = f"{corpus_name}_ERGraph"
    if not graphrag_context.get_graph_instance(er_graph_id):
        logger.info("Building ER graph...")
        graph_input = BuildERGraphInputs(
            corpus_json_path=f"Results/{corpus_name}/Corpus.json",
            graph_id=er_graph_id,
            save_graph_path=f"Results/{corpus_name}/er_graph.pkl",
            target_dataset_name=corpus_name
        )
        graph_build_result = await build_er_graph(
            graph_input, 
            main_config=graphrag_context.main_config,
            llm_instance=graphrag_context.llm_provider,
            encoder_instance=graphrag_context.embedding_provider,
            chunk_factory=graphrag_context.chunk_storage_manager
        )
        if graph_build_result.status == "success" and graph_build_result.graph_instance:
            graphrag_context.add_graph_instance(er_graph_id, graph_build_result.graph_instance)
            logger.info(f"ER graph '{er_graph_id}' built and registered successfully.")
        else:
            logger.error(f"ER graph building failed: {graph_build_result.message}")
    else:
        logger.info("ER graph already exists")
    
    # Check and prepare VDB
    vdb_context_key = f"{corpus_name}_entities"
    if not graphrag_context.get_vdb_instance(vdb_context_key):
        logger.info(f"VDB '{vdb_context_key}' not found in context. Building VDB...")
        vdb_input = EntityVDBBuildInputs(
            graph_reference_id=er_graph_id, # From the graph building step
            vdb_id=vdb_context_key, # ID for the VDB instance itself
            vdb_collection_name=vdb_context_key # Name of collection, used as context key by tool
            # force_rebuild defaults to False, tool handles logic if it's already in context but we are building because our initial check failed.
        )
        vdb_build_result = await entity_vdb_build_tool(vdb_input, graphrag_context=graphrag_context)
        
        # The tool logs its own success/failure and registration status.
        # We add a summary log here based on the tool's output.
        if vdb_build_result.status.lower().startswith("error") or \
           (vdb_build_result.status.lower() != "vdb already exists" and vdb_build_result.num_entities_indexed == 0):
            logger.error(f"VDB building/registration for '{vdb_context_key}' may have failed or resulted in an empty index. Status: {vdb_build_result.status}, Entities: {vdb_build_result.num_entities_indexed}")
        else:
            logger.info(f"VDB build/registration process for '{vdb_context_key}' completed. Status: {vdb_build_result.status}, Entities indexed: {vdb_build_result.num_entities_indexed}.")
            # Optional: Verify if it's in context, though the tool should have logged this.
            if graphrag_context.get_vdb_instance(vdb_context_key):
                logger.info(f"Confirmed: VDB '{vdb_context_key}' is registered in context.")
            else:
                # This case should ideally not happen if the tool reported success and is supposed to register.
                logger.error(f"Error: VDB '{vdb_context_key}' NOT found in context after build call, despite tool output: {vdb_build_result.status}")
    else:
        logger.info(f"VDB '{vdb_context_key}' already exists in context. Skipping build.")


async def test_react_with_existing_data():
    """Test ReACT with existing corpus, graph, and VDB."""
    
    # Load configuration
    config = Config.from_yaml_file("Option/Config2.yaml")
    
    # Initialize providers
    llm_provider = LiteLLMProvider(config.llm)
    embedding_provider = get_rag_embedding(config=config)
    
    # Setup corpus - assuming Fictional_Test already exists
    corpus_name = "Fictional_Test"
    
    # Initialize chunk factory
    chunk_factory = ChunkFactory(config)
    
    # Create GraphRAG context
    graphrag_context = GraphRAGContext(
        target_dataset_name=corpus_name,
        main_config=config,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        chunk_storage_manager=chunk_factory
    )
    
    # Prepare existing data
    await prepare_existing_data(graphrag_context, corpus_name)
    
    # Initialize PlanningAgent
    planning_agent = PlanningAgent(
        config=config,
        graphrag_context=graphrag_context
    )
    
    # Test queries that should work with existing data
    test_queries = [
        "What are the Zorathian Empire's relationships with other entities?"
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing query: {query}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Process query with ReACT
            result = await planning_agent.process_query_react(query, actual_corpus_name=corpus_name)
            
            if "error" in result:
                logger.error(f"Error processing query: {result['error']}")
            else:
                logger.info(f"\nFINAL ANSWER: {result.get('generated_answer', 'No answer generated')}")
                logger.info(f"\nOBSERVATIONS MADE: {len(result.get('observations', []))}")
                for i, obs in enumerate(result.get('observations', []), 1):
                    logger.info(f"  {i}. {obs.get('step', 'Unknown step')}: {obs.get('summary', 'No summary')}")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
        
        # Add delay between queries
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(test_react_with_existing_data())
