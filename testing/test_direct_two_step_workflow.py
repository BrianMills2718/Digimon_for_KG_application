"""
Test for direct execution of a two-step workflow:
1. Prepare corpus from directory of text files
2. Build ERGraph from the prepared corpus

This test bypasses the agent plan generation and directly calls
the tool functions to verify they work in sequence.
"""
import sys
import os
import json
import asyncio
from pathlib import Path
from typing import List, Tuple, Any
import logging
from loguru import logger
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Now import modules from the project
from Config.LLMConfig import LLMConfig
# Import the real embedding model factory
from Core.Index.EmbeddingFactory import get_rag_embedding

from Core.Common.Logger import logger
from Option.Config2 import Config
from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
from Core.AgentTools.graph_construction_tools import build_er_graph
from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs
from Core.AgentSchema.graph_construction_tool_contracts import BuildERGraphInputs, ERGraphConfigOverrides
from Core.AgentSchema.context import GraphRAGContext
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Chunk.ChunkFactory import ChunkingFactory, ChunkFactory
import json
from pathlib import Path
from Core.Schema.ChunkSchema import TextChunk

# Previously had a MockChunkFactory here, now using the real ChunkFactory from Core.Chunk.ChunkFactory

# We're now using the real embedding model from Core.Index.EmbeddingFactory

async def test_direct_two_step_workflow():
    """
    Test direct execution of the two-step workflow:
    1. Prepare corpus from directory
    2. Build ERGraph from the prepared corpus
    """
    logger.info("Starting test_direct_two_step_workflow...")
    
    try:
        # Initialize Config
        main_config = Config.default()
        
        # Set paths
        project_root = Path(__file__).resolve().parent.parent
        main_config.data_root = str(project_root / "Data")
        main_config.working_dir = str(project_root / "results")
        
        logger.info(f"Using data_root: {main_config.data_root}")
        logger.info(f"Using working_dir: {main_config.working_dir}")
        
        # Setup input/output paths
        input_txt_dir = str(Path(main_config.data_root) / "MySampleTexts")
        corpus_output_dir_name = "MySampleTexts_Corpus_Direct"
        corpus_output_dir_path = str(Path(main_config.working_dir) / corpus_output_dir_name)
        
        # Create output directory if it doesn't exist
        os.makedirs(corpus_output_dir_path, exist_ok=True)
        
        # Check for text files in input directory
        input_files = list(Path(input_txt_dir).glob("*.txt"))
        if not input_files:
            logger.error(f"No .txt files found in {input_txt_dir}")
            return False
        
        logger.info(f"Found {len(input_files)} text files in {input_txt_dir}")
        for file in input_files:
            logger.info(f"  - {file.name} ({file.stat().st_size} bytes)")
        
        # Step 1: Prepare corpus from directory
        logger.info("STEP 1: Preparing corpus from directory...")
        
        corpus_inputs = PrepareCorpusInputs(
            input_directory_path=input_txt_dir,
            output_directory_path=corpus_output_dir_path,
            target_corpus_name=corpus_output_dir_name
        )
        
        # Call the corpus preparation tool directly
        corpus_result = await prepare_corpus_from_directory(corpus_inputs)
        
        logger.info(f"Corpus preparation completed with status: {corpus_result.status}")
        logger.info(f"Corpus message: {corpus_result.message}")
        logger.info(f"Documents processed: {corpus_result.document_count}")
        logger.info(f"Corpus JSON path: {corpus_result.corpus_json_path}")
        
        # Verify corpus creation
        corpus_json_path = Path(corpus_result.corpus_json_path)
        if corpus_json_path.exists():
            logger.info(f"✅ Corpus.json created at {corpus_json_path}")
            logger.info(f"Corpus.json size: {corpus_json_path.stat().st_size} bytes")
            
            # Check corpus content
            with open(corpus_json_path, 'r') as f:
                corpus_content = f.read()
                
            contains_american = "american_revolution" in corpus_content
            contains_french = "french_revolution" in corpus_content
            
            if contains_american and contains_french:
                logger.info("✅ Corpus.json contains entries for both american_revolution and french_revolution")
            else:
                logger.error(f"❌ Corpus.json missing entries: american_revolution: {contains_american}, french_revolution: {contains_french}")
                return False
        else:
            logger.error(f"❌ Corpus.json not found at expected path: {corpus_json_path}")
            return False
        
        # Step 2: Build ERGraph from the prepared corpus
        logger.info("STEP 2: Building ERGraph from prepared corpus...")
        
        # Create GraphRAGContext with the target dataset name
        graphrag_context = GraphRAGContext(
            main_config=main_config,
            target_dataset_name=corpus_output_dir_name
        )
        
        # Use the real LiteLLMProvider with properly configured API key
        from Core.Provider.LiteLLMProvider import LiteLLMProvider
        import asyncio
        
        # Get LLM config from the main configuration
        llm_config = main_config.llm
        
        # Initialize LiteLLMProvider with the config
        llm_provider = LiteLLMProvider(llm_config)
        
        # Add semaphore for rate limiting LLM calls
        llm_provider.semaphore = asyncio.Semaphore(3)  # Allow 3 concurrent LLM calls
        
        logger.info(f"Using real LiteLLMProvider with model: {llm_config.model}")
        if not llm_config.api_key:
            logger.warning("No API key found in config. The test may fail when making API calls.")
            logger.warning("Please ensure API keys are configured in Option/Config2.yaml")
        
        # Use real embedding model instead of MockEncoder
        real_encoder = get_rag_embedding(config=main_config)
        logger.info(f"Initialized REAL Encoder from EmbeddingFactory: {type(real_encoder)}")
        
        # Use the real ChunkFactory implementation that handles JSONL format
        chunk_factory = ChunkFactory(config=main_config)
        logger.info(f"Using real ChunkFactory implementation for handling JSONL Corpus.json files")
        
        # Create ERGraph build inputs
        er_graph_inputs = BuildERGraphInputs(
            target_dataset_name=corpus_output_dir_name,
            force_rebuild=True,
            config_overrides=ERGraphConfigOverrides(
                extraction_steps=2
            )
        )
        
        # Call the ERGraph build tool directly
        er_graph_result = await build_er_graph(
            tool_input=er_graph_inputs,
            main_config=main_config,
            llm_instance=llm_provider,
            encoder_instance=real_encoder,  # Use the real encoder
            chunk_factory=chunk_factory
        )
        
        logger.info(f"ERGraph build completed with status: {er_graph_result.status}")
        logger.info(f"ERGraph build message: {er_graph_result.message}")
        logger.info(f"ERGraph ID: {er_graph_result.graph_id}")
        
        # Verify ERGraph creation
        graph_dir = Path(corpus_output_dir_path) / "er_graph"
        graph_file = graph_dir / "nx_data.graphml"
        
        if graph_dir.exists() and graph_file.exists():
            logger.info(f"✅ ERGraph created at {graph_file}")
            logger.info(f"ERGraph size: {graph_file.stat().st_size} bytes")
        else:
            logger.error(f"❌ ERGraph not found at expected location: {graph_file}")
            return False
        
        logger.info("✅ Two-step workflow test completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error in test_direct_two_step_workflow: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Run the test"""
    success = await test_direct_two_step_workflow()
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
    return success

if __name__ == "__main__":
    asyncio.run(main())
