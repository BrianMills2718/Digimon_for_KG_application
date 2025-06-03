"""
Test script for direct execution of the two-step workflow:
1. Prepare a corpus from a directory of text files
2. Build an ERGraph using the prepared corpus
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs, PrepareCorpusOutputs
from Core.AgentSchema.graph_construction_tool_contracts import BuildERGraphInputs, BuildERGraphOutputs, ERGraphConfigOverrides
from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
from Core.AgentTools.graph_construction_tools import build_er_graph
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Common.Logger import logger
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Embedding.SentenceTransformerEncoder import SentenceTransformerEncoder
from Core.Chunk.ChunkFactory import ChunkFactory

async def test_direct_corpus_and_er_graph():
    """
    Test direct execution of prepare_corpus_from_directory followed by build_er_graph.
    """
    logger.info("Starting test_direct_corpus_and_er_graph...")
    
    try:
        # Initialize configuration
        main_config = Config.default()
        main_config.data_root = str(Path(__file__).resolve().parent.parent / "Data")
        main_config.working_dir = str(Path(__file__).resolve().parent.parent / "results")
        
        # Create working directory if it doesn't exist
        Path(main_config.working_dir).mkdir(parents=True, exist_ok=True)
        
        # Define dataset name and paths
        dataset_name = "MySampleTexts_Corpus_Direct"
        input_txt_dir = str(Path(main_config.data_root) / "MySampleTexts")
        corpus_output_dir = str(Path(main_config.working_dir) / dataset_name)
        Path(corpus_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create GraphRAGContext with the target dataset name
        graphrag_context = GraphRAGContext(
            config=main_config,
            target_dataset_name=dataset_name
        )
        
        # Step 1: Prepare corpus from directory
        logger.info(f"Step 1: Preparing corpus from directory: {input_txt_dir}")
        corpus_inputs = PrepareCorpusInputs(
            input_directory_path=input_txt_dir,
            output_directory_path=corpus_output_dir,
            target_corpus_name=dataset_name
        )
        
        corpus_result = await prepare_corpus_from_directory(corpus_inputs)
        logger.info(f"Corpus preparation result: {corpus_result}")
        
        if corpus_result.status != "success":
            logger.error(f"Failed to prepare corpus: {corpus_result.message}")
            return
        
        # Verify corpus file exists
        corpus_file_path = Path(corpus_result.corpus_json_path)
        if corpus_file_path.exists():
            logger.info(f"Corpus file created at: {corpus_file_path}")
            with open(corpus_file_path, 'r', encoding='utf-8') as f:
                first_few_lines = [next(f) for _ in range(min(3, corpus_result.document_count))]
                logger.info(f"First few lines of corpus file: {first_few_lines}")
        else:
            logger.error(f"Corpus file not found at: {corpus_file_path}")
            return
        
        # Step 2: Build ERGraph using the prepared corpus
        logger.info(f"Step 2: Building ERGraph for dataset: {dataset_name}")
        
        # Initialize required components for graph building
        # 1. LLM provider
        llm_provider = LiteLLMProvider(model="openai/o4-mini-2025-04-16")
        
        # 2. Encoder for embeddings
        encoder = SentenceTransformerEncoder(device="cpu") 
        
        # 3. ChunkFactory for processing corpus
        chunk_factory = ChunkFactory(config=main_config)
        
        # Create graph input model
        graph_inputs = BuildERGraphInputs(
            target_dataset_name=dataset_name,
            force_rebuild=True
        )
        
        # Add config overrides for extraction settings
        graph_inputs.config_overrides = ERGraphConfigOverrides(
            extract_two_step=True,
            enable_entity_type=True,
            enable_entity_description=True
        )
        
        logger.info("Starting ERGraph build with all required components...")
        graph_result = await build_er_graph(
            tool_input=graph_inputs,
            main_config=main_config,
            llm_instance=llm_provider,
            encoder_instance=encoder,
            chunk_factory=chunk_factory
        )
        logger.info(f"ERGraph build result: {graph_result}")
        
        # Verify graph was built successfully
        if hasattr(graph_result, 'node_count') and graph_result.node_count > 0:
            logger.info(f"ERGraph built successfully with {graph_result.node_count} nodes and {graph_result.edge_count} edges")
            
            # Check artifact path
            if hasattr(graph_result, 'artifact_path') and graph_result.artifact_path:
                artifact_path = Path(graph_result.artifact_path)
                logger.info(f"ERGraph artifact directory: {artifact_path}")
                
                # Check for graphml file
                graphml_files = list(artifact_path.glob("*.graphml"))
                if graphml_files:
                    logger.info(f"Found graphml files: {graphml_files}")
                    for graphml_file in graphml_files:
                        logger.info(f"GraphML file size: {graphml_file.stat().st_size} bytes")
                else:
                    logger.error(f"No GraphML files found in artifact directory: {artifact_path}")
            else:
                logger.error("No artifact path returned in graph result")
        else:
            logger.error(f"ERGraph build failed or returned no nodes")
        
        logger.info("test_direct_corpus_and_er_graph completed successfully")
        return graph_result
        
    except Exception as e:
        logger.error(f"Error in test_direct_corpus_and_er_graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

async def main():
    """Run the test"""
    result = await test_direct_corpus_and_er_graph()
    return result

if __name__ == "__main__":
    asyncio.run(main())
