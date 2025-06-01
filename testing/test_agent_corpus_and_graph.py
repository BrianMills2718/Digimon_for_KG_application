"""
Test script for the agent to execute a two-step workflow:
1. Prepare a corpus from a directory of text files
2. Build an ERGraph using the prepared corpus
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Common.Logger import logger
from Core.Storage.NameSpace import NameSpace, Workspace

async def test_agent_prepare_corpus_and_build_er_graph():
    """
    Test the agent's ability to execute a two-step plan:
    1. Prepare a corpus from text files in a directory
    2. Build an ERGraph using the prepared corpus
    """
    logger.info("Starting test_agent_prepare_corpus_and_build_er_graph...")
    try:
        # Initialize configuration
        main_config = Config.default()
        main_config.data_root = str(Path(__file__).resolve().parent.parent / "Data")
        main_config.working_dir = str(Path(__file__).resolve().parent.parent / "results")
        
        # Create working directory if it doesn't exist
        Path(main_config.working_dir).mkdir(parents=True, exist_ok=True)
        
        # Define dataset name for corpus and graph building
        corpus_output_dir_name = "MySampleTexts_Corpus_TestOutput"
        derived_dataset_name_for_graph = corpus_output_dir_name
        
        # Create GraphRAGContext with required target_dataset_name
        graphrag_context = GraphRAGContext(
            config=main_config,
            target_dataset_name=derived_dataset_name_for_graph
        )
        
        # Initialize PlanningAgent
        agent = PlanningAgent(config=main_config, graphrag_context=graphrag_context)
        
        # Define input and output paths
        input_txt_dir = str(Path(main_config.data_root) / "MySampleTexts")
        corpus_output_dir_path = str(Path(main_config.working_dir) / derived_dataset_name_for_graph)
        Path(corpus_output_dir_path).mkdir(parents=True, exist_ok=True)
        
        # Construct agent task
        agent_task = (
            f"SYSTEM_TASK: Step 1: Prepare a corpus. Input directory: '{input_txt_dir}'. "
            f"Output directory for Corpus.json: '{corpus_output_dir_path}'. Name this corpus '{derived_dataset_name_for_graph}'. "
            f"Step 2: Build an ERGraph using the dataset named '{derived_dataset_name_for_graph}'. "
            f"Force rebuild. Use two-step extraction. Enable entity descriptions and types."
        )
        logger.info(f"Agent Task: {agent_task}")
        
        # Process the query
        agent_response = await agent.process_query(agent_task)
        logger.info(f"Agent final response: {agent_response}")
        
        # Verify results
        expected_corpus_path = Path(corpus_output_dir_path) / "Corpus.json"
        if expected_corpus_path.exists():
            logger.info(f"Corpus.json found at {expected_corpus_path}")
            with open(expected_corpus_path, 'r', encoding='utf-8') as f:
                first_few_lines = [next(f) for _ in range(3)]
                logger.info(f"First few lines of Corpus.json: {first_few_lines}")
        else:
            logger.error(f"Corpus.json not found at expected path: {expected_corpus_path}")
        
        # Check for ERGraph artifact
        graph_artifact_ns = NameSpace(
            Workspace(
                working_dir=Path(main_config.working_dir) / derived_dataset_name_for_graph, 
                exp_name=derived_dataset_name_for_graph
            ), 
            "er_graph"
        )
        expected_er_graph_file = Path(graph_artifact_ns.path) / "nx_data.graphml"
        
        if expected_er_graph_file.exists():
            logger.info(f"ERGraph artifact found at {expected_er_graph_file}")
            logger.info(f"ERGraph file size: {expected_er_graph_file.stat().st_size} bytes")
        else:
            logger.error(f"ERGraph artifact not found at expected path: {expected_er_graph_file}")
        
        logger.info("test_agent_prepare_corpus_and_build_er_graph completed.")
        return agent_response
        
    except Exception as e:
        logger.error(f"Error in test_agent_prepare_corpus_and_build_er_graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

async def main():
    """Run the test"""
    await test_agent_prepare_corpus_and_build_er_graph()

if __name__ == "__main__":
    asyncio.run(main())
