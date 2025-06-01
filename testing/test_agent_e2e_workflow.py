"""
End-to-end test for agent's ability to execute a two-step workflow:
1. Prepare corpus from directory of text files
2. Build ERGraph from the prepared corpus

This test validates the agent's planning and tool-chaining capabilities
with real operations, including live LLM calls.
"""
import asyncio
import os
import sys
from pathlib import Path
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.Common.Logger import logger
from Option.Config2 import Config
from Core.GraphRAG import GraphRAG
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentBrain.agent_brain import PlanningAgent

async def test_agent_corpus_prep_then_er_graph_build():
    """
    Test that the agent can execute a two-step workflow:
    1. Prepare corpus from directory of text files
    2. Build ERGraph from the prepared corpus
    """
    logger.info("Starting test_agent_corpus_prep_then_er_graph_build...")
    
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
        corpus_output_dir_name = "MySampleTexts_Corpus_E2E"
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
        
        # Initialize GraphRAG context directly
        graphrag_context = GraphRAGContext(main_config=main_config, target_dataset_name=corpus_output_dir_name)
        
        # Create agent brain
        agent_brain = PlanningAgent(config=main_config, graphrag_context=graphrag_context)
        
        # Craft the agent task
        agent_task = (
            f"SYSTEM_TASK: Process the text files from the directory '{input_txt_dir}' "
            f"to create a corpus, saving the Corpus.json in '{corpus_output_dir_path}'. "
            f"Let's refer to this prepared dataset as '{corpus_output_dir_name}'. "
            f"Then, build an ERGraph using the dataset named '{corpus_output_dir_name}', "
            f"forcing a rebuild. Use two-step extraction for the ERGraph."
        )
        
        logger.info(f"Sending task to agent: {agent_task}")
        
        # Process the query with the agent
        agent_response = await agent_brain.process_query(agent_task)
        
        logger.info(f"Agent response: {agent_response}")
        
        # Verify corpus creation
        corpus_json_path = Path(corpus_output_dir_path) / "Corpus.json"
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
        else:
            logger.error(f"❌ Corpus.json not found at {corpus_json_path}")
            return False
        
        # Verify ERGraph creation
        graph_dir = Path(corpus_output_dir_path) / "er_graph"
        graph_file = graph_dir / "nx_data.graphml"
        
        if graph_dir.exists() and graph_file.exists():
            logger.info(f"✅ ERGraph created at {graph_file}")
            logger.info(f"ERGraph size: {graph_file.stat().st_size} bytes")
        else:
            logger.error(f"❌ ERGraph not found at expected location: {graph_file}")
            return False
        
        logger.info("✅ End-to-end test completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error in test_agent_corpus_prep_then_er_graph_build: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Run the test"""
    success = await test_agent_corpus_prep_then_er_graph_build()
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
    return success

if __name__ == "__main__":
    asyncio.run(main())
