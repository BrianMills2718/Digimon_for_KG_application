#!/usr/bin/env python3
"""
Test the new ReACT implementation with step-by-step execution and reasoning.
"""

import asyncio
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Option.Config2 import Config
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentBrain.agent_brain import PlanningAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_react_implementation.log')
    ]
)
logger = logging.getLogger(__name__)

async def test_react_implementation():
    """Test the true iterative ReACT implementation"""
    
    # Load configuration
    config = Config.from_yaml_file("Option/Config2.yaml")
    
    # Initialize providers
    llm_provider = LiteLLMProvider(config.llm)
    embedding_provider = get_rag_embedding(config=config)
    
    # Setup corpus - the Fictional_Test corpus already exists in Data/Fictional_Test
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
    
    # Initialize PlanningAgent
    planning_agent = PlanningAgent(
        config=config,
        graphrag_context=graphrag_context
    )
    
    # First, prepare the corpus that already exists in Data/Fictional_Test
    # This step is crucial - the corpus needs to be indexed before we can query it
    from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs
    from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
    
    logger.info("Preparing corpus from existing text files...")
    prepare_input = PrepareCorpusInputs(
        input_directory_path=f"Data/{corpus_name}",
        output_directory_path=f"Results/{corpus_name}",
        target_corpus_name=corpus_name
    )
    
    # Prepare the corpus
    corpus_result = await prepare_corpus_from_directory(prepare_input, main_config=config)
    logger.info(f"Corpus preparation result: {corpus_result}")
    
    # Test queries that should trigger iterative ReACT behavior
    test_queries = [
        "What is the capital of the Zorathian Empire?",
        "Tell me about the relationship between the Zorathian Empire and Mystara Confederation",
        "What caused the downfall of the Zorathian Empire and who was ruling at the time?",
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing query: {query}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Execute query with ReACT
            result = await planning_agent.process_query_react(query)
            
            logger.info(f"\nFINAL ANSWER: {result.get('answer', 'No answer generated')}")
            
            if result.get('retrieved_context'):
                logger.info(f"\nRETRIEVED CONTEXT: {len(result['retrieved_context'])} items")
                
            if result.get('observations'):
                logger.info(f"\nOBSERVATIONS MADE: {len(result['observations'])}")
                for i, obs in enumerate(result['observations'][:3]):  # Show first 3
                    logger.info(f"  {i+1}. {obs.get('step', 'Unknown step')}: {obs.get('summary', 'No summary')}")
                    
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            
        # Brief pause between queries
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(test_react_implementation())
