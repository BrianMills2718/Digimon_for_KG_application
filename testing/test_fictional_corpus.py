#!/usr/bin/env python3
"""
Test script for running DIGIMON pipeline on fictional corpus
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from Option.Config2 import Config
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Graph.GraphFactory import GraphFactory


async def test_fictional_corpus():
    """Test the DIGIMON pipeline on fictional Zorathian Empire corpus"""
    
    # Initialize configuration
    config = Config.default()
    logger.info("Initialized config")
    
    # Set up the corpus directory
    corpus_name = "Fictional_Test"
    corpus_dir = f"Data/{corpus_name}"
    
    # Initialize providers
    llm_provider = LiteLLMProvider(config.llm)
    embedding_provider = get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    # Create GraphRAGContext
    graphrag_context = GraphRAGContext(
        target_dataset_name=corpus_name,
        main_config=config,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        chunk_storage_manager=chunk_factory
    )
    
    # Initialize PlanningAgent with correct corpus name
    planning_agent = PlanningAgent(
        config=config,
        graphrag_context=graphrag_context
    )
    logger.info(f"Initialized PlanningAgent with corpus from GraphRAGContext: {corpus_name}")
    
    # Test query about the fictional empire
    test_query = "What caused the fall of the Zorathian Empire?"
    logger.info(f"Running query: {test_query}")
    
    try:
        # Process the query
        result = await planning_agent.process_query(test_query, actual_corpus_name=corpus_name)
        
        print("\n" + "="*80)
        print("QUERY RESULT")
        print("="*80)
        print(f"Query: {test_query}")
        
        # Extract the generated answer from the result dict
        if isinstance(result, dict):
            answer = result.get('generated_answer', '')
            context = result.get('retrieved_context', {})
            print(f"\nAnswer:\n{answer}")
            print(f"\nContext retrieved: {context}")
        else:
            answer = str(result)
            print(f"\nAnswer:\n{answer}")
        
        print("="*80 + "\n")
        
        # Log retrieved context details
        logger.info(f"Retrieved context keys: {list(context.keys())}")
        for step_id, outputs in context.items():
            logger.info(f"Step {step_id} outputs: {outputs}")
        
        # Check if answer is grounded in the fictional corpus
        answer_lower = answer.lower()
        expected_keywords = ["crystal", "plague", "aerophantis", "zorthak", "emperor"]
        
        if any(keyword in answer_lower for keyword in expected_keywords):
            logger.info("✅ SUCCESS: Answer is grounded in the fictional corpus content!")
            # Check specifically for the correct cause
            if "crystal" in answer_lower or "plague" in answer_lower:
                logger.info("✅ EXCELLENT: Answer correctly identifies the Crystal Plague as the cause!")
            elif "aerophantis" in answer_lower:
                logger.info("✅ GOOD: Answer references relevant events from the corpus!")
        else:
            logger.warning("⚠️  WARNING: Answer may not be using the fictional corpus content")
            
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        import traceback
        traceback.print_exc()
        
    return result


if __name__ == "__main__":
    asyncio.run(test_fictional_corpus())
