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
    """Test the DIGIMON pipeline on fictional Zorathian Empire corpus with multiple queries"""
    
    # Initialize configuration
    config = Config.default()
    logger.info("Initialized config")
    
    # Set up the corpus directory
    corpus_name = "Fictional_Test"
    # corpus_dir = f"Data/{corpus_name}" # Not directly used in this function scope anymore
    
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

    queries_to_test = [
        "What led to the downfall of the Zorathian Empire?",
        "Describe the societal structure of the Zorathians.",
        "Tell me about Zorathian technological advancements.",
        "Who were the key figures in the Zorathian Empire and what were their roles?",
        "What was the 'Crystal Plague' and how did it affect the Zorathians?",
        "Did the Zorathians interact with other species or empires?"
    ]

    all_query_results = []

    for test_query in queries_to_test:
        logger.info(f"\nProcessing query: {test_query}")
        
        answer = "Error or no answer generated."
        context = {}
        
        try:
            # Process the query
            result = await planning_agent.process_query(test_query, actual_corpus_name=corpus_name)
            
            print("\n" + "="*80)
            print(f"QUERY: {test_query}")
            print("="*80)
            
            # Extract the generated answer from the result dict
            if isinstance(result, dict):
                answer = result.get('generated_answer', 'No answer found in result dict.')
                context = result.get('retrieved_context', {})
                print(f"\nAnswer:\n{answer}")
                logger.info(f"Retrieved context keys for query '{test_query}': {list(context.keys())}")
                # Optionally log full context, but can be verbose:
                # for step_id, outputs in context.items():
                #     logger.debug(f"Step {step_id} outputs for query '{test_query}': {outputs}")
            else:
                answer = str(result) # Should ideally be a dict from process_query
                print(f"\nAnswer (raw result):\n{answer}")
            
            print("="*80 + "\n")
            
            # Simplified grounding check for stress test
            answer_lower = answer.lower()
            # General keywords that should appear if grounded in Zorathian lore
            general_zorathian_keywords = ["zorathian", "zorthak", "aerophantis", "crystal", "empire", "emperor", "plague"]
            if any(keyword in answer_lower for keyword in general_zorathian_keywords):
                logger.info(f"✅ INFO: Answer for '{test_query}' seems to reference Zorathian corpus content.")
            else:
                logger.warning(f"⚠️  WARNING: Answer for '{test_query}' may not be using Zorathian corpus content. Answer: {answer[:200]}...")

            all_query_results.append({"query": test_query, "answer": answer, "context_keys": list(context.keys())})
            
        except Exception as e:
            logger.error(f"Error during query processing for '{test_query}': {e}")
            import traceback
            traceback.print_exc()
            all_query_results.append({"query": test_query, "answer": f"ERROR: {e}", "context_keys": []})
            
    logger.info("Completed all test queries.")
    # The function no longer returns a single result, but processes multiple.
    # If needed, could return all_query_results or write them to a file.


if __name__ == "__main__":
    asyncio.run(test_fictional_corpus())
