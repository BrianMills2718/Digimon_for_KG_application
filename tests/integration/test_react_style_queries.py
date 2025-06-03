#!/usr/bin/env python3
"""
Test script to stress-test the DIGIMON pipeline with ReACT-style reasoning queries.
These queries are designed to test iterative reasoning capabilities.
"""

import asyncio
import logging
from pathlib import Path
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Option.Config2 import Config
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.Common.Logger import logger

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_react_style_queries():
    """Test the pipeline with ReACT-style iterative reasoning queries."""
    
    # Initialize configuration
    config_path = Path("Option/Config2.yaml")
    corpus_name = "Fictional_Test"
    
    logger.info(f"Initializing test with corpus: {corpus_name}")
    
    # Load configuration
    config = Config.parse(config_path, dataset_name=corpus_name)
    
    # Initialize providers
    llm_provider = LiteLLMProvider(config.llm)
    embedding_provider = get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    # Create GraphRAGContext
    from Core.AgentSchema.context import GraphRAGContext
    graphrag_context = GraphRAGContext(
        target_dataset_name=corpus_name,
        main_config=config,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        chunk_storage_manager=chunk_factory
    )
    
    # Initialize the planning agent
    planning_agent = PlanningAgent(
        config=config,
        graphrag_context=graphrag_context
    )
    
    # Define ReACT-style queries that would benefit from iterative reasoning
    react_queries = [
        {
            "query": "What caused the Crystal Plague and how did it lead to the fall of the Zorathian Empire? First identify the plague's origin, then trace its effects.",
            "reasoning_steps": ["Identify Crystal Plague origin", "Trace effects on society", "Connect to empire's fall"]
        },
        {
            "query": "Compare the technological capabilities of the Zorathians with the Kingdom of Mystara. What advantages did each have?",
            "reasoning_steps": ["Find Zorathian tech", "Find Mystara tech", "Compare and contrast"]
        },
        {
            "query": "Analyze the leadership succession from Emperor Zorthak the Luminous to Emperor Zorthak III. What changes occurred and why?",
            "reasoning_steps": ["Identify first emperor", "Identify successor", "Analyze changes between reigns"]
        },
        {
            "query": "What was the relationship between the Xelandra and the Zorathian Empire? Were they allies, enemies, or something else?",
            "reasoning_steps": ["Find Xelandra mentions", "Find interaction patterns", "Determine relationship type"]
        },
        {
            "query": "Trace the timeline of the Zorathian Empire's decline. What were the key events in chronological order?",
            "reasoning_steps": ["Identify decline markers", "Order events temporally", "Create timeline"]
        },
        {
            "query": "How did the Aerophantis factor into the Zorathian Empire's story? What role did they play?",
            "reasoning_steps": ["Find Aerophantis references", "Determine their role", "Assess impact on empire"]
        }
    ]
    
    # Run each query
    for i, query_info in enumerate(react_queries, 1):
        query = query_info["query"]
        expected_steps = query_info["reasoning_steps"]
        
        print(f"\n{'='*80}")
        print(f"REACT-STYLE QUERY {i}: {query}")
        print(f"Expected reasoning steps: {expected_steps}")
        print(f"{'='*80}\n")
        
        try:
            # Process the query
            result = await planning_agent.process_query(query, actual_corpus_name=corpus_name)
            
            if isinstance(result, dict):
                answer = result.get('generated_answer', '')
                context = result.get('retrieved_context', {})
                
                print(f"Answer:\n{answer}")
                
                # Log the execution plan to see if it captures multi-step reasoning
                logger.info(f"Retrieved context keys for ReACT query {i}: {list(context.keys())}")
                
                # Analyze if the answer addresses the multi-part nature of the query
                reasoning_indicators = ["first", "then", "because", "therefore", "as a result", "consequently"]
                has_reasoning = any(indicator in answer.lower() for indicator in reasoning_indicators)
                
                if has_reasoning:
                    logger.info(f"✅ Answer shows reasoning structure for query {i}")
                else:
                    logger.warning(f"⚠️ Answer may lack explicit reasoning structure for query {i}")
                
                # Check if answer addresses multiple aspects
                query_parts = query.lower().split('?')
                addressed_parts = sum(1 for part in query_parts[:-1] if any(word in answer.lower() for word in part.split()))
                
                logger.info(f"Query {i} has {len(query_parts)-1} parts, answer addresses approximately {addressed_parts} parts")
                
            else:
                logger.error(f"Unexpected result type for query {i}: {type(result)}")
                
        except Exception as e:
            logger.error(f"Error processing ReACT query {i}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    logger.info("\nReACT-style query testing completed.")
    logger.info("\nNote: The current system generates a full execution plan upfront.")
    logger.info("True ReACT-style iterative reasoning would require:")
    logger.info("1. Execute one step")
    logger.info("2. Observe the result")
    logger.info("3. Reason about next step based on observation")
    logger.info("4. Repeat until answer is found")

if __name__ == "__main__":
    asyncio.run(test_react_style_queries())
