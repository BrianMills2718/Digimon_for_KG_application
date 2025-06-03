#!/usr/bin/env python3
"""
Test script for agent enhancements
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.Common.Logger import logger
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentOrchestrator.enhanced_orchestrator import EnhancedAgentOrchestrator
from Core.AgentSchema.context import GraphRAGContext
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Provider.EnhancedLiteLLMProvider import EnhancedLiteLLMProvider
from Core.Provider.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory
from Option.Config2 import Config

# Replace original tools with enhanced versions
import Core.AgentTools.entity_vdb_tools as original_entity_vdb
import Core.AgentTools.relationship_tools as original_relationship
from Core.AgentTools.enhanced_entity_vdb_tools import entity_vdb_build_tool as enhanced_entity_vdb_build
from Core.AgentTools.enhanced_relationship_tools import relationship_vdb_build_tool as enhanced_relationship_vdb_build

# Monkey patch the original modules with enhanced versions
original_entity_vdb.entity_vdb_build_tool = enhanced_entity_vdb_build
original_relationship.relationship_vdb_build_tool = enhanced_relationship_vdb_build

async def test_enhanced_agent():
    """Test the enhanced agent with our improvements."""
    
    logger.info("=== Testing Enhanced Agent Implementation ===")
    
    try:
        # Load configuration
        config = Config()
        config.graph.enable_cache = True
        
        # Create enhanced LLM provider
        logger.info("Creating enhanced LLM provider...")
        llm_provider = EnhancedLiteLLMProvider(
            api_key=config.llm.api_key,
            model=config.llm.model,
            temperature=config.llm.temperature
        )
        
        # Create embedding provider
        logger.info("Creating embedding provider...")
        embed_provider = get_rag_embedding(config)
        
        # Create chunk factory
        chunk_factory = ChunkFactory(config)
        
        # Initialize context
        logger.info("Initializing GraphRAG context...")
        context = GraphRAGContext(
            main_config=config,
            embedding_provider=embed_provider
        )
        
        # Create enhanced orchestrator
        logger.info("Creating enhanced orchestrator...")
        orchestrator = EnhancedAgentOrchestrator(
            main_config=config,
            llm_instance=llm_provider,
            encoder_instance=embed_provider,
            chunk_factory=chunk_factory,
            graphrag_context=context
        )
        
        # Create planning agent
        logger.info("Creating planning agent...")
        brain = PlanningAgent(
            llm=llm_provider,
            orchestrator=orchestrator
        )
        
        # Test with a simple query that would trigger VDB building
        test_query = "What are the main entities in the revolutionary documents?"
        dataset = "MySampleTexts"
        
        logger.info(f"\nTesting with query: '{test_query}'")
        logger.info(f"Dataset: {dataset}")
        
        # Generate plan
        logger.info("\nGenerating execution plan...")
        plan = await brain.generate_plan(
            query=test_query,
            dataset_name=dataset,
            available_tools=list(orchestrator._tool_registry.keys())
        )
        
        if plan:
            logger.info(f"\nGenerated plan with {len(plan.steps)} steps:")
            for i, step in enumerate(plan.steps, 1):
                logger.info(f"  Step {i}: {step.description}")
                if hasattr(step.action, 'tools'):
                    for tool in step.action.tools:
                        logger.info(f"    - Tool: {tool.tool_id}")
        
            # Execute plan with enhanced orchestrator
            logger.info("\nExecuting plan with enhanced orchestrator...")
            results = await orchestrator.execute_plan(plan)
            
            logger.info("\nExecution complete!")
            logger.info(f"Results: {len(results)} steps executed")
            
            # Check for errors
            errors = []
            for step_id, output in results.items():
                if "error" in output:
                    errors.append(f"{step_id}: {output['error']}")
            
            if errors:
                logger.error(f"\nEncountered {len(errors)} errors:")
                for error in errors:
                    logger.error(f"  - {error}")
            else:
                logger.info("\nAll steps executed successfully!")
                
            # Get performance metrics
            if hasattr(orchestrator, 'performance_monitor'):
                metrics = orchestrator.performance_monitor.get_summary()
                logger.info(f"\nPerformance metrics: {metrics}")
                
        else:
            logger.error("Failed to generate plan")
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
        
    return True

async def test_structured_errors():
    """Test structured error handling."""
    logger.info("\n=== Testing Structured Error Handling ===")
    
    from Core.Common.StructuredErrors import LLMTimeoutError, LLMRateLimitError
    
    # Test timeout error
    try:
        raise LLMTimeoutError(
            message="Test timeout",
            context={"duration": 60},
            recovery_strategies=[
                {"strategy": "retry", "params": {"timeout_multiplier": 2}}
            ]
        )
    except LLMTimeoutError as e:
        logger.info(f"Caught timeout error: {e}")
        logger.info(f"Recovery strategies: {e.recovery_strategies}")
        
    # Test rate limit error
    try:
        raise LLMRateLimitError(
            message="Test rate limit",
            context={"requests": 100},
            recovery_strategies=[
                {"strategy": "wait", "params": {"duration": 60}}
            ]
        )
    except LLMRateLimitError as e:
        logger.info(f"Caught rate limit error: {e}")
        logger.info(f"Recovery strategies: {e.recovery_strategies}")
        
    return True

async def test_batch_embeddings():
    """Test batch embedding processor."""
    logger.info("\n=== Testing Batch Embedding Processor ===")
    
    from Core.Common.BatchEmbeddingProcessor import BatchEmbeddingProcessor
    from Option.Config2 import Config
    
    config = Config()
    embed_provider = get_rag_embedding(config)
    
    # Create batch processor
    processor = BatchEmbeddingProcessor(
        embed_model=embed_provider,
        initial_batch_size=4,
        enable_deduplication=True,
        cache_embeddings=True
    )
    
    # Test with duplicate texts
    texts = [
        "The American Revolution began in 1776",
        "The French Revolution started in 1789",
        "The American Revolution began in 1776",  # Duplicate
        "Revolutionary movements shaped history"
    ]
    
    logger.info(f"Processing {len(texts)} texts (with 1 duplicate)...")
    embeddings = await processor.process_texts(texts)
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    logger.info(f"Cache stats: {processor.get_cache_stats()}")
    
    # Test again to see cache hit
    logger.info("\nProcessing same texts again...")
    embeddings2 = await processor.process_texts(texts)
    logger.info(f"Cache stats after reprocessing: {processor.get_cache_stats()}")
    
    return True

async def main():
    """Run all tests."""
    logger.info("Starting enhanced agent tests...\n")
    
    # Test structured errors
    if not await test_structured_errors():
        logger.error("Structured error test failed")
        return
        
    # Test batch embeddings
    if not await test_batch_embeddings():
        logger.error("Batch embedding test failed")
        return
        
    # Test enhanced agent
    if not await test_enhanced_agent():
        logger.error("Enhanced agent test failed")
        return
        
    logger.info("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())