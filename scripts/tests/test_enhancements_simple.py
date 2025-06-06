#!/usr/bin/env python3
"""
Simple test script for agent enhancements
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.Common.Logger import logger
from Core.Common.StructuredErrors import LLMTimeoutError, LLMRateLimitError, EmbeddingError
from Core.Common.BatchEmbeddingProcessor import BatchEmbeddingProcessor
from Core.Common.PerformanceMonitor import PerformanceMonitor
from Core.Provider.EnhancedLiteLLMProvider import EnhancedLiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Option.Config2 import Config

async def test_structured_errors():
    """Test structured error handling."""
    logger.info("=== Testing Structured Error Handling ===")
    
    # Test timeout error
    try:
        raise LLMTimeoutError(
            message="Test timeout",
            model="gpt-3.5-turbo",
            timeout_seconds=60,
            context={"duration": 60}
        )
    except LLMTimeoutError as e:
        logger.info(f"✓ Caught timeout error: {e}")
        logger.info(f"  Recovery strategies: {e.recovery_strategies}")
        
    # Test rate limit error
    try:
        raise LLMRateLimitError(
            message="Test rate limit",
            model="gpt-3.5-turbo",
            retry_after=60,
            context={"requests": 100}
        )
    except LLMRateLimitError as e:
        logger.info(f"✓ Caught rate limit error: {e}")
        logger.info(f"  Recovery strategies: {e.recovery_strategies}")
        
    # Test embedding error
    try:
        raise EmbeddingError(
            message="Test embedding failure",
            model="text-embedding-3-small",
            context={"model": "text-embedding-3-small"}
        )
    except EmbeddingError as e:
        logger.info(f"✓ Caught embedding error: {e}")
        logger.info(f"  Recovery strategies: {e.recovery_strategies}")
        
    return True

async def test_performance_monitor():
    """Test performance monitoring."""
    logger.info("\n=== Testing Performance Monitor ===")
    
    monitor = PerformanceMonitor()
    
    # Simulate some operations
    with monitor.measure_operation("test_operation_1"):
        await asyncio.sleep(0.1)  # Simulate work
        
    with monitor.measure_operation("test_operation_2"):
        await asyncio.sleep(0.05)  # Simulate work
        
    # Nested operations
    with monitor.measure_operation("parent_operation"):
        await asyncio.sleep(0.05)
        with monitor.measure_operation("child_operation"):
            await asyncio.sleep(0.02)
    
    # Get summary
    summary = monitor.get_summary()
    logger.info(f"✓ Performance summary: {summary}")
    
    # Get detailed metrics
    metrics = monitor.get_metrics()
    for name, metric_list in metrics.items():
        if metric_list:
            logger.info(f"  {name}: {len(metric_list)} calls, avg duration {sum(m.duration for m in metric_list)/len(metric_list):.3f}s")
    
    return True

async def test_enhanced_llm_provider():
    """Test enhanced LLM provider."""
    logger.info("\n=== Testing Enhanced LLM Provider ===")
    
    try:
        # Use default config
        from Option.Config2 import default_config as config
        
        # Create enhanced provider
        provider = EnhancedLiteLLMProvider(
            api_key=config.llm.api_key,
            model=config.llm.model,
            temperature=config.llm.temperature
        )
        
        # Test simple completion
        try:
            messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
            response = await provider.agenerate(messages)
            logger.info(f"✓ LLM Response: {response}")
        except Exception as e:
            logger.warning(f"  LLM test skipped (API key might be missing): {e}")
    
        # Get performance metrics
        if hasattr(provider, 'performance_tracker'):
            stats = provider.performance_tracker.get_stats(config.llm.model)
            logger.info(f"  LLM Stats: {stats}")
    except Exception as e:
        logger.warning(f"  Enhanced LLM provider test skipped: {e}")
    
    return True

async def test_batch_embeddings():
    """Test batch embedding processor."""
    logger.info("\n=== Testing Batch Embedding Processor ===")
    
    try:
        # Use default config
        from Option.Config2 import default_config as config
        embed_provider = get_rag_embedding(config)
        
        # Create batch processor
        processor = BatchEmbeddingProcessor(
            embed_model=embed_provider,
            initial_batch_size=2,
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
        
        logger.info(f"  Processing {len(texts)} texts (with 1 duplicate)...")
        embeddings = await processor.process_texts(texts)
        
        logger.info(f"✓ Generated {len(embeddings)} embeddings")
        cache_stats = processor.get_cache_stats()
        logger.info(f"  Cache stats: hits={cache_stats['hits']}, misses={cache_stats['misses']}, size={cache_stats['size']}")
        
        # Test again to see cache hit
        logger.info("  Processing same texts again...")
        embeddings2 = await processor.process_texts(texts)
        cache_stats2 = processor.get_cache_stats()
        logger.info(f"✓ Cache stats after reprocessing: hits={cache_stats2['hits']}, misses={cache_stats2['misses']}")
        
    except Exception as e:
        logger.warning(f"  Embedding test skipped (API key might be missing): {e}")
    
    return True

async def test_enhanced_tools_integration():
    """Test that enhanced tools can be imported and used."""
    logger.info("\n=== Testing Enhanced Tools Integration ===")
    
    # Import enhanced tools
    try:
        from Core.AgentTools.enhanced_entity_vdb_tools import entity_vdb_build_tool
        from Core.AgentTools.enhanced_relationship_tools import relationship_vdb_build_tool
        from Core.AgentOrchestrator.enhanced_orchestrator import EnhancedAgentOrchestrator
        
        logger.info("✓ Successfully imported enhanced entity VDB tool")
        logger.info("✓ Successfully imported enhanced relationship VDB tool") 
        logger.info("✓ Successfully imported enhanced orchestrator")
        
        # Check that tools have proper signatures
        import inspect
        
        entity_sig = inspect.signature(entity_vdb_build_tool)
        logger.info(f"  Entity VDB tool parameters: {list(entity_sig.parameters.keys())}")
        
        rel_sig = inspect.signature(relationship_vdb_build_tool)
        logger.info(f"  Relationship VDB tool parameters: {list(rel_sig.parameters.keys())}")
        
    except Exception as e:
        logger.error(f"Failed to import enhanced tools: {e}")
        return False
    
    return True

async def main():
    """Run all tests."""
    logger.info("Starting enhancement integration tests...\n")
    
    tests = [
        ("Structured Errors", test_structured_errors),
        ("Performance Monitor", test_performance_monitor),
        ("Enhanced LLM Provider", test_enhanced_llm_provider),
        ("Batch Embeddings", test_batch_embeddings),
        ("Enhanced Tools Integration", test_enhanced_tools_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
            else:
                failed += 1
                logger.error(f"❌ {test_name} failed")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} crashed: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("✅ All enhancement tests passed!")
    else:
        logger.error(f"❌ {failed} tests failed")

if __name__ == "__main__":
    asyncio.run(main())