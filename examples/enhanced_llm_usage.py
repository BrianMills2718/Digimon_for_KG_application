"""
Example of using the Enhanced LLM Provider with performance monitoring.
"""

import asyncio
from Config.LLMConfig import LLMConfig, LLMType
from Option.Config2 import Config
from Core.Provider.EnhancedLiteLLMProvider import EnhancedLiteLLMProvider
from Core.Common.Logger import logger


async def demo_enhanced_llm():
    """Demonstrate enhanced LLM features."""
    
    # Load configuration
    config = Config()
    
    # Create enhanced LLM config
    llm_config = LLMConfig(
        api_type=LLMType.LITELLM,
        model="openai/gpt-3.5-turbo",  # or any LiteLLM supported model
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        temperature=0.0,  # Deterministic for caching
        max_token=2000,
        timeout=60,
        calc_usage=True,
        # Enhanced features
        enable_cache=True,
        enable_adaptive_timeout=True,
        enable_performance_tracking=True
    )
    
    # Create provider
    provider = EnhancedLiteLLMProvider(llm_config)
    
    # Example 1: Simple completion with caching
    logger.info("=== Example 1: Simple completion ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # First call - will be cached
    start = asyncio.get_event_loop().time()
    response1 = await provider.acompletion(messages)
    duration1 = asyncio.get_event_loop().time() - start
    logger.info(f"First call took {duration1:.2f}s: {provider.get_choice_text(response1)}")
    
    # Second call - should be from cache
    start = asyncio.get_event_loop().time()
    response2 = await provider.acompletion(messages)
    duration2 = asyncio.get_event_loop().time() - start
    logger.info(f"Second call took {duration2:.2f}s (cached): {provider.get_choice_text(response2)}")
    
    # Example 2: Long response with adaptive timeout
    logger.info("\n=== Example 2: Long response with adaptive timeout ===")
    long_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a detailed 500-word essay about artificial intelligence."}
    ]
    
    try:
        response3 = await provider.acompletion(long_messages, max_tokens=1000)
        logger.info(f"Long response completed successfully")
    except Exception as e:
        logger.error(f"Long response failed: {e}")
        # The error will include recovery strategies
    
    # Example 3: Streaming with performance tracking
    logger.info("\n=== Example 3: Streaming response ===")
    streaming_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 10 slowly."}
    ]
    
    logger.info("Streaming response:")
    full_response = ""
    async for chunk in provider._achat_completion_stream(streaming_messages):
        print(chunk, end="", flush=True)
        full_response += chunk
    print()  # New line after streaming
    
    # Example 4: Performance summary
    logger.info("\n=== Performance Summary ===")
    summary = await provider.get_performance_summary()
    
    logger.info("Performance Metrics:")
    logger.info(f"  Total requests: {summary['performance'].get('total_requests', 0)}")
    logger.info(f"  Success rate: {summary['performance'].get('success_rate', 0):.1%}")
    logger.info(f"  Avg duration: {summary['performance'].get('avg_duration_seconds', 0):.2f}s")
    
    logger.info("\nCache Statistics:")
    logger.info(f"  Hit rate: {summary['cache']['hit_rate']:.1%}")
    logger.info(f"  Cache size: {summary['cache']['size']}/{summary['cache']['max_size']}")
    
    logger.info("\nAdaptive Timeout:")
    logger.info(f"  Performance multiplier: {summary['adaptive_timeout']['performance_multiplier']:.2f}")
    
    # Example 5: Optimize settings based on performance
    logger.info("\n=== Optimization Suggestions ===")
    await provider.optimize_settings()
    
    # Example 6: Error handling with recovery
    logger.info("\n=== Error Handling Example ===")
    error_messages = [
        {"role": "user", "content": "Test" * 10000}  # Very long message
    ]
    
    try:
        # This might timeout or hit token limits
        response = await provider.acompletion(error_messages, timeout=5)
    except Exception as e:
        logger.error(f"Expected error occurred: {type(e).__name__}")
        # The structured error will include recovery strategies


async def demo_performance_comparison():
    """Compare performance with and without enhancements."""
    from Core.Provider.LiteLLMProvider import LiteLLMProvider
    
    config = Config()
    base_config = LLMConfig(
        api_type=LLMType.LITELLM,
        model="openai/gpt-3.5-turbo",
        api_key=config.llm.api_key,
        temperature=0.0,
        max_token=1000
    )
    
    # Create both providers
    basic_provider = LiteLLMProvider(base_config)
    enhanced_provider = EnhancedLiteLLMProvider(base_config)
    
    # Test messages
    test_messages = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Explain quantum physics in one sentence."}],
        [{"role": "user", "content": "List 5 programming languages."}],
    ]
    
    logger.info("=== Performance Comparison ===")
    
    # Test basic provider
    logger.info("\nBasic Provider:")
    basic_start = asyncio.get_event_loop().time()
    for msgs in test_messages:
        await basic_provider.acompletion(msgs)
    basic_duration = asyncio.get_event_loop().time() - basic_start
    logger.info(f"Total time: {basic_duration:.2f}s")
    
    # Test enhanced provider (first run - no cache)
    logger.info("\nEnhanced Provider (first run):")
    enhanced_start = asyncio.get_event_loop().time()
    for msgs in test_messages:
        await enhanced_provider.acompletion(msgs)
    enhanced_duration1 = asyncio.get_event_loop().time() - enhanced_start
    logger.info(f"Total time: {enhanced_duration1:.2f}s")
    
    # Test enhanced provider (second run - with cache)
    logger.info("\nEnhanced Provider (cached):")
    enhanced_start = asyncio.get_event_loop().time()
    for msgs in test_messages:
        await enhanced_provider.acompletion(msgs)
    enhanced_duration2 = asyncio.get_event_loop().time() - enhanced_start
    logger.info(f"Total time: {enhanced_duration2:.2f}s")
    
    # Show improvement
    cache_speedup = (enhanced_duration1 - enhanced_duration2) / enhanced_duration1 * 100
    logger.info(f"\nCache speedup: {cache_speedup:.1f}%")


if __name__ == "__main__":
    # Run the demos
    asyncio.run(demo_enhanced_llm())
    # asyncio.run(demo_performance_comparison())