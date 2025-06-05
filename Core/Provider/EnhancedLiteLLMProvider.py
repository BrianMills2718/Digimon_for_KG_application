"""
Enhanced LiteLLM Provider with performance monitoring, adaptive timeouts, and structured error handling.
"""

import os
import json
import asyncio
import time
from typing import List, Dict, Any, Optional, Type, AsyncGenerator

from pydantic import BaseModel
import litellm
litellm.drop_params = True
import instructor

from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Common.Logger import logger
from Core.Common.LLMEnhancements import (
    get_llm_performance_tracker,
    get_adaptive_timeout,
    get_llm_cache,
    enhanced_llm_call
)
from Core.Common.StructuredErrors import (
    LLMTimeoutError,
    LLMRateLimitError,
    LLMError,
    ErrorHandler,
    ErrorContext,
    ErrorSeverity
)
from Core.Utils.TokenCounter import count_input_tokens, count_output_tokens
from Core.Provider.LLMProviderRegister import register_provider
from Config.LLMConfig import LLMType


@register_provider(LLMType.LITELLM)
class EnhancedLiteLLMProvider(LiteLLMProvider):
    """Enhanced LiteLLM provider with advanced features."""
    
    def __init__(self, config):
        super().__init__(config)
        self.error_handler = ErrorHandler()
        self.performance_tracker = get_llm_performance_tracker()
        self.adaptive_timeout = get_adaptive_timeout()
        self.cache = get_llm_cache()
        
        # Performance settings
        self.enable_cache = getattr(config, 'enable_cache', True)
        self.enable_adaptive_timeout = getattr(config, 'enable_adaptive_timeout', True)
        self.enable_performance_tracking = getattr(config, 'enable_performance_tracking', True)
        
        logger.info(
            f"EnhancedLiteLLMProvider initialized with features: "
            f"cache={self.enable_cache}, adaptive_timeout={self.enable_adaptive_timeout}, "
            f"performance_tracking={self.enable_performance_tracking}"
        )
    
    def _estimate_tokens(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> int:
        """Estimate total tokens for a request."""
        prompt_tokens = count_input_tokens(messages, self.pricing_plan)
        completion_tokens = max_tokens or self.config.max_token // 2  # Rough estimate
        return prompt_tokens + completion_tokens
    
    async def _handle_llm_error(self, e: Exception, context: Dict[str, Any]) -> Any:
        """Convert exceptions to structured errors and handle them."""
        error_msg = str(e).lower()
        
        # Determine error type and create structured error
        if "timeout" in error_msg:
            structured_error = LLMTimeoutError(
                message=str(e),
                model=self.model,
                timeout_seconds=context.get("timeout", self.config.timeout),
                estimated_tokens=context.get("estimated_tokens"),
                context=ErrorContext(
                    operation="llm_completion",
                    component="EnhancedLiteLLMProvider",
                    metadata=context
                )
            )
        elif "rate limit" in error_msg or "429" in error_msg:
            retry_after = None
            # Try to extract retry-after from error message
            import re
            match = re.search(r'retry after (\d+)', error_msg)
            if match:
                retry_after = int(match.group(1))
                
            structured_error = LLMRateLimitError(
                message=str(e),
                model=self.model,
                retry_after=retry_after,
                context=ErrorContext(
                    operation="llm_completion",
                    component="EnhancedLiteLLMProvider",
                    metadata=context
                )
            )
        else:
            structured_error = LLMError(
                message=str(e),
                context=ErrorContext(
                    operation="llm_completion",
                    component="EnhancedLiteLLMProvider",
                    metadata=context
                ),
                cause=e
            )
        
        # Handle the error
        recovery = self.error_handler.handle_error(
            structured_error,
            logger=logger,
            auto_recover=True
        )
        
        # If recovery suggested increasing timeout, return that info
        if recovery and recovery.action == "increase_timeout":
            context["suggested_timeout"] = recovery.params["new_timeout"]
        
        raise structured_error
    
    async def _achat_completion(
        self, 
        messages: list[dict], 
        timeout: Optional[int] = None, 
        max_tokens: Optional[int] = None, 
        **kwargs
    ) -> litellm.ModelResponse:
        """Enhanced completion with caching and adaptive timeout."""
        estimated_tokens = 1000  # Default estimate
        try:
            # Estimate tokens for timeout calculation
            estimated_tokens = self._estimate_tokens(messages, max_tokens)
            
            # Get adaptive timeout if enabled
            if self.enable_adaptive_timeout and timeout is None:
                timeout = await self.adaptive_timeout.get_timeout(
                    estimated_tokens,
                    self.model,
                    is_streaming=False
                )
            
            # Check cache if enabled
            if self.enable_cache and self.temperature == 0:
                cached_response = await self.cache.get(
                    messages,
                    self.model,
                    self.temperature
                )
                if cached_response:
                    logger.debug("Returning cached LLM response")
                    return cached_response
            
            # Track start time
            start_time = time.time()
            
            # Make the actual call
            response = await super()._achat_completion(
                messages,
                timeout=timeout,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Track performance
            duration = time.time() - start_time
            if self.enable_performance_tracking:
                await self.performance_tracker.track_request(
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else estimated_tokens // 2,
                    completion_tokens=response.usage.completion_tokens if response.usage else estimated_tokens // 2,
                    duration=duration,
                    success=True
                )
                
                # Update adaptive timeout performance
                if self.enable_adaptive_timeout:
                    total_tokens = (
                        response.usage.total_tokens if response.usage 
                        else estimated_tokens
                    )
                    await self.adaptive_timeout.update_performance(duration, total_tokens)
            
            # Cache successful response
            if self.enable_cache and self.temperature == 0:
                await self.cache.set(
                    messages,
                    self.model,
                    self.temperature,
                    response
                )
            
            return response
            
        except Exception as e:
            # Track failed request
            if self.enable_performance_tracking:
                # Use estimated_tokens if defined, otherwise use a default
                prompt_tokens = estimated_tokens // 2 if 'estimated_tokens' in locals() else 1000
                await self.performance_tracker.track_request(
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    duration=time.time() - start_time if 'start_time' in locals() else 0,
                    success=False,
                    error=str(e)
                )
            
            # Handle error with structured error handling
            await self._handle_llm_error(e, {
                "timeout": timeout,
                "max_tokens": max_tokens,
                "estimated_tokens": estimated_tokens,
                "messages_count": len(messages)
            })
    
    async def _achat_completion_stream(
        self,
        messages: list[dict],
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Enhanced streaming completion with adaptive timeout."""
        try:
            # Estimate tokens for timeout calculation
            estimated_tokens = self._estimate_tokens(messages, max_tokens)
            
            # Get adaptive timeout if enabled
            if self.enable_adaptive_timeout and timeout is None:
                timeout = await self.adaptive_timeout.get_timeout(
                    estimated_tokens,
                    self.model,
                    is_streaming=True
                )
            
            # Track start time
            start_time = time.time()
            collected_tokens = 0
            
            # Make the actual streaming call
            async for chunk in super()._achat_completion_stream(
                messages,
                timeout=timeout,
                max_tokens=max_tokens,
                **kwargs
            ):
                collected_tokens += len(chunk) // 4  # Rough token estimate
                yield chunk
            
            # Track performance for successful stream
            duration = time.time() - start_time
            if self.enable_performance_tracking:
                await self.performance_tracker.track_request(
                    model=self.model,
                    prompt_tokens=estimated_tokens // 2,
                    completion_tokens=collected_tokens,
                    duration=duration,
                    success=True
                )
                
        except Exception as e:
            # Track failed request
            if self.enable_performance_tracking:
                # Use estimated_tokens if defined, otherwise use a default
                prompt_tokens = estimated_tokens // 2 if 'estimated_tokens' in locals() else 1000
                await self.performance_tracker.track_request(
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    duration=time.time() - start_time if 'start_time' in locals() else 0,
                    success=False,
                    error=str(e)
                )
            
            # Handle error
            await self._handle_llm_error(e, {
                "timeout": timeout,
                "max_tokens": max_tokens,
                "estimated_tokens": estimated_tokens,
                "streaming": True
            })
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this provider."""
        summary = await self.performance_tracker.get_performance_summary()
        cache_stats = await self.cache.get_stats()
        
        return {
            "model": self.model,
            "performance": summary,
            "cache": cache_stats,
            "adaptive_timeout": {
                "performance_multiplier": self.adaptive_timeout.performance_multiplier,
                "base_timeout": self.adaptive_timeout.base_timeout
            }
        }
    
    async def optimize_settings(self):
        """Optimize provider settings based on performance data."""
        summary = await self.performance_tracker.get_performance_summary()
        
        if summary.get("recommendations"):
            logger.info("Performance optimization recommendations:")
            for rec in summary["recommendations"]:
                logger.info(f"  - {rec}")
        
        # Auto-adjust settings based on performance
        if summary.get("avg_duration_seconds", 0) > 45:
            logger.warning(
                "Average response time is high. Consider enabling streaming "
                "or reducing context size."
            )
        
        cache_stats = await self.cache.get_stats()
        if cache_stats["hit_rate"] < 0.1 and self.temperature == 0:
            logger.info(
                f"Low cache hit rate ({cache_stats['hit_rate']:.1%}). "
                "Consider increasing cache TTL or size."
            )