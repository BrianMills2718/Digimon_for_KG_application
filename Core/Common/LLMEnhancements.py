"""
Enhanced LLM utilities for better performance and reliability.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta

from .LoggerConfig import get_logger
from .PerformanceMonitor import monitor_performance, get_performance_monitor
from .RetryUtils import retry_llm_call

logger = get_logger(__name__)


@dataclass
class LLMRequestStats:
    """Statistics for LLM requests."""
    model: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = False
    error: Optional[str] = None
    retry_count: int = 0


class LLMPerformanceTracker:
    """Tracks LLM performance metrics and provides optimization suggestions."""
    
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.requests: List[LLMRequestStats] = []
        self._lock = asyncio.Lock()
        
    async def track_request(
        self, 
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration: float,
        success: bool,
        error: Optional[str] = None,
        retry_count: int = 0
    ):
        """Track an LLM request."""
        stats = LLMRequestStats(
            model=model,
            start_time=time.time() - duration,
            end_time=time.time(),
            duration=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            success=success,
            error=error,
            retry_count=retry_count
        )
        
        async with self._lock:
            self.requests.append(stats)
            # Clean old requests outside window
            cutoff_time = time.time() - (self.window_minutes * 60)
            self.requests = [r for r in self.requests if r.start_time > cutoff_time]
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for optimization."""
        async with self._lock:
            if not self.requests:
                return {"status": "no_data"}
            
            total_requests = len(self.requests)
            successful_requests = sum(1 for r in self.requests if r.success)
            failed_requests = total_requests - successful_requests
            
            durations = [r.duration for r in self.requests if r.duration and r.success]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            retry_requests = sum(1 for r in self.requests if r.retry_count > 0)
            total_retries = sum(r.retry_count for r in self.requests)
            
            # Group by model
            model_stats = defaultdict(lambda: {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "avg_duration": 0,
                "total_tokens": 0,
                "errors": defaultdict(int)
            })
            
            for req in self.requests:
                model_stats[req.model]["requests"] += 1
                if req.success:
                    model_stats[req.model]["successes"] += 1
                else:
                    model_stats[req.model]["failures"] += 1
                    if req.error:
                        model_stats[req.model]["errors"][req.error] += 1
                model_stats[req.model]["total_tokens"] += req.total_tokens
            
            # Calculate averages
            for model, stats in model_stats.items():
                model_durations = [
                    r.duration for r in self.requests 
                    if r.model == model and r.duration and r.success
                ]
                if model_durations:
                    stats["avg_duration"] = sum(model_durations) / len(model_durations)
            
            return {
                "window_minutes": self.window_minutes,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                "avg_duration_seconds": avg_duration,
                "retry_requests": retry_requests,
                "total_retries": total_retries,
                "model_stats": dict(model_stats),
                "recommendations": self._generate_recommendations(
                    total_requests, failed_requests, avg_duration, model_stats
                )
            }
    
    def _generate_recommendations(
        self, 
        total_requests: int,
        failed_requests: int,
        avg_duration: float,
        model_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # High failure rate
        if total_requests > 0 and failed_requests / total_requests > 0.1:
            recommendations.append(
                f"High failure rate ({failed_requests}/{total_requests}). "
                "Consider increasing timeout or improving error handling."
            )
        
        # Slow response times
        if avg_duration > 30:
            recommendations.append(
                f"Average response time is {avg_duration:.1f}s. "
                "Consider using a faster model or implementing caching."
            )
        
        # Model-specific issues
        for model, stats in model_stats.items():
            if stats["failures"] > stats["successes"]:
                recommendations.append(
                    f"Model '{model}' has more failures than successes. "
                    "Consider switching to a more reliable model."
                )
                
            # Check for specific error patterns
            for error, count in stats["errors"].items():
                if "timeout" in error.lower() and count > 2:
                    recommendations.append(
                        f"Multiple timeout errors for '{model}'. "
                        "Increase timeout or use streaming for long responses."
                    )
                elif "rate limit" in error.lower() and count > 2:
                    recommendations.append(
                        f"Rate limiting detected for '{model}'. "
                        "Implement request throttling or increase API limits."
                    )
        
        return recommendations


class AdaptiveTimeout:
    """Adaptive timeout based on request characteristics and historical performance."""
    
    def __init__(
        self,
        base_timeout: int = 60,
        min_timeout: int = 30,
        max_timeout: int = 300,
        tokens_per_second: float = 50.0
    ):
        self.base_timeout = base_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.tokens_per_second = tokens_per_second
        self.performance_multiplier = 1.0
        self._lock = asyncio.Lock()
        
    async def get_timeout(
        self,
        estimated_tokens: int,
        model: str = "default",
        is_streaming: bool = False
    ) -> int:
        """Calculate adaptive timeout based on request characteristics."""
        # Base calculation from token count
        token_based_timeout = estimated_tokens / self.tokens_per_second
        
        # Add buffer for network and processing
        calculated_timeout = max(
            self.base_timeout,
            token_based_timeout * 1.5 * self.performance_multiplier
        )
        
        # Streaming requests may need more time
        if is_streaming:
            calculated_timeout *= 1.2
        
        # Clamp to min/max
        timeout = int(min(max(calculated_timeout, self.min_timeout), self.max_timeout))
        
        logger.debug(
            f"Adaptive timeout: {timeout}s for ~{estimated_tokens} tokens "
            f"(model: {model}, streaming: {is_streaming})"
        )
        
        return timeout
    
    async def update_performance(self, actual_duration: float, token_count: int):
        """Update performance multiplier based on actual performance."""
        if token_count <= 0:
            return
            
        actual_tokens_per_second = token_count / actual_duration
        performance_ratio = self.tokens_per_second / actual_tokens_per_second
        
        async with self._lock:
            # Exponential moving average
            self.performance_multiplier = (
                0.7 * self.performance_multiplier + 0.3 * performance_ratio
            )
            # Keep multiplier reasonable
            self.performance_multiplier = max(0.5, min(3.0, self.performance_multiplier))
            
        logger.debug(
            f"Updated performance multiplier to {self.performance_multiplier:.2f} "
            f"(actual: {actual_tokens_per_second:.1f} tokens/s)"
        )


class LLMCache:
    """Simple in-memory cache for LLM responses."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
        
    def _make_key(self, messages: List[Dict], model: str, temperature: float) -> str:
        """Create cache key from request parameters."""
        import hashlib
        import json
        
        # Normalize messages for consistent hashing
        normalized = json.dumps({
            "messages": messages,
            "model": model,
            "temperature": temperature
        }, sort_keys=True)
        
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    async def get(
        self, 
        messages: List[Dict], 
        model: str, 
        temperature: float
    ) -> Optional[Any]:
        """Get cached response if available."""
        if temperature > 0:  # Don't cache non-deterministic responses
            return None
            
        key = self._make_key(messages, model, temperature)
        
        async with self._lock:
            if key in self.cache:
                response, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    self.hits += 1
                    logger.debug(f"Cache hit (rate: {self.hits}/{self.hits + self.misses})")
                    return response
                else:
                    # Expired
                    del self.cache[key]
            
            self.misses += 1
            return None
    
    async def set(
        self, 
        messages: List[Dict], 
        model: str, 
        temperature: float,
        response: Any
    ):
        """Cache a response."""
        if temperature > 0:  # Don't cache non-deterministic responses
            return
            
        key = self._make_key(messages, model, temperature)
        
        async with self._lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (response, time.time())
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self.hits + self.misses
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total_requests if total_requests > 0 else 0,
                "ttl_seconds": self.ttl_seconds
            }


# Global instances
_llm_tracker = LLMPerformanceTracker()
_adaptive_timeout = AdaptiveTimeout()
_llm_cache = LLMCache()


def get_llm_performance_tracker() -> LLMPerformanceTracker:
    """Get global LLM performance tracker."""
    return _llm_tracker


def get_adaptive_timeout() -> AdaptiveTimeout:
    """Get global adaptive timeout calculator."""
    return _adaptive_timeout


def get_llm_cache() -> LLMCache:
    """Get global LLM cache."""
    return _llm_cache


@monitor_performance("llm_call")
async def enhanced_llm_call(
    llm_func: Callable,
    messages: List[Dict],
    model: str,
    temperature: float = 0.0,
    estimated_tokens: int = 1000,
    use_cache: bool = True,
    **kwargs
) -> Any:
    """
    Enhanced LLM call with caching, adaptive timeout, and performance tracking.
    
    Args:
        llm_func: The LLM function to call
        messages: Messages to send
        model: Model name
        temperature: Temperature setting
        estimated_tokens: Estimated total tokens for timeout calculation
        use_cache: Whether to use caching
        **kwargs: Additional arguments for llm_func
    """
    # Check cache first
    if use_cache:
        cached_response = await _llm_cache.get(messages, model, temperature)
        if cached_response is not None:
            return cached_response
    
    # Get adaptive timeout
    timeout = await _adaptive_timeout.get_timeout(
        estimated_tokens, 
        model, 
        kwargs.get("stream", False)
    )
    kwargs["timeout"] = timeout
    
    # Track performance
    start_time = time.time()
    success = False
    error = None
    retry_count = 0
    
    try:
        # Make the actual call
        response = await llm_func(messages, **kwargs)
        success = True
        
        # Cache successful response
        if use_cache and temperature == 0:
            await _llm_cache.set(messages, model, temperature, response)
        
        return response
        
    except Exception as e:
        error = str(e)
        if "retry" in error.lower():
            retry_count = kwargs.get("_retry_count", 0)
        raise
        
    finally:
        # Track request
        duration = time.time() - start_time
        await _llm_tracker.track_request(
            model=model,
            prompt_tokens=estimated_tokens // 2,  # Rough estimate
            completion_tokens=estimated_tokens // 2,
            duration=duration,
            success=success,
            error=error,
            retry_count=retry_count
        )
        
        # Update adaptive timeout performance
        if success:
            await _adaptive_timeout.update_performance(duration, estimated_tokens)