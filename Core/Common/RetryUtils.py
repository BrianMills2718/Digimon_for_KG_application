"""
Retry utilities with exponential backoff for LLM calls and other operations.
"""

import asyncio
import functools
import random
import time
from typing import Any, Callable, Optional, Tuple, Type, Union

from .LoggerConfig import get_logger

logger = get_logger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: Tuple[Type[Exception], ...] = (Exception,),
        retry_condition: Optional[Callable[[Exception], bool]] = None
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            retry_on: Tuple of exception types to retry on
            retry_condition: Optional function to determine if exception should trigger retry
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on
        self.retry_condition = retry_condition
    
    def should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry."""
        if not isinstance(exception, self.retry_on):
            return False
        if self.retry_condition:
            return self.retry_condition(exception)
        return True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add jitter of ±25%
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            
        return max(0, delay)


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    retry_condition: Optional[Callable[[Exception], bool]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        retry_on: Tuple of exception types to retry on
        retry_condition: Optional function to determine if exception should trigger retry
        on_retry: Optional callback called before each retry with (exception, attempt)
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_on=retry_on,
        retry_condition=retry_condition
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    logger.debug(f"Attempting {func.__name__} (attempt {attempt}/{config.max_attempts})")
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    if not config.should_retry(e) or attempt == config.max_attempts:
                        logger.error(
                            f"Failed {func.__name__} after {attempt} attempts: {str(e)}",
                            extra={'function': func.__name__, 'attempt': attempt, 'error': str(e)}
                        )
                        raise
                    
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {str(e)}"
                    )
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(delay)
            
            raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    logger.debug(f"Attempting {func.__name__} (attempt {attempt}/{config.max_attempts})")
                    return await func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    if not config.should_retry(e) or attempt == config.max_attempts:
                        logger.error(
                            f"Failed {func.__name__} after {attempt} attempts: {str(e)}",
                            extra={'function': func.__name__, 'attempt': attempt, 'error': str(e)}
                        )
                        raise
                    
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {str(e)}"
                    )
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Specialized retry configurations for common use cases

def retry_llm_call(
    max_attempts: int = 3,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Retry decorator specifically for LLM API calls.
    Handles common LLM API errors with appropriate backoff.
    """
    def is_retryable_llm_error(e: Exception) -> bool:
        """Check if LLM error is retryable."""
        error_msg = str(e).lower()
        
        # Rate limit errors
        if any(msg in error_msg for msg in ['rate limit', 'too many requests', '429']):
            return True
        
        # Temporary errors
        if any(msg in error_msg for msg in ['timeout', 'connection', '500', '502', '503', '504']):
            return True
        
        # OpenAI specific
        if 'openai' in error_msg and any(msg in error_msg for msg in ['temporary', 'try again']):
            return True
            
        return False
    
    return retry(
        max_attempts=max_attempts,
        initial_delay=2.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True,
        retry_condition=is_retryable_llm_error,
        on_retry=on_retry
    )


def retry_db_operation(
    max_attempts: int = 3,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Retry decorator for database operations.
    Handles transient database errors.
    """
    def is_retryable_db_error(e: Exception) -> bool:
        """Check if database error is retryable."""
        error_msg = str(e).lower()
        
        # Common transient DB errors
        if any(msg in error_msg for msg in ['lock', 'timeout', 'connection', 'busy']):
            return True
            
        return False
    
    return retry(
        max_attempts=max_attempts,
        initial_delay=0.5,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=True,
        retry_condition=is_retryable_db_error,
        on_retry=on_retry
    )