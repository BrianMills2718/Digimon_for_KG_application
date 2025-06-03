"""
Unit tests for retry utilities.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch
import time

from Core.Common.RetryUtils import RetryConfig, retry, retry_llm_call


class TestRetryConfig:
    """Test RetryConfig class."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retry_on == (Exception,)
        assert config.retry_condition is None
    
    def test_should_retry_with_matching_exception(self):
        """Test should_retry with matching exception type."""
        config = RetryConfig(retry_on=(ValueError, TypeError))
        assert config.should_retry(ValueError("test")) is True
        assert config.should_retry(TypeError("test")) is True
        assert config.should_retry(RuntimeError("test")) is False
    
    def test_should_retry_with_condition(self):
        """Test should_retry with custom condition."""
        config = RetryConfig(
            retry_condition=lambda e: "retry" in str(e)
        )
        assert config.should_retry(Exception("please retry")) is True
        assert config.should_retry(Exception("do not")) is False
    
    def test_get_delay_exponential_backoff(self):
        """Test delay calculation with exponential backoff."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False
        )
        
        assert config.get_delay(1) == 1.0  # 1 * 2^0
        assert config.get_delay(2) == 2.0  # 1 * 2^1
        assert config.get_delay(3) == 4.0  # 1 * 2^2
        assert config.get_delay(4) == 8.0  # 1 * 2^3
        assert config.get_delay(5) == 10.0  # capped at max_delay
    
    def test_get_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(
            initial_delay=10.0,
            jitter=True
        )
        
        # Run multiple times to test jitter randomness
        delays = [config.get_delay(1) for _ in range(10)]
        
        # All delays should be within ±25% of base delay
        assert all(7.5 <= d <= 12.5 for d in delays)
        # Delays should vary (very unlikely to be all the same)
        assert len(set(delays)) > 1


class TestRetryDecorator:
    """Test retry decorator functionality."""
    
    def test_sync_retry_success_first_attempt(self):
        """Test sync function succeeds on first attempt."""
        mock_func = Mock(return_value="success")
        
        @retry(max_attempts=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_sync_retry_success_after_failures(self):
        """Test sync function succeeds after failures."""
        mock_func = Mock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])
        
        @retry(max_attempts=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_sync_retry_exhausted(self):
        """Test sync function exhausts retries."""
        mock_func = Mock(side_effect=ValueError("always fails"))
        
        @retry(max_attempts=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with pytest.raises(ValueError, match="always fails"):
            test_func()
        
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_retry_success_first_attempt(self):
        """Test async function succeeds on first attempt."""
        mock_func = Mock(return_value="success")
        
        @retry(max_attempts=3)
        async def test_func():
            return mock_func()
        
        result = await test_func()
        assert result == "success"
        assert mock_func.call_count == 1
    
    @pytest.mark.asyncio
    async def test_async_retry_success_after_failures(self):
        """Test async function succeeds after failures."""
        mock_func = Mock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])
        
        @retry(max_attempts=3, initial_delay=0.01)
        async def test_func():
            return mock_func()
        
        result = await test_func()
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_retry_with_specific_exceptions(self):
        """Test retry only on specific exceptions."""
        mock_func = Mock(side_effect=[ValueError("retry this"), RuntimeError("don't retry")])
        
        @retry(max_attempts=3, retry_on=(ValueError,), initial_delay=0.01)
        def test_func():
            return mock_func()
        
        # Should not retry RuntimeError
        with pytest.raises(RuntimeError, match="don't retry"):
            test_func()
        
        assert mock_func.call_count == 2
    
    def test_retry_with_callback(self):
        """Test retry with on_retry callback."""
        callback_calls = []
        
        def on_retry_callback(exception, attempt):
            callback_calls.append((str(exception), attempt))
        
        mock_func = Mock(side_effect=[ValueError("fail1"), ValueError("fail2"), "success"])
        
        @retry(max_attempts=3, initial_delay=0.01, on_retry=on_retry_callback)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert len(callback_calls) == 2
        assert callback_calls[0] == ("fail1", 1)
        assert callback_calls[1] == ("fail2", 2)


class TestLLMRetry:
    """Test LLM-specific retry functionality."""
    
    def test_retry_llm_call_rate_limit(self):
        """Test LLM retry handles rate limit errors."""
        mock_func = Mock(side_effect=[
            Exception("Rate limit exceeded"),
            Exception("429 Too Many Requests"),
            "success"
        ])
        
        @retry_llm_call(max_attempts=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_retry_llm_call_timeout(self):
        """Test LLM retry handles timeout errors."""
        mock_func = Mock(side_effect=[
            Exception("Request timeout"),
            Exception("Connection error"),
            "success"
        ])
        
        @retry_llm_call(max_attempts=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_retry_llm_call_non_retryable(self):
        """Test LLM retry doesn't retry non-retryable errors."""
        mock_func = Mock(side_effect=Exception("Invalid API key"))
        
        @retry_llm_call(max_attempts=3)
        def test_func():
            return mock_func()
        
        with pytest.raises(Exception, match="Invalid API key"):
            test_func()
        
        # Should only try once for non-retryable errors
        assert mock_func.call_count == 1
    
    @pytest.mark.asyncio
    async def test_async_retry_llm_call(self):
        """Test async LLM retry functionality."""
        mock_func = Mock(side_effect=[
            Exception("500 Internal Server Error"),
            Exception("503 Service Unavailable"),
            "success"
        ])
        
        @retry_llm_call(max_attempts=3)
        async def test_func():
            return mock_func()
        
        result = await test_func()
        assert result == "success"
        assert mock_func.call_count == 3