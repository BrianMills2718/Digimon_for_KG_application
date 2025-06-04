"""
Tests for Phase 1, Checkpoint 1.2: Enhanced MCP Client Manager
"""

import asyncio
import pytest
from datetime import datetime, timedelta
import time

from Core.MCP import MCPServerInfo
from Core.MCP.mcp_client_enhanced import (
    EnhancedMCPClientManager, ConnectionState, ConnectionMetrics,
    ServerHealth, RequestCache, LoadBalancer, EnhancedMCPConnection
)


@pytest.fixture
async def enhanced_client():
    """Create test enhanced client manager"""
    client = EnhancedMCPClientManager(
        cache_ttl=60,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=30,
        load_balance_strategy="round_robin"
    )
    yield client
    await client.close()


class TestConnectionMetrics:
    """Test connection metrics tracking"""
    
    def test_metrics_initialization(self):
        """Test: Metrics initialize correctly"""
        metrics = ConnectionMetrics()
        assert metrics.total_requests == 0
        assert metrics.success_rate == 1.0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.consecutive_failures == 0
    
    def test_success_recording(self):
        """Test: Success metrics recorded correctly"""
        metrics = ConnectionMetrics()
        
        # Record successes
        metrics.record_success(10.5)
        metrics.record_success(15.3)
        metrics.record_success(12.1)
        
        assert metrics.total_requests == 3
        assert metrics.successful_requests == 3
        assert metrics.failed_requests == 0
        assert metrics.success_rate == 1.0
        assert abs(metrics.avg_latency_ms - 12.63) < 0.1
        assert metrics.consecutive_failures == 0
    
    def test_failure_recording(self):
        """Test: Failure metrics recorded correctly"""
        metrics = ConnectionMetrics()
        
        # Mix of success and failures
        metrics.record_success(10.0)
        metrics.record_failure("Connection timeout")
        metrics.record_failure("Server error")
        
        assert metrics.total_requests == 3
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 2
        assert metrics.success_rate == 1/3
        assert metrics.consecutive_failures == 2
        assert metrics.last_error == "Server error"
        assert metrics.last_error_time is not None
    
    def test_consecutive_failures_reset(self):
        """Test: Consecutive failures reset on success"""
        metrics = ConnectionMetrics()
        
        metrics.record_failure("Error 1")
        metrics.record_failure("Error 2")
        assert metrics.consecutive_failures == 2
        
        metrics.record_success(10.0)
        assert metrics.consecutive_failures == 0


class TestServerHealth:
    """Test server health tracking"""
    
    def test_health_initialization(self):
        """Test: Server health initializes correctly"""
        health = ServerHealth(
            server_name="test-server",
            state=ConnectionState.UNKNOWN,
            metrics=ConnectionMetrics(),
            last_health_check=datetime.utcnow()
        )
        
        assert health.server_name == "test-server"
        assert health.state == ConnectionState.UNKNOWN
        assert health.is_available() is True
        assert health.circuit_breaker_open is False
    
    def test_circuit_breaker(self):
        """Test: Circuit breaker functionality"""
        health = ServerHealth(
            server_name="test-server",
            state=ConnectionState.HEALTHY,
            metrics=ConnectionMetrics(),
            last_health_check=datetime.utcnow()
        )
        
        # Open circuit breaker
        health.circuit_breaker_open = True
        health.circuit_breaker_until = datetime.utcnow() + timedelta(seconds=30)
        
        assert health.is_available() is False
        
        # Simulate timeout expiry
        health.circuit_breaker_until = datetime.utcnow() - timedelta(seconds=1)
        assert health.is_available() is True
        assert health.circuit_breaker_open is False


class TestRequestCache:
    """Test request caching"""
    
    async def test_cache_operations(self):
        """Test: Cache get/set operations"""
        cache = RequestCache(ttl_seconds=60)
        
        # Cache miss
        result = await cache.get("test.tool", {"param": "value"})
        assert result is None
        
        # Cache set
        await cache.set("test.tool", {"param": "value"}, "cached_result")
        
        # Cache hit
        result = await cache.get("test.tool", {"param": "value"})
        assert result == "cached_result"
        
        # Different params = cache miss
        result = await cache.get("test.tool", {"param": "different"})
        assert result is None
    
    async def test_cache_expiry(self):
        """Test: Cache entries expire correctly"""
        cache = RequestCache(ttl_seconds=1)
        
        await cache.set("test.tool", {"param": "value"}, "result")
        assert await cache.get("test.tool", {"param": "value"}) == "result"
        
        # Wait for expiry
        await asyncio.sleep(1.1)
        assert await cache.get("test.tool", {"param": "value"}) is None
    
    async def test_cache_key_generation(self):
        """Test: Cache keys are deterministic"""
        cache = RequestCache()
        
        # Same params in different order should produce same key
        key1 = cache._make_key("tool", {"b": 2, "a": 1})
        key2 = cache._make_key("tool", {"a": 1, "b": 2})
        assert key1 == key2
        
        # Different tools should have different keys
        key3 = cache._make_key("other_tool", {"a": 1, "b": 2})
        assert key1 != key3


class TestLoadBalancer:
    """Test load balancing strategies"""
    
    def test_round_robin(self):
        """Test: Round-robin load balancing"""
        lb = LoadBalancer(strategy="round_robin")
        servers = ["server1", "server2", "server3"]
        
        # Mock health - all healthy
        health = {
            s: ServerHealth(s, ConnectionState.HEALTHY, ConnectionMetrics(), datetime.utcnow())
            for s in servers
        }
        
        # Should cycle through servers
        selections = []
        for _ in range(6):
            selected = lb.select_server(servers, "test.tool", health)
            selections.append(selected)
        
        assert selections == ["server1", "server2", "server3", "server1", "server2", "server3"]
    
    def test_unhealthy_server_skipped(self):
        """Test: Unhealthy servers are skipped"""
        lb = LoadBalancer(strategy="round_robin")
        servers = ["server1", "server2", "server3"]
        
        # server2 is unhealthy
        health = {
            "server1": ServerHealth("server1", ConnectionState.HEALTHY, ConnectionMetrics(), datetime.utcnow()),
            "server2": ServerHealth("server2", ConnectionState.UNHEALTHY, ConnectionMetrics(), datetime.utcnow()),
            "server3": ServerHealth("server3", ConnectionState.HEALTHY, ConnectionMetrics(), datetime.utcnow())
        }
        
        # Should skip server2
        selections = []
        for _ in range(4):
            selected = lb.select_server(servers, "test.tool", health)
            selections.append(selected)
        
        assert selections == ["server1", "server3", "server1", "server3"]
        assert "server2" not in selections
    
    def test_least_latency(self):
        """Test: Least latency selection"""
        lb = LoadBalancer(strategy="least_latency")
        servers = ["server1", "server2", "server3"]
        
        # Create health with different latencies
        health = {}
        for i, server in enumerate(servers):
            metrics = ConnectionMetrics()
            # server1: 20ms, server2: 10ms, server3: 30ms
            latencies = [20, 10, 30]
            metrics.record_success(latencies[i])
            health[server] = ServerHealth(server, ConnectionState.HEALTHY, metrics, datetime.utcnow())
        
        # Should always select server2 (lowest latency)
        for _ in range(3):
            selected = lb.select_server(servers, "test.tool", health)
            assert selected == "server2"


class TestEnhancedMCPClientManager:
    """Test enhanced client manager functionality"""
    
    async def test_multi_server_registration(self, enhanced_client):
        """Test: Multiple servers can register same tool"""
        server1 = MCPServerInfo("server1", "127.0.0.1", 8001, ["tool.shared", "tool.unique1"])
        server2 = MCPServerInfo("server2", "127.0.0.1", 8002, ["tool.shared", "tool.unique2"])
        
        enhanced_client.register_server(server1)
        enhanced_client.register_server(server2)
        
        # Check tool registry
        assert "tool.shared" in enhanced_client.tool_registry
        assert len(enhanced_client.tool_registry["tool.shared"]) == 2
        assert set(enhanced_client.tool_registry["tool.shared"]) == {"server1", "server2"}
        
        assert enhanced_client.tool_registry["tool.unique1"] == ["server1"]
        assert enhanced_client.tool_registry["tool.unique2"] == ["server2"]
    
    async def test_circuit_breaker_trigger(self, enhanced_client):
        """Test: Circuit breaker triggers after threshold failures"""
        server = MCPServerInfo("failing-server", "127.0.0.1", 9999, ["fail.tool"])
        enhanced_client.register_server(server)
        
        # Simulate failures
        health = enhanced_client.server_health["failing-server"]
        for i in range(enhanced_client.circuit_breaker_threshold):
            health.metrics.record_failure(f"Failure {i+1}")
        
        # Monitor should open circuit breaker
        await enhanced_client._monitor_server_health()
        
        assert health.circuit_breaker_open is True
        assert health.state == ConnectionState.UNHEALTHY
        assert health.is_available() is False
    
    async def test_retry_with_backoff(self, enhanced_client):
        """Test: Exponential backoff on retries"""
        delays = []
        for attempt in range(4):
            delay = enhanced_client._get_retry_delay(attempt)
            delays.append(delay)
        
        # Each delay should be roughly double the previous (with jitter)
        assert delays[0] >= 0.5 and delays[0] <= 1.5  # ~1s
        assert delays[1] >= 1.0 and delays[1] <= 3.0  # ~2s
        assert delays[2] >= 2.0 and delays[2] <= 6.0  # ~4s
        assert delays[3] >= 4.0 and delays[3] <= 12.0  # ~8s
    
    async def test_health_report(self, enhanced_client):
        """Test: Health report generation"""
        server = MCPServerInfo("test-server", "127.0.0.1", 8001, ["test.tool"])
        enhanced_client.register_server(server)
        
        # Simulate some activity
        health = enhanced_client.server_health["test-server"]
        health.metrics.record_success(15.0)
        health.metrics.record_success(20.0)
        health.metrics.record_failure("Test error")
        
        report = enhanced_client.get_health_report()
        
        assert "servers" in report
        assert "test-server" in report["servers"]
        
        server_report = report["servers"]["test-server"]
        assert server_report["status"] == "unknown"
        assert "66.67%" in server_report["metrics"]["success_rate"]
        assert server_report["metrics"]["avg_latency_ms"] == "17.50"
        assert server_report["metrics"]["total_requests"] == 3
        assert server_report["circuit_breaker"]["open"] is False
        
        assert report["cache_stats"]["ttl_seconds"] == 60
        assert report["load_balancer"]["strategy"] == "round_robin"


@pytest.mark.asyncio
async def test_caching_integration():
    """Test: End-to-end caching behavior"""
    client = EnhancedMCPClientManager(cache_ttl=60)
    
    # Mock successful tool result
    async def mock_invoke(tool_name, params, context, session_id, bypass_cache=False):
        # Simulate work
        await asyncio.sleep(0.1)
        return f"Result for {tool_name}"
    
    # Temporarily replace invoke method
    original_invoke = client.invoke_tool
    client.invoke_tool = mock_invoke
    
    try:
        # First call - should take time
        start = time.time()
        result1 = await mock_invoke("test.tool", {"param": "value"}, {}, "session1")
        duration1 = time.time() - start
        
        # Store in cache manually
        await client.cache.set("test.tool", {"param": "value"}, result1)
        
        # Second call - should be cached
        start = time.time()
        cached = await client.cache.get("test.tool", {"param": "value"})
        duration2 = time.time() - start
        
        # Cache hit should be much faster
        assert duration2 < duration1 / 10
        assert cached == result1
        
        # Different params - cache miss
        start = time.time()
        cached_miss = await client.cache.get("test.tool", {"param": "different"})
        result3 = await mock_invoke("test.tool", {"param": "different"}, {}, "session1")
        duration3 = time.time() - start
        
        # Should take time again for the mock call
        assert cached_miss is None
        assert duration3 > duration2 * 5
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_caching_integration())