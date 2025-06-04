"""
Enhanced MCP Client implementation with advanced features
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
from enum import Enum
import hashlib

from .mcp_server import MCPRequest, MCPResponse
from .mcp_client import MCPServerInfo, MCPConnection, AsyncConnectionPool

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ConnectionMetrics:
    """Metrics for a single connection"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    consecutive_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests
    
    def record_success(self, latency_ms: float):
        """Record successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.consecutive_failures = 0
    
    def record_failure(self, error: str):
        """Record failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_error = error
        self.last_error_time = datetime.utcnow()


@dataclass
class ServerHealth:
    """Health status for an MCP server"""
    server_name: str
    state: ConnectionState
    metrics: ConnectionMetrics
    last_health_check: datetime
    circuit_breaker_open: bool = False
    circuit_breaker_until: Optional[datetime] = None
    
    def is_available(self) -> bool:
        """Check if server is available for requests"""
        if self.circuit_breaker_open:
            if self.circuit_breaker_until and datetime.utcnow() > self.circuit_breaker_until:
                # Circuit breaker timeout expired, allow retry
                self.circuit_breaker_open = False
                return True
            return False
        return self.state != ConnectionState.UNHEALTHY


class RequestCache:
    """Simple request cache with TTL"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()
    
    def _make_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Create cache key from tool and params"""
        key_data = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available and not expired"""
        async with self._lock:
            key = self._make_key(tool_name, params)
            if key in self.cache:
                result, timestamp = self.cache[key]
                if datetime.utcnow() - timestamp < self.ttl:
                    logger.debug(f"Cache hit for {tool_name}")
                    return result
                else:
                    # Expired, remove it
                    del self.cache[key]
        return None
    
    async def set(self, tool_name: str, params: Dict[str, Any], result: Any):
        """Cache a result"""
        async with self._lock:
            key = self._make_key(tool_name, params)
            self.cache[key] = (result, datetime.utcnow())
            
            # Simple cleanup - remove expired entries
            now = datetime.utcnow()
            expired_keys = [
                k for k, (_, ts) in self.cache.items()
                if now - ts > self.ttl
            ]
            for k in expired_keys:
                del self.cache[k]


class EnhancedMCPConnection(MCPConnection):
    """Enhanced connection with metrics and circuit breaker"""
    
    def __init__(self, server_info: MCPServerInfo):
        super().__init__(server_info)
        self.metrics = ConnectionMetrics()
        self.health_check_interval = timedelta(seconds=30)
        self.last_health_check = datetime.utcnow()
    
    async def send_request_with_metrics(self, request: MCPRequest) -> MCPResponse:
        """Send request and record metrics"""
        start_time = time.time()
        
        try:
            response = await self.send_request(request)
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_success(latency_ms)
            return response
        except Exception as e:
            self.metrics.record_failure(str(e))
            raise
    
    def needs_health_check(self) -> bool:
        """Check if connection needs health check"""
        return datetime.utcnow() - self.last_health_check > self.health_check_interval


class LoadBalancer:
    """Load balancing strategy for multiple servers with same capabilities"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.current_index: Dict[str, int] = {}
    
    def select_server(self, servers: List[str], tool_name: str, 
                     server_health: Dict[str, ServerHealth]) -> Optional[str]:
        """Select best server for request"""
        # Filter available servers
        available = [
            s for s in servers 
            if s in server_health and server_health[s].is_available()
        ]
        
        if not available:
            return None
        
        if self.strategy == "round_robin":
            # Simple round-robin
            if tool_name not in self.current_index:
                self.current_index[tool_name] = 0
            
            index = self.current_index[tool_name] % len(available)
            self.current_index[tool_name] = (index + 1) % len(available)
            return available[index]
            
        elif self.strategy == "least_latency":
            # Select server with lowest average latency
            return min(available, key=lambda s: server_health[s].metrics.avg_latency_ms)
            
        elif self.strategy == "least_connections":
            # Select server with fewest active requests
            # For now, approximate with total requests
            return min(available, key=lambda s: server_health[s].metrics.total_requests)
        
        # Default to first available
        return available[0]


class EnhancedMCPClientManager:
    """
    Enhanced MCP Client Manager with:
    - Connection health monitoring
    - Circuit breaker pattern
    - Request caching
    - Load balancing
    - Automatic retries with exponential backoff
    """
    
    def __init__(self, 
                 cache_ttl: int = 300,
                 circuit_breaker_threshold: int = 5,
                 circuit_breaker_timeout: int = 60,
                 load_balance_strategy: str = "round_robin"):
        # Original attributes
        self.servers: Dict[str, MCPServerInfo] = {}
        self.connection_pool = AsyncConnectionPool()
        self.tool_registry: Dict[str, List[str]] = {}  # tool_name -> [server_names]
        
        # Enhanced attributes
        self.cache = RequestCache(ttl_seconds=cache_ttl)
        self.server_health: Dict[str, ServerHealth] = {}
        self.load_balancer = LoadBalancer(strategy=load_balance_strategy)
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = timedelta(seconds=circuit_breaker_timeout)
        
        # Retry configuration with exponential backoff
        self.max_retries = 3
        self.base_retry_delay = 1.0
        self.max_retry_delay = 30.0
        
        # Background tasks
        self._health_monitor_task = None
        self._start_health_monitor()
    
    def _start_health_monitor(self):
        """Start background health monitoring"""
        async def monitor_loop():
            while True:
                try:
                    await asyncio.sleep(30)  # Check every 30 seconds
                    await self._monitor_server_health()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health monitor error: {e}", exc_info=True)
        
        try:
            self._health_monitor_task = asyncio.create_task(monitor_loop())
        except RuntimeError:
            logger.debug("No event loop for health monitor")
    
    async def _monitor_server_health(self):
        """Monitor health of all servers"""
        for server_name, server_info in self.servers.items():
            if server_name not in self.server_health:
                self.server_health[server_name] = ServerHealth(
                    server_name=server_name,
                    state=ConnectionState.UNKNOWN,
                    metrics=ConnectionMetrics(),
                    last_health_check=datetime.utcnow()
                )
            
            health = self.server_health[server_name]
            
            # Update state based on metrics
            if health.metrics.consecutive_failures >= self.circuit_breaker_threshold:
                # Open circuit breaker
                health.circuit_breaker_open = True
                health.circuit_breaker_until = datetime.utcnow() + self.circuit_breaker_timeout
                health.state = ConnectionState.UNHEALTHY
                logger.warning(f"Circuit breaker opened for {server_name}")
            elif health.metrics.success_rate < 0.5:
                health.state = ConnectionState.DEGRADED
            elif health.metrics.success_rate > 0.9:
                health.state = ConnectionState.HEALTHY
            
            health.last_health_check = datetime.utcnow()
    
    def register_server(self, server_info: MCPServerInfo):
        """Register an MCP server with multi-server support per tool"""
        logger.info(f"Registering MCP server: {server_info.name}")
        self.servers[server_info.name] = server_info
        
        # Update tool registry - support multiple servers per tool
        for capability in server_info.capabilities:
            if capability not in self.tool_registry:
                self.tool_registry[capability] = []
            if server_info.name not in self.tool_registry[capability]:
                self.tool_registry[capability].append(server_info.name)
        
        # Initialize health tracking
        self.server_health[server_info.name] = ServerHealth(
            server_name=server_info.name,
            state=ConnectionState.UNKNOWN,
            metrics=ConnectionMetrics(),
            last_health_check=datetime.utcnow()
        )
    
    async def invoke_tool(self, tool_name: str, params: Dict[str, Any],
                         context: Dict[str, Any], session_id: str = "default",
                         bypass_cache: bool = False) -> Any:
        """
        Enhanced tool invocation with caching, load balancing, and circuit breaker
        """
        # Check cache first
        if not bypass_cache:
            cached_result = await self.cache.get(tool_name, params)
            if cached_result is not None:
                return cached_result
        
        # Get available servers for this tool
        server_names = self.tool_registry.get(tool_name, [])
        if not server_names:
            raise ValueError(f"No servers registered for tool: {tool_name}")
        
        # Try servers with load balancing and circuit breaker
        last_error = None
        for retry_attempt in range(self.max_retries):
            # Select server using load balancer
            selected_server = self.load_balancer.select_server(
                server_names, tool_name, self.server_health
            )
            
            if not selected_server:
                # All servers unavailable
                await asyncio.sleep(self._get_retry_delay(retry_attempt))
                continue
            
            server_info = self.servers[selected_server]
            
            try:
                # Create request
                request = MCPRequest(
                    id=str(uuid.uuid4()),
                    tool_name=tool_name,
                    params=params,
                    context=context,
                    session_id=session_id,
                    timestamp=datetime.utcnow()
                )
                
                # Get connection
                conn = await self.connection_pool.get_connection(server_info)
                if isinstance(conn, EnhancedMCPConnection):
                    enhanced_conn = conn
                else:
                    # Wrap in enhanced connection
                    enhanced_conn = EnhancedMCPConnection(server_info)
                    enhanced_conn.reader = conn.reader
                    enhanced_conn.writer = conn.writer
                    enhanced_conn.connected = conn.connected
                
                try:
                    # Send request with metrics
                    response = await enhanced_conn.send_request_with_metrics(request)
                    
                    if response.status == 'success':
                        # Cache successful result
                        await self.cache.set(tool_name, params, response.result)
                        return response.result
                    elif response.status == 'error':
                        raise Exception(f"Tool error: {response.result}")
                    else:
                        raise Exception(f"Unexpected response status: {response.status}")
                        
                finally:
                    await self.connection_pool.return_connection(enhanced_conn)
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {retry_attempt + 1} failed for {tool_name} on {selected_server}: {e}")
                
                # Update server health
                if selected_server in self.server_health:
                    self.server_health[selected_server].metrics.record_failure(str(e))
                
                if retry_attempt < self.max_retries - 1:
                    await asyncio.sleep(self._get_retry_delay(retry_attempt))
        
        raise Exception(f"Failed to invoke tool {tool_name} after {self.max_retries} attempts: {last_error}")
    
    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff"""
        delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
        # Add jitter
        import random
        return delay * (0.5 + random.random() * 0.5)
    
    async def invoke_parallel(self, tool_calls: List[Dict[str, Any]],
                            context: Dict[str, Any], session_id: str = "default") -> List[Any]:
        """
        Enhanced parallel invocation with proper error handling
        """
        tasks = []
        for call in tool_calls:
            task = self.invoke_tool(
                call['tool_name'],
                call.get('params', {}),
                context,
                session_id,
                call.get('bypass_cache', False)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        return {
            "servers": {
                name: {
                    "status": health.state.value,
                    "metrics": {
                        "success_rate": f"{health.metrics.success_rate:.2%}",
                        "avg_latency_ms": f"{health.metrics.avg_latency_ms:.2f}",
                        "total_requests": health.metrics.total_requests,
                        "consecutive_failures": health.metrics.consecutive_failures
                    },
                    "circuit_breaker": {
                        "open": health.circuit_breaker_open,
                        "until": health.circuit_breaker_until.isoformat() if health.circuit_breaker_until else None
                    },
                    "last_error": health.metrics.last_error,
                    "last_health_check": health.last_health_check.isoformat()
                }
                for name, health in self.server_health.items()
            },
            "cache_stats": {
                "entries": len(self.cache.cache),
                "ttl_seconds": self.cache.ttl.total_seconds()
            },
            "load_balancer": {
                "strategy": self.load_balancer.strategy
            }
        }
    
    async def close(self):
        """Clean up resources"""
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        
        await self.connection_pool.close_all()