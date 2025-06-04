"""
MCP Client implementation for connecting to MCP servers
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import uuid

from .mcp_server import MCPRequest, MCPResponse

logger = logging.getLogger(__name__)


@dataclass
class MCPServerInfo:
    """Information about an available MCP server"""
    name: str
    host: str
    port: int
    capabilities: List[str]
    status: str = 'unknown'  # 'online', 'offline', 'unknown'
    last_check: Optional[datetime] = None


class MCPConnection:
    """Single connection to an MCP server"""
    
    def __init__(self, server_info: MCPServerInfo):
        self.server_info = server_info
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False
        self._lock = asyncio.Lock()
        
    async def connect(self):
        """Establish connection to MCP server"""
        if self.connected:
            return
            
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.server_info.host, self.server_info.port
            )
            self.connected = True
            self.server_info.status = 'online'
            self.server_info.last_check = datetime.utcnow()
            logger.info(f"Connected to MCP server: {self.server_info.name}")
        except Exception as e:
            self.server_info.status = 'offline'
            self.server_info.last_check = datetime.utcnow()
            logger.error(f"Failed to connect to {self.server_info.name}: {e}")
            raise
    
    async def disconnect(self):
        """Close connection to MCP server"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.connected = False
        logger.info(f"Disconnected from MCP server: {self.server_info.name}")
    
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """Send request to MCP server and wait for response"""
        async with self._lock:
            if not self.connected:
                await self.connect()
            
            try:
                # Send request
                request_data = json.dumps(request.to_json()).encode()
                self.writer.write(request_data)
                await self.writer.drain()
                
                # Read response length
                length_data = await self.reader.readexactly(4)
                response_length = int.from_bytes(length_data, 'big')
                
                # Read response
                response_data = await self.reader.readexactly(response_length)
                response_dict = json.loads(response_data.decode())
                
                return MCPResponse(
                    id=response_dict['id'],
                    request_id=response_dict['request_id'],
                    status=response_dict['status'],
                    result=response_dict['result'],
                    metadata=response_dict['metadata'],
                    timestamp=datetime.fromisoformat(response_dict['timestamp'])
                )
                
            except Exception as e:
                logger.error(f"Request failed to {self.server_info.name}: {e}")
                self.connected = False
                raise


class AsyncConnectionPool:
    """Connection pool for managing multiple MCP connections"""
    
    def __init__(self, max_connections_per_server: int = 5):
        self.max_connections_per_server = max_connections_per_server
        self.pools: Dict[str, List[MCPConnection]] = {}
        self.available: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
    
    async def get_connection(self, server_info: MCPServerInfo) -> MCPConnection:
        """Get an available connection from the pool"""
        server_name = server_info.name
        
        async with self._lock:
            if server_name not in self.pools:
                self.pools[server_name] = []
                self.available[server_name] = asyncio.Queue(maxsize=self.max_connections_per_server)
        
        # Try to get available connection
        try:
            conn = self.available[server_name].get_nowait()
            if conn.connected:
                return conn
        except asyncio.QueueEmpty:
            pass
        
        # Create new connection if under limit
        async with self._lock:
            if len(self.pools[server_name]) < self.max_connections_per_server:
                conn = MCPConnection(server_info)
                await conn.connect()
                self.pools[server_name].append(conn)
                return conn
        
        # Wait for available connection
        conn = await self.available[server_name].get()
        if not conn.connected:
            await conn.connect()
        return conn
    
    async def return_connection(self, conn: MCPConnection):
        """Return connection to the pool"""
        server_name = conn.server_info.name
        if server_name in self.available:
            await self.available[server_name].put(conn)
    
    async def close_all(self):
        """Close all connections in the pool"""
        for server_name, connections in self.pools.items():
            for conn in connections:
                await conn.disconnect()
        self.pools.clear()
        self.available.clear()


class MCPClientManager:
    """
    Manager for MCP client connections
    Handles server discovery, connection pooling, and request routing
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPServerInfo] = {}
        self.connection_pool = AsyncConnectionPool()
        self.tool_registry: Dict[str, str] = {}  # tool_name -> server_name
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        
    def register_server(self, server_info: MCPServerInfo):
        """Register an MCP server"""
        logger.info(f"Registering MCP server: {server_info.name}")
        self.servers[server_info.name] = server_info
        
        # Update tool registry
        for capability in server_info.capabilities:
            self.tool_registry[capability] = server_info.name
    
    async def discover_servers(self, discovery_endpoints: Optional[List[str]] = None):
        """
        Discover available MCP servers
        In production, this would query service discovery endpoints
        """
        # Default local servers for development
        if not discovery_endpoints:
            default_servers = [
                MCPServerInfo(
                    name="graphrag-tools",
                    host="127.0.0.1",
                    port=8765,
                    capabilities=[
                        "Entity.VDBSearch",
                        "Entity.PPR",
                        "Graph.Build",
                        "Chunk.FromRelationships"
                    ]
                ),
                MCPServerInfo(
                    name="synthesis-tools",
                    host="127.0.0.1",
                    port=8766,
                    capabilities=[
                        "Chunk.Retrieve",
                        "Answer.Generate",
                        "Community.Summarize"
                    ]
                )
            ]
            
            for server in default_servers:
                self.register_server(server)
        else:
            # TODO: Implement actual service discovery
            pass
        
        # Test connections to all servers
        await self.health_check_all()
    
    async def health_check_all(self):
        """Check health of all registered servers"""
        tasks = []
        for server_info in self.servers.values():
            tasks.append(self._health_check_server(server_info))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        online_count = sum(1 for r in results if r is True)
        logger.info(f"Health check complete: {online_count}/{len(self.servers)} servers online")
    
    async def _health_check_server(self, server_info: MCPServerInfo) -> bool:
        """Health check a single server"""
        try:
            conn = MCPConnection(server_info)
            await conn.connect()
            await conn.disconnect()
            return True
        except Exception:
            return False
    
    def route_to_server(self, tool_name: str) -> Optional[str]:
        """Route tool request to appropriate server"""
        return self.tool_registry.get(tool_name)
    
    async def invoke_tool(self, tool_name: str, params: Dict[str, Any], 
                         context: Dict[str, Any], session_id: str = "default") -> Any:
        """
        Invoke a tool on the appropriate MCP server
        Handles routing, retries, and connection management
        """
        server_name = self.route_to_server(tool_name)
        if not server_name:
            raise ValueError(f"No server found for tool: {tool_name}")
        
        server_info = self.servers.get(server_name)
        if not server_info:
            raise ValueError(f"Server not found: {server_name}")
        
        request = MCPRequest(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            params=params,
            context=context,
            session_id=session_id,
            timestamp=datetime.utcnow()
        )
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                conn = await self.connection_pool.get_connection(server_info)
                try:
                    response = await conn.send_request(request)
                    
                    if response.status == 'success':
                        return response.result
                    elif response.status == 'error':
                        raise Exception(f"Tool error: {response.result}")
                    else:
                        raise Exception(f"Unexpected response status: {response.status}")
                        
                finally:
                    await self.connection_pool.return_connection(conn)
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for {tool_name}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        raise Exception(f"Failed to invoke tool {tool_name} after {self.max_retries} attempts: {last_error}")
    
    async def invoke_parallel(self, tool_calls: List[Dict[str, Any]], 
                            context: Dict[str, Any], session_id: str = "default") -> List[Any]:
        """
        Invoke multiple tools in parallel
        Each tool_call should have 'tool_name' and 'params'
        """
        tasks = []
        for call in tool_calls:
            task = self.invoke_tool(
                call['tool_name'],
                call.get('params', {}),
                context,
                session_id
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_available_tools(self) -> List[str]:
        """Get list of all available tools across all servers"""
        return list(self.tool_registry.keys())
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get status of all registered servers"""
        return {
            server_name: {
                'host': server.host,
                'port': server.port,
                'status': server.status,
                'last_check': server.last_check.isoformat() if server.last_check else None,
                'capabilities': server.capabilities
            }
            for server_name, server in self.servers.items()
        }
    
    async def close(self):
        """Clean up all connections"""
        await self.connection_pool.close_all()