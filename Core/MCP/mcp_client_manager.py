"""
MCP Client Manager Implementation

This module provides client-side functionality for connecting to MCP servers,
managing connection pools, and invoking remote methods.
"""

import asyncio
import json
import logging
import time
import websockets
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for connection pooling"""
    connections_created: int = 0
    connections_reused: int = 0
    total_requests: int = 0
    
    @property
    def reuse_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.connections_reused / self.total_requests


class MCPConnection:
    """Represents a connection to an MCP server"""
    
    def __init__(self, websocket, server_url: str):
        self.websocket = websocket
        self.server_url = server_url
        self.in_use = False
        self.last_used = time.time()
        
    @property
    def is_open(self) -> bool:
        """Check if connection is still open"""
        return self.websocket.state.name == 'OPEN'
        
    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request and receive response"""
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        return json.loads(response)


class MCPClientManager:
    """Manages MCP client connections with pooling support"""
    
    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size
        self.connections: Dict[str, List[MCPConnection]] = defaultdict(list)
        self.stats = ConnectionStats()
        self.connection_state = "disconnected"
        self.server_url = None
        
    async def connect(self, host: str, port: int):
        """Connect to MCP server"""
        self.server_url = f"ws://{host}:{port}"
        self.connection_state = "connected"
        logger.info(f"Connected to MCP server at {host}:{port}")
        
    async def get_connection(self) -> MCPConnection:
        """Get a connection from the pool or create new one"""
        self.stats.total_requests += 1
        
        # Try to find available connection
        for conn in self.connections[self.server_url]:
            if not conn.in_use and conn.is_open:
                conn.in_use = True
                conn.last_used = time.time()
                self.stats.connections_reused += 1
                logger.debug("Reusing existing connection")
                return conn
                
        # Create new connection if under pool limit
        if len(self.connections[self.server_url]) < self.pool_size:
            websocket = await websockets.connect(self.server_url)
            conn = MCPConnection(websocket, self.server_url)
            conn.in_use = True
            self.connections[self.server_url].append(conn)
            self.stats.connections_created += 1
            logger.debug("Created new connection")
            return conn
            
        # Wait for available connection
        while True:
            for conn in self.connections[self.server_url]:
                if not conn.in_use and conn.is_open:
                    conn.in_use = True
                    conn.last_used = time.time()
                    self.stats.connections_reused += 1
                    return conn
            await asyncio.sleep(0.01)
            
    async def release_connection(self, conn: MCPConnection):
        """Release connection back to pool"""
        conn.in_use = False
        
    async def invoke_method(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a method on the MCP server"""
        start_time = time.time()
        
        request = {
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params
        }
        
        # Get connection from pool
        conn = await self.get_connection()
        
        try:
            # Send request and get response
            response = await conn.send_request(request)
            
            # Log performance
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Method {method} completed in {elapsed_ms:.1f}ms")
            
            return response
            
        finally:
            # Always release connection back to pool
            await self.release_connection(conn)
            
    async def invoke_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool on the MCP server"""
        return await self.invoke_method(tool_name, params)
        
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from server"""
        response = await self.invoke_method("list_tools", {})
        return response.get("tools", [])
        
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        active_connections = sum(
            1 for conns in self.connections.values() 
            for conn in conns if conn.is_open
        )
        
        return {
            "pool_size": self.pool_size,
            "connections_created": self.stats.connections_created,
            "connections_reused": self.stats.connections_reused,
            "total_requests": self.stats.total_requests,
            "reuse_rate": self.stats.reuse_rate,
            "active_connections": active_connections
        }
        
    async def close(self):
        """Close all connections"""
        for server_conns in self.connections.values():
            for conn in server_conns:
                if conn.is_open:
                    await conn.websocket.close()
        self.connections.clear()
        self.connection_state = "disconnected"
        logger.info("All connections closed")