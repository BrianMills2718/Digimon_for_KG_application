"""
Core MCP Server implementation for DIGIMON
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import uuid

from .shared_context import SharedContextStore

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """MCP Request format"""
    id: str
    tool_name: str
    params: Dict[str, Any]
    context: Dict[str, Any]
    session_id: str
    timestamp: datetime
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'MCPRequest':
        """Create request from JSON data"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            tool_name=data['tool_name'],
            params=data.get('params', {}),
            context=data.get('context', {}),
            session_id=data.get('session_id', 'default'),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat()))
        )
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            'id': self.id,
            'tool_name': self.tool_name,
            'params': self.params,
            'context': self.context,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class MCPResponse:
    """MCP Response format"""
    id: str
    request_id: str
    status: str  # 'success', 'error', 'partial'
    result: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    
    @property
    def is_error(self) -> bool:
        return self.status == 'error'
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            'id': self.id,
            'request_id': self.request_id,
            'status': self.status,
            'result': self.result,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class MCPError(Exception):
    """MCP-specific error"""
    def __init__(self, message: str, code: str = "MCP_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class MCPTool:
    """Wrapper for tools exposed via MCP"""
    def __init__(self, name: str, handler: Callable, schema: Optional[Dict] = None):
        self.name = name
        self.handler = handler
        self.schema = schema or {}
        
    async def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute the tool with given parameters"""
        try:
            # Check if handler expects context and if there's a naming collision
            handler_params = self.handler.__code__.co_varnames
            
            if 'context' in handler_params and 'context' not in params:
                # No collision, pass context as keyword arg
                return await self.handler(**params, context=context)
            elif 'context' in handler_params and 'context' in params:
                # Collision! Pass MCP context as mcp_context
                return await self.handler(**params, mcp_context=context)
            else:
                # Handler doesn't expect context
                return await self.handler(**params)
        except Exception as e:
            logger.error(f"Tool execution failed: {self.name}", exc_info=True)
            raise MCPError(f"Tool execution failed: {str(e)}", "TOOL_ERROR")


class DigimonMCPServer:
    """
    Core MCP Server for DIGIMON
    Handles tool registration, request routing, and context management
    """
    
    def __init__(self, server_name: str, capabilities: List[str], port: int = 8765):
        self.name = server_name
        self.capabilities = capabilities
        self.port = port
        self.tools: Dict[str, MCPTool] = {}
        self.context_store = SharedContextStore()
        self.server = None
        self.clients = set()
        
        # Performance metrics
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        
    def register_tool(self, tool: MCPTool):
        """Register a tool with the server"""
        logger.info(f"Registering tool: {tool.name}")
        self.tools[tool.name] = tool
        
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle incoming MCP request
        Routes to appropriate tool and manages context
        """
        start_time = datetime.utcnow()
        self.request_count += 1
        
        try:
            # Validate tool exists
            if request.tool_name not in self.tools:
                raise MCPError(f"Tool not found: {request.tool_name}", "TOOL_NOT_FOUND")
            
            tool = self.tools[request.tool_name]
            
            # Get session context
            context = await self.context_store.get_context(request.session_id)
            merged_context = {**context, **request.context}
            
            # Execute tool
            logger.debug(f"Executing tool: {request.tool_name}")
            result = await tool.execute(request.params, merged_context)
            
            # Update context with result
            await self.context_store.update_context(
                request.session_id, 
                {'last_tool': request.tool_name, 'last_result': result}
            )
            
            # Calculate latency
            latency = (datetime.utcnow() - start_time).total_seconds()
            self.total_latency += latency
            
            return MCPResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                status='success',
                result=result,
                metadata={
                    'latency_ms': int(latency * 1000),
                    'server': self.name
                },
                timestamp=datetime.utcnow()
            )
            
        except MCPError as e:
            self.error_count += 1
            return MCPResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                status='error',
                result={'error': str(e), 'code': e.code, 'details': e.details},
                metadata={'server': self.name},
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Unexpected error handling request", exc_info=True)
            return MCPResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                status='error',
                result={'error': str(e), 'code': 'INTERNAL_ERROR'},
                metadata={'server': self.name},
                timestamp=datetime.utcnow()
            )
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client connection"""
        addr = writer.get_extra_info('peername')
        logger.info(f"New client connected: {addr}")
        self.clients.add(writer)
        
        try:
            while True:
                # Read request
                data = await reader.read(65536)
                if not data:
                    break
                    
                try:
                    # Parse request
                    request_data = json.loads(data.decode())
                    request = MCPRequest.from_json(request_data)
                    
                    # Handle request
                    response = await self.handle_request(request)
                    
                    # Send response
                    response_data = json.dumps(response.to_json()).encode()
                    writer.write(len(response_data).to_bytes(4, 'big'))
                    writer.write(response_data)
                    await writer.drain()
                    
                except json.JSONDecodeError:
                    error_response = MCPResponse(
                        id=str(uuid.uuid4()),
                        request_id='unknown',
                        status='error',
                        result={'error': 'Invalid JSON', 'code': 'PARSE_ERROR'},
                        metadata={'server': self.name},
                        timestamp=datetime.utcnow()
                    )
                    response_data = json.dumps(error_response.to_json()).encode()
                    writer.write(len(response_data).to_bytes(4, 'big'))
                    writer.write(response_data)
                    await writer.drain()
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Client handler error: {e}", exc_info=True)
        finally:
            self.clients.discard(writer)
            writer.close()
            await writer.wait_closed()
            logger.info(f"Client disconnected: {addr}")
    
    async def start(self):
        """Start the MCP server"""
        self.server = await asyncio.start_server(
            self.handle_client, '127.0.0.1', self.port
        )
        
        addr = self.server.sockets[0].getsockname()
        logger.info(f"MCP Server '{self.name}' started on {addr}")
        
        async with self.server:
            await self.server.serve_forever()
    
    async def stop(self):
        """Stop the MCP server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        # Close all client connections
        for writer in self.clients:
            writer.close()
            await writer.wait_closed()
            
        logger.info(f"MCP Server '{self.name}' stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server performance metrics"""
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        return {
            'server_name': self.name,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / self.request_count if self.request_count > 0 else 0,
            'avg_latency_ms': int(avg_latency * 1000),
            'connected_clients': len(self.clients),
            'registered_tools': len(self.tools)
        }