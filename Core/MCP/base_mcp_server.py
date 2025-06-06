"""
Basic MCP (Model Context Protocol) Server Implementation

This module provides the foundation for MCP communication in DIGIMON.
It implements a WebSocket server that handles MCP protocol messages.
"""

import asyncio
import json
import logging
import time
import websockets
from typing import Dict, Any, Optional, Set
import websockets.server
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCPServer:
    """Basic MCP Server implementation with WebSocket support"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.server.WebSocketServerProtocol] = set()
        self.server = None
        self.start_time = None
        self.tools: Dict[str, Any] = {}  # Tool registry
        self._register_default_tools()
        
    async def start(self):
        """Start the MCP server"""
        self.start_time = datetime.now()
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info(f"MCP Server started on port {self.port}")
        print(f"MCP Server started on port {self.port}")
        
    async def stop(self):
        """Stop the MCP server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("MCP Server stopped")
            
    async def handle_client(self, websocket):
        """Handle a client connection"""
        # Register client
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Client connected from {client_addr}")
        
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_addr} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_addr}: {e}")
        finally:
            # Unregister client
            self.clients.discard(websocket)
            
    async def process_message(self, websocket, message: str):
        """Process an incoming message"""
        start_time = time.time()
        
        try:
            # Parse JSON message
            request = json.loads(message)
            logger.debug(f"Received request: {request}")
            
            # Extract request components
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            # Route to appropriate handler
            if method == "echo":
                result = await self.handle_echo(params)
                response = {
                    "id": request_id,
                    "status": "success",
                    "result": result
                }
            elif method == "list_tools":
                result = await self.handle_list_tools(params)
                response = {
                    "id": request_id,
                    "status": "success",
                    "result": result
                }
            elif method == "invoke_tool":
                result = await self.handle_invoke_tool(params)
                response = {
                    "id": request_id,
                    "status": result.get("status", "success"),
                    "result": result.get("result"),
                    "error": result.get("error"),
                    "metadata": result.get("metadata")
                }
            elif method in self.tools:
                # Direct tool invocation
                result = await self.handle_invoke_tool({"tool_name": method, "params": params})
                response = {
                    "id": request_id,
                    "status": result.get("status", "success"),
                    "result": result.get("result"),
                    "error": result.get("error"),
                    "metadata": result.get("metadata")
                }
            else:
                # Method not found
                response = {
                    "id": request_id,
                    "status": "error",
                    "error": f"Method not found: {method}",
                    "code": "METHOD_NOT_FOUND"
                }
                
            # Send response
            await websocket.send(json.dumps(response))
            
            # Log performance
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Request processed in {elapsed_ms:.1f}ms")
            
        except json.JSONDecodeError as e:
            # Invalid JSON
            error_response = {
                "status": "error",
                "error": f"Invalid JSON: {str(e)}",
                "code": "INVALID_JSON"
            }
            await websocket.send(json.dumps(error_response))
        except Exception as e:
            # General error
            logger.error(f"Error processing message: {e}")
            error_response = {
                "status": "error",
                "error": f"Internal server error: {str(e)}",
                "code": "INTERNAL_ERROR"
            }
            await websocket.send(json.dumps(error_response))
            
    async def handle_echo(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle echo request - for testing"""
        message = params.get("message", "")
        return {"echo": message}
        
    def _register_default_tools(self):
        """Register default tools"""
        # Register echo tool for testing
        self.register_tool({
            "name": "echo",
            "description": "Echo test tool",
            "handler": self.handle_echo,
            "input_schema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                }
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "echo": {"type": "string"}
                }
            }
        })
        
    def register_tool(self, tool_definition: Dict[str, Any]):
        """Register a tool with the server"""
        tool_name = tool_definition["name"]
        self.tools[tool_name] = tool_definition
        logger.info(f"Registered tool: {tool_name}")
        
    async def handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list tools request"""
        # Return all registered tools
        tools_list = []
        for tool_name, tool_def in self.tools.items():
            tools_list.append({
                "name": tool_name,
                "description": tool_def.get("description", ""),
                "input_schema": tool_def.get("input_schema", {}),
                "output_schema": tool_def.get("output_schema", {})
            })
        return {"tools": tools_list}
        
    async def handle_invoke_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool invocation"""
        tool_name = params.get("tool_name")
        tool_params = params.get("params", {})
        session_id = params.get("session_id")
        
        if not tool_name:
            return {
                "status": "error",
                "error": "tool_name is required"
            }
            
        if tool_name not in self.tools:
            return {
                "status": "error",
                "error": f"Tool '{tool_name}' not found"
            }
            
        tool_def = self.tools[tool_name]
        handler = tool_def.get("handler")
        
        if not handler:
            return {
                "status": "error",
                "error": f"Tool '{tool_name}' has no handler"
            }
            
        try:
            # If handler is an MCP tool instance, use its execute method
            if hasattr(handler, 'execute'):
                result = await handler.execute(tool_params, session_id)
                return result
            else:
                # Otherwise, call the handler directly
                result = await handler(tool_params)
                return {
                    "status": "success",
                    "result": result
                }
        except Exception as e:
            logger.error(f"Error invoking tool '{tool_name}': {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
        
    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        return {
            "running": self.server is not None,
            "host": self.host,
            "port": self.port,
            "clients": len(self.clients),
            "uptime": str(datetime.now() - self.start_time) if self.start_time else None
        }


async def start_server(host: str = "localhost", port: int = 8765):
    """Start the MCP server and keep it running"""
    server = MCPServer(host, port)
    await server.start()
    
    # Keep server running
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        await server.stop()


def main():
    """Main entry point for running the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DIGIMON MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run server
    asyncio.run(start_server(args.host, args.port))


if __name__ == "__main__":
    main()