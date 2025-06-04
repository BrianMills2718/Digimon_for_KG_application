"""
Tests for Phase 1, Checkpoint 1.1: Core MCP Server Framework
"""

import asyncio
import pytest
import json
from datetime import datetime

from Core.MCP import (
    DigimonMCPServer, MCPRequest, MCPResponse, MCPError,
    MCPClientManager, SharedContextStore, MCPTool, MCPServerInfo
)


@pytest.fixture
async def mcp_server():
    """Create test MCP server"""
    server = DigimonMCPServer("test-server", ["test-capability"], port=9999)
    
    # Register test tool
    async def test_tool(input: str, context: dict = None):
        return f"Processed: {input}"
    
    server.register_tool(MCPTool("test.tool", test_tool))
    
    yield server
    
    # Cleanup
    await server.stop()


@pytest.fixture
async def context_store():
    """Create test context store"""
    store = SharedContextStore()
    yield store
    await store.close()


class TestMCPServer:
    """Test MCP Server functionality"""
    
    async def test_server_creation(self):
        """Test: Server starts and accepts connections"""
        server = DigimonMCPServer("test-server", ["capability1", "capability2"])
        assert server.name == "test-server"
        assert server.capabilities == ["capability1", "capability2"]
        assert len(server.tools) == 0
        assert server.request_count == 0
    
    async def test_tool_registration(self, mcp_server):
        """Test: Tools can be registered"""
        # Tool already registered in fixture
        assert "test.tool" in mcp_server.tools
        
        # Register another tool
        async def another_tool(x: int, y: int):
            return x + y
        
        mcp_server.register_tool(MCPTool("math.add", another_tool))
        assert "math.add" in mcp_server.tools
        assert len(mcp_server.tools) == 2
    
    async def test_request_response_cycle(self, mcp_server):
        """Test: Request/response cycle completes"""
        request = MCPRequest(
            id="test-123",
            tool_name="test.tool",
            params={"input": "hello"},
            context={},
            session_id="test-session",
            timestamp=datetime.utcnow()
        )
        
        response = await mcp_server.handle_request(request)
        
        assert response.status == "success"
        assert response.request_id == "test-123"
        assert response.result == "Processed: hello"
        assert mcp_server.request_count == 1
    
    async def test_error_handling(self, mcp_server):
        """Test: Errors handled gracefully"""
        # Test non-existent tool
        request = MCPRequest(
            id="test-456",
            tool_name="non.existent",
            params={},
            context={},
            session_id="test-session",
            timestamp=datetime.utcnow()
        )
        
        response = await mcp_server.handle_request(request)
        
        assert response.status == "error"
        assert response.request_id == "test-456"
        assert "Tool not found" in str(response.result)
        assert mcp_server.error_count == 1
    
    async def test_metrics(self, mcp_server):
        """Test: Server metrics tracked correctly"""
        # Make some requests
        for i in range(3):
            request = MCPRequest(
                id=f"test-{i}",
                tool_name="test.tool",
                params={"input": f"test{i}"},
                context={},
                session_id="test-session",
                timestamp=datetime.utcnow()
            )
            await mcp_server.handle_request(request)
        
        # Make an error request
        error_request = MCPRequest(
            id="error-1",
            tool_name="bad.tool",
            params={},
            context={},
            session_id="test-session",
            timestamp=datetime.utcnow()
        )
        await mcp_server.handle_request(error_request)
        
        metrics = mcp_server.get_metrics()
        assert metrics['request_count'] == 4
        assert metrics['error_count'] == 1
        assert metrics['error_rate'] == 0.25
        assert metrics['registered_tools'] == 1


class TestSharedContext:
    """Test shared context functionality"""
    
    async def test_context_persistence(self, context_store):
        """Test: Context persists between requests"""
        session_id = "test-session"
        
        # Set context
        await context_store.update_context(session_id, {"key1": "value1", "key2": 42})
        
        # Retrieve context
        context = await context_store.get_context(session_id)
        assert context["key1"] == "value1"
        assert context["key2"] == 42
    
    async def test_session_management(self, context_store):
        """Test: Sessions managed correctly"""
        # Create multiple sessions
        for i in range(3):
            await context_store.update_context(f"session-{i}", {"index": i})
        
        # Verify sessions exist
        stats = context_store.get_stats()
        assert stats['session_count'] == 3
        
        # Delete a session
        await context_store.delete_session("session-1")
        stats = context_store.get_stats()
        assert stats['session_count'] == 2
    
    async def test_value_operations(self, context_store):
        """Test: Individual value operations"""
        session_id = "test-session"
        
        # Set individual values
        await context_store.set_value(session_id, "name", "test")
        await context_store.set_value(session_id, "count", 10)
        
        # Get individual values
        name = await context_store.get_value(session_id, "name")
        count = await context_store.get_value(session_id, "count")
        missing = await context_store.get_value(session_id, "missing", "default")
        
        assert name == "test"
        assert count == 10
        assert missing == "default"
    
    async def test_export_import(self, context_store):
        """Test: Session export/import"""
        # Create sessions
        await context_store.update_context("session-1", {"data": "test1"})
        await context_store.update_context("session-2", {"data": "test2"})
        
        # Export
        exported = await context_store.export_sessions()
        assert len(exported) == 2
        
        # Clear and import
        await context_store.delete_session("session-1")
        await context_store.delete_session("session-2")
        
        await context_store.import_sessions(exported)
        
        # Verify imported data
        context1 = await context_store.get_context("session-1")
        assert context1["data"] == "test1"


class TestConnectionPooling:
    """Test connection pooling functionality"""
    
    async def test_client_manager_creation(self):
        """Test: Client manager initialized correctly"""
        manager = MCPClientManager()
        assert len(manager.servers) == 0
        assert len(manager.tool_registry) == 0
    
    async def test_server_registration(self):
        """Test: Servers can be registered"""
        manager = MCPClientManager()
        server_info = MCPServerInfo(
            name="test-server",
            host="127.0.0.1",
            port=8765,
            capabilities=["tool1", "tool2"]
        )
        
        manager.register_server(server_info)
        
        assert "test-server" in manager.servers
        assert manager.tool_registry["tool1"] == "test-server"
        assert manager.tool_registry["tool2"] == "test-server"
    
    async def test_tool_routing(self):
        """Test: Tools routed to correct servers"""
        manager = MCPClientManager()
        server1 = MCPServerInfo("server1", "127.0.0.1", 8765, ["tool1", "tool2"])
        server2 = MCPServerInfo("server2", "127.0.0.1", 8766, ["tool3", "tool4"])
        
        manager.register_server(server1)
        manager.register_server(server2)
        
        assert manager.route_to_server("tool1") == "server1"
        assert manager.route_to_server("tool3") == "server2"
        assert manager.route_to_server("unknown") is None


@pytest.mark.asyncio
async def test_end_to_end():
    """Test: End-to-end server/client communication"""
    # Start server
    server = DigimonMCPServer("e2e-server", ["echo", "math"], port=9998)
    
    # Register tools
    async def echo_tool(message: str, context: dict = None):
        return f"Echo: {message}"
    
    async def add_tool(a: int, b: int, context: dict = None):
        return a + b
    
    server.register_tool(MCPTool("echo", echo_tool))
    server.register_tool(MCPTool("math.add", add_tool))
    
    # Start server in background
    server_task = asyncio.create_task(server.start())
    
    # Give server time to start
    await asyncio.sleep(0.1)
    
    try:
        # Create client
        reader, writer = await asyncio.open_connection('127.0.0.1', 9998)
        
        # Send echo request
        request = MCPRequest(
            id="e2e-1",
            tool_name="echo",
            params={"message": "Hello MCP"},
            context={},
            session_id="e2e-session",
            timestamp=datetime.utcnow()
        )
        
        request_data = json.dumps(request.to_json()).encode()
        writer.write(request_data)
        await writer.drain()
        
        # Read response
        response_length_data = await reader.readexactly(4)
        response_length = int.from_bytes(response_length_data, 'big')
        response_data = await reader.readexactly(response_length)
        response_dict = json.loads(response_data.decode())
        
        assert response_dict['status'] == 'success'
        assert response_dict['result'] == 'Echo: Hello MCP'
        
        # Test context persistence
        context_request = MCPRequest(
            id="e2e-2",
            tool_name="math.add",
            params={"a": 5, "b": 3},
            context={},
            session_id="e2e-session",
            timestamp=datetime.utcnow()
        )
        
        request_data = json.dumps(context_request.to_json()).encode()
        writer.write(request_data)
        await writer.drain()
        
        response_length_data = await reader.readexactly(4)
        response_length = int.from_bytes(response_length_data, 'big')
        response_data = await reader.readexactly(response_length)
        response_dict = json.loads(response_data.decode())
        
        assert response_dict['status'] == 'success'
        assert response_dict['result'] == 8
        
        writer.close()
        await writer.wait_closed()
        
    finally:
        # Stop server
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        await server.stop()


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_end_to_end())