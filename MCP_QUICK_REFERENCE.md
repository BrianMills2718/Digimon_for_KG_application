# MCP Implementation Quick Reference

## File Structure to Create

```
Core/MCP/
├── __init__.py
├── base_mcp_server.py          # Checkpoint 1.1
├── mcp_client_manager.py       # Checkpoint 1.2  
├── shared_context.py           # Checkpoint 1.3
├── tools/
│   ├── __init__.py
│   ├── entity_vdb_search.py   # Checkpoint 2.1
│   ├── graph_builders.py       # Checkpoint 2.2
│   └── tool_registry.py        # Checkpoint 2.3
├── agents/
│   ├── __init__.py
│   ├── mcp_agent_interface.py  # Checkpoint 3.1
│   ├── coordination.py         # Checkpoint 3.2
│   └── cross_modal_bridge.py   # Checkpoint 3.3
└── monitoring/
    ├── __init__.py
    ├── metrics.py              # Checkpoint 4.1
    └── security.py             # Checkpoint 4.2
```

## Key Classes to Implement

### Phase 1: Foundation
```python
# base_mcp_server.py
class MCPServer:
    async def start(self, port: int)
    async def handle_request(self, websocket, path)
    async def process_message(self, message: dict)
    
# mcp_client_manager.py
class MCPClientManager:
    async def connect(self, host: str, port: int)
    async def invoke_method(self, method: str, params: dict)
    def get_pool_statistics(self) -> dict

# shared_context.py
class SharedContextStore:
    async def get(self, session_id: str, key: str)
    async def set(self, session_id: str, key: str, value: Any)
    async def clear_session(self, session_id: str)
```

### Phase 2: Tool Migration
```python
# tools/tool_registry.py
class MCPToolRegistry:
    def register_tool(self, tool_id: str, handler: Callable)
    def get_tool(self, tool_id: str) -> MCPTool
    def list_tools(self) -> List[dict]
```

### Phase 3: Multi-Agent
```python
# agents/mcp_agent_interface.py
class MCPAgent:
    async def register(self, capabilities: List[str])
    async def discover_peers(self) -> List[AgentInfo]
    async def send_task(self, agent_id: str, task: dict)
```

## MCP Message Format

### Request
```json
{
    "id": "unique-request-id",
    "method": "Entity.VDBSearch",
    "params": {
        "query": "example"
    },
    "session_id": "session-123"
}
```

### Success Response
```json
{
    "id": "unique-request-id",
    "status": "success",
    "result": {
        // Method-specific results
    }
}
```

### Error Response
```json
{
    "id": "unique-request-id", 
    "status": "error",
    "error": "Error message",
    "code": "ERROR_CODE"
}
```

## Testing Commands

```bash
# Start MCP server for testing
python -m Core.MCP.base_mcp_server --port 8765

# Run specific checkpoint test
pytest tests/mcp/test_mcp_checkpoint_1_1.py -v -s

# Run with debugging
pytest tests/mcp/test_mcp_checkpoint_1_1.py -v -s --pdb

# Check coverage
pytest tests/mcp/ --cov=Core.MCP --cov-report=term-missing
```

## Common Issues & Solutions

### WebSocket Connection Errors
```python
# Ensure server is running before client tests
await asyncio.sleep(1)  # Give server time to start
```

### Import Errors
```python
# Add to PYTHONPATH if needed
import sys
sys.path.append('/home/brian/digimon_cc')
```

### Async Test Issues
```python
# Use pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
```

## Performance Targets

- Server startup: < 1s
- Echo request: < 100ms  
- Tool invocation overhead: < 200ms
- Connection pool reuse: > 90%
- Context operations: < 10ms
- End-to-end simple query: < 2s
- Complex cross-modal query: < 5s

## Dependencies to Add

```txt
# Add to requirements.txt
websockets>=10.0
aiohttp>=3.8.0
prometheus-client>=0.15.0
pyjwt>=2.6.0
```

## Useful Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Time operations
import time
start = time.time()
# ... operation ...
elapsed = time.time() - start
print(f"Operation took {elapsed*1000:.1f}ms")

# Check WebSocket state
print(f"WebSocket state: {websocket.state}")
```