# UKRF MCP Interface Specification
**Version:** 1.0  
**Purpose:** Alignment document for StructGPT and Autocoder integration teams

## Overview

This document defines the exact MCP interfaces that StructGPT and Autocoder must implement to integrate with DIGIMON's Universal Knowledge Reasoning Framework.

## 1. MCP Server Requirements

Each system MUST implement an MCP server with:

```python
# Required server info
server = MCPServerInfo(
    name: str,          # "structgpt-mcp" or "autocoder-mcp"
    host: "127.0.0.1",  
    port: int,          # 8766 for StructGPT, 8767 for Autocoder
    capabilities: List[str]  # Tool names this server provides
)
```

## 2. Required Tool Interfaces

### StructGPT Tools

```python
# Tool 1: SQL Generation
@mcp_tool("structgpt.sql_generation")
async def generate_sql(params: Dict, context: Dict) -> Dict:
    """
    Input:
      params: {
        "question": str,
        "database_schema": Dict,  # {"tables": [...], "relationships": [...]}
        "dialect": str  # "postgresql", "mysql", etc.
      }
    Output: {
        "sql": str,
        "confidence": float,
        "explanation": str
    }
    """

# Tool 2: Table QA
@mcp_tool("structgpt.table_qa")  
async def answer_table_question(params: Dict, context: Dict) -> Dict:
    """
    Input:
      params: {
        "question": str,
        "table_data": List[Dict],  # Row-oriented data
        "table_schema": Dict       # Column definitions
      }
    Output: {
        "answer": Any,
        "reasoning": str,
        "confidence": float
    }
    """

# Tool 3: Entity Extraction
@mcp_tool("structgpt.extract_entities")
async def extract_entities(params: Dict, context: Dict) -> Dict:
    """
    Input:
      params: {
        "sql_result": List[Dict],
        "entity_types": List[str]  # From DIGIMON ontology
      }
    Output: {
        "entities": List[{
            "name": str,
            "type": str,
            "properties": Dict,
            "source_column": str
        }]
    }
    """
```

### Autocoder Tools

```python
# Tool 1: Capability Generation
@mcp_tool("autocoder.generate_capability")
async def generate_capability(params: Dict, context: Dict) -> Dict:
    """
    Input:
      params: {
        "capability_gap": {
            "description": str,
            "input_schema": Dict,
            "output_schema": Dict,
            "examples": List[Dict]
        },
        "target_language": str,  # "python"
        "constraints": List[str]
      }
    Output: {
        "tool_name": str,
        "source_code": str,
        "dependencies": List[str],
        "validation_status": str,
        "mcp_wrapper_code": str  # Ready to register
    }
    """

# Tool 2: Tool Validation
@mcp_tool("autocoder.validate_tool")
async def validate_tool(params: Dict, context: Dict) -> Dict:
    """
    Input:
      params: {
        "source_code": str,
        "test_cases": List[Dict],
        "performance_requirements": Dict
      }
    Output: {
        "valid": bool,
        "test_results": List[Dict],
        "performance_metrics": Dict,
        "issues": List[str]
    }
    """
```

## 3. Shared Context Format

All tools MUST use this context structure:

```python
context = {
    "session_id": str,
    "query_id": str,
    "entities": {  # Shared entity registry
        "<entity_id>": {
            "name": str,
            "type": str,
            "source": str,  # "graph", "sql", "generated"
            "properties": Dict,
            "embeddings": List[float]  # Optional
        }
    },
    "schemas": {  # Shared schema mappings
        "graph_to_sql": Dict,
        "sql_to_graph": Dict
    },
    "execution_history": List[{
        "tool": str,
        "timestamp": str,
        "result_summary": str
    }]
}
```

## 4. Cross-Modal Entity Linking

When entities are discovered, they MUST be registered in shared context:

```python
# After StructGPT extracts entities from SQL
await mcp_context.update({
    "entities": {
        "entity_123": {
            "name": "Albert Einstein",
            "type": "Person",
            "source": "sql",
            "properties": {
                "table": "authors",
                "id": 42
            }
        }
    }
})

# DIGIMON can then link to graph entities
if similar_to(context["entities"]["entity_123"], graph_entity):
    await mcp_context.update({
        "schemas": {
            "sql_to_graph": {
                "authors.id=42": "graph_node_456"
            }
        }
    })
```

## 5. Error Handling

All tools MUST return errors in this format:

```python
{
    "error": {
        "code": str,  # "INVALID_SCHEMA", "GENERATION_FAILED", etc.
        "message": str,
        "details": Dict,
        "recoverable": bool
    }
}
```

## 6. Performance Requirements

- Tool execution: <1 second (p50)
- MCP overhead: <50ms
- Context updates: <10ms
- Memory per session: <100MB

## 7. Integration Test Scenarios

### Scenario 1: SQL to Graph Entity Linking
```python
# 1. DIGIMON asks about "papers by Einstein"
# 2. StructGPT generates SQL for authors table
# 3. StructGPT extracts "Albert Einstein" entity
# 4. DIGIMON links to existing graph node
# 5. Combined results returned
```

### Scenario 2: Dynamic Tool Generation
```python
# 1. DIGIMON detects need for custom aggregation
# 2. Autocoder generates aggregation tool
# 3. Tool registered via MCP
# 4. DIGIMON uses new tool immediately
# 5. Results integrated with other tools
```

## 8. Development Checklist

For StructGPT team:
- [ ] Implement MCP server on port 8766
- [ ] Implement 3 required tools
- [ ] Support shared context format
- [ ] Register entities after extraction
- [ ] Handle cross-modal schema updates

For Autocoder team:
- [ ] Implement MCP server on port 8767
- [ ] Implement capability generation tool
- [ ] Generate MCP-compatible tool wrappers
- [ ] Support validation and testing
- [ ] Enable runtime registration

## 9. Communication Protocol

All MCP communication follows:
```
Client → Request → MCP Server → Tool Execution → Response → Client
                        ↓               ↑
                   Shared Context Store
```

## Questions?

Contact integration lead or refer to MCP_INTEGRATION_PLAN.md for architectural details.