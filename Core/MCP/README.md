# DIGIMON MCP Server

This directory contains the Model Context Protocol (MCP) server implementation for DIGIMON, which wraps 5 core DIGIMON tools for external access.

## Overview

The DIGIMON MCP Server (`digimon_mcp_server.py`) provides MCP access to the following core tools:

1. **Entity.VDBSearch** - Search for entities in a vector database using text queries
2. **Graph.Build** - Build an entity-relationship graph from a corpus
3. **Corpus.Prepare** - Prepare a corpus from a directory of text files
4. **Chunk.Retrieve** - Retrieve text chunks associated with entities
5. **Answer.Generate** - Generate answers based on query and context

## Architecture

The MCP server:
- Runs on port 8765 by default
- Manages GraphRAGContext instances per session
- Handles tool registration and request routing
- Provides proper error handling and logging
- Maintains performance metrics

## Usage

### Starting the Server

```bash
# From the digimon_cc directory
./run_mcp_server.sh

# Or directly:
python -m Core.MCP.digimon_mcp_server --port 8765 --config Option/Config2.yaml
```

### Configuration

The server requires a valid `Option/Config2.yaml` configuration file with:
- LLM provider settings
- Embedding model configuration
- Chunk processing parameters
- Graph construction settings

### Tool Details

#### Entity.VDBSearch
```json
{
  "tool_name": "Entity.VDBSearch",
  "params": {
    "vdb_reference_id": "dataset_entity_vdb",
    "query_text": "search query",
    "top_k_results": 5,
    "session_id": "session-123",
    "dataset_name": "MySampleTexts"
  }
}
```

#### Graph.Build
```json
{
  "tool_name": "Graph.Build", 
  "params": {
    "target_dataset_name": "MySampleTexts",
    "force_rebuild": false,
    "session_id": "session-123"
  }
}
```

#### Corpus.Prepare
```json
{
  "tool_name": "Corpus.Prepare",
  "params": {
    "input_directory_path": "Data/MySampleTexts",
    "output_directory_path": "Data/MySampleTexts_Output",
    "target_corpus_name": "my_corpus",
    "session_id": "session-123"
  }
}
```

#### Chunk.Retrieve
```json
{
  "tool_name": "Chunk.Retrieve",
  "params": {
    "entity_ids": ["entity1", "entity2"],
    "graph_reference_id": "MySampleTexts_ERGraph",
    "max_chunks_per_entity": 3,
    "session_id": "session-123",
    "dataset_name": "MySampleTexts"
  }
}
```

#### Answer.Generate
```json
{
  "tool_name": "Answer.Generate",
  "params": {
    "query": "What is the main topic?",
    "context": "Retrieved context from other tools...",
    "response_type": "default",
    "use_tree_search": false
  }
}
```

## Testing

Run the test script to verify the server is working:

```bash
# Start the server first
./run_mcp_server.sh

# In another terminal, run tests
python test_mcp_server.py
```

## Integration

The MCP server can be integrated with:
- External AI agents
- API gateways
- Workflow orchestration systems
- Custom client applications

## Error Handling

The server provides structured error responses:
```json
{
  "status": "error",
  "result": {
    "error": "Error message",
    "code": "ERROR_CODE",
    "details": {}
  }
}
```

## Performance Metrics

The server tracks:
- Request count
- Error count and rate
- Average latency
- Connected clients
- Registered tools

Access metrics via the `get_metrics()` method.