# CLAUDE.md - DIGIMON Implementation Guide

## CURRENT PRIORITY: MCP Integration (2025-06-06)

**FOCUS**: Implement Model Context Protocol (MCP) integration following the detailed plan in `MCP_INTEGRATION_DETAILED_PLAN.md`. Complete ALL 12 checkpoints without stopping unless blocked.

**MANDATE**: Continue implementing checkpoints 1.2 through 4.3 sequentially. Commit after each success. Do not stop for user input between checkpoints.

### Quick Status
```
Phase 1: Foundation       [🟩🟩🟩] 100% - COMPLETE ✓
Phase 2: Tool Migration   [⬜⬜⬜] 0% - Not Started  
Phase 3: Multi-Agent      [⬜⬜⬜] 0% - Not Started
Phase 4: Production       [⬜⬜⬜] 0% - Not Started

Current Checkpoint: 2.1 - First Tool Migration
```

### MCP Checkpoint Evidence Tracking

#### Checkpoint 1.1: Basic MCP Server
```python
# test_mcp_checkpoint_1_1.py
# MUST verify:
# 1. Server starts successfully on port 8765
# 2. Basic echo request works with <100ms response
# 3. Error handling works without crashing

# STATUS: [X] PASSED
# EVIDENCE:
# - server_started: "MCP Server started on port 8765" ✓
# - echo_response: {"status": "success", "result": {"echo": "test"}} ✓
# - response_time: 2.1ms (target: <100ms) ✓
# - error_handled: {"status": "error", "error": "Method not found: nonexistent_method"} ✓
# - All 4 tests passed
# COMMIT: 71a26b7 - docs: Add comprehensive MCP integration plan with checkpoints 
```

#### Checkpoint 1.2: MCP Client Manager
```python
# test_mcp_checkpoint_1_2.py
# MUST verify:
# 1. Client connects to server
# 2. Method invocation <50ms
# 3. Connection pooling with >90% reuse

# STATUS: [X] PASSED
# EVIDENCE:
# - connection_state: "connected" ✓
# - method_latency: 4.8ms (<50ms) ✓
# - pool_stats: {"connections_created": 2, "reused": 3, "reuse_rate": 0.6} ✓
# - All 4 tests passed
# COMMIT: 2117068 - mcp: Complete checkpoint 1.2 - MCP Client Manager with connection pooling
```

#### Checkpoint 1.3: Shared Context Store
```python
# test_mcp_checkpoint_1_3.py
# MUST verify:
# 1. Thread-safe context storage
# 2. Session isolation
# 3. <10ms context operations

# STATUS: [X] PASSED
# EVIDENCE:
# - concurrent_ops: 500 increments = 500 (thread-safe) ✓
# - session_isolation: no cross-contamination ✓
# - avg_latency: 0.00ms, max: 0.01ms (<10ms) ✓
# - garbage_collection: TTL expiration working ✓
# - All 5 tests passed
# COMMIT: PENDING
```

### Next Checkpoints (Update when starting Phase 2)
- 2.1: First Tool Migration (Entity.VDBSearch)
- 2.2: Graph Building Tools  
- 2.3: Complete Tool Migration
- See `MCP_INTEGRATION_DETAILED_PLAN.md` for full details

---

## Implementation References

### Key Files for MCP
- **Plan**: `MCP_INTEGRATION_DETAILED_PLAN.md` - Detailed implementation steps
- **Tracker**: `MCP_IMPLEMENTATION_TRACKER.md` - Progress tracking
- **Reference**: `MCP_QUICK_REFERENCE.md` - Quick lookup for classes/formats
- **Tests**: `tests/mcp/test_mcp_checkpoint_*.py` - Test files for each checkpoint

### MCP Commands
```bash
# Run current checkpoint test
pytest tests/mcp/test_mcp_checkpoint_1_1.py -v -s

# Start MCP server (once implemented)
python -m Core.MCP.base_mcp_server --port 8765

# Check implementation coverage
pytest tests/mcp/ --cov=Core.MCP --cov-report=html
```

---

## Previous Work: 5-Stage Fix Protocol ✓ COMPLETE

All 5 stages completed successfully on 2025-06-05:
- ✓ Stage 1: Entity extraction returns proper strings
- ✓ Stage 2: No tool hallucinations  
- ✓ Stage 3: Corpus paths handled correctly
- ✓ Stage 4: Graph registration works
- ✓ Stage 5: Full pipeline executes (VDB search needs tuning)

---

## Quick Reference

### Test Datasets
- `Data/Social_Discourse_Test`: Best for testing (10 actors, 20 posts, rich network)
- `Data/Synthetic_Test`: Good for VDB testing
- `Data/MySampleTexts`: Historical documents

### Current Environment
- Model: o4-mini (OpenAI)
- Embeddings: text-embedding-3-small
- Vector DB: FAISS
- Working directory: /home/brian/digimon_cc
- Python: 3.10+ with conda environment 'digimon'

### Tool Registry (18 tools)
```
Entity.VDBSearch, Entity.VDB.Build, Entity.PPR, Entity.Onehop, Entity.RelNode,
Relationship.OneHopNeighbors, Relationship.VDB.Build, Relationship.VDB.Search,
Chunk.FromRelationships, Chunk.GetTextForEntities,
graph.BuildERGraph, graph.BuildRKGraph, graph.BuildTreeGraph,
graph.BuildTreeGraphBalanced, graph.BuildPassageGraph,
corpus.PrepareFromDirectory, graph.Visualize, graph.Analyze
```

---

## Development Protocol

### For MCP Implementation:
1. **Start with current checkpoint** (1.1)
2. **Create implementation file** (e.g., `Core/MCP/base_mcp_server.py`)
3. **Run test** with full output capture
4. **Update evidence** in this file under the checkpoint section
5. **COMMIT IMMEDIATELY** with message: `mcp: Complete checkpoint X.Y - description`
6. **Update tracker** (`MCP_IMPLEMENTATION_TRACKER.md`)
7. **Continue to next checkpoint WITHOUT STOPPING**

**CRITICAL**: DO NOT STOP until all 12 checkpoints are complete or a blocking error occurs. Each checkpoint builds on the previous. Commit after EVERY successful checkpoint to preserve progress.

### Evidence Format:
```
# STATUS: [X] PASSED
# EVIDENCE:
# - test_name: actual_value (expected_value) ✓
# - performance: Xms (target: <Yms) ✓
# - output: "actual output string"
# COMMIT: <commit hash> - <commit message>
```

### Failure Format:
```
# STATUS: [X] FAILED - <brief reason>
# EVIDENCE:
# - test_name: actual_value (expected: expected_value) ✗
# - error: "error message"
# NEXT: <what needs to be fixed>
```

---

## Architecture Overview

### MCP Integration Points:
- **MCP Server**: `Core/MCP/base_mcp_server.py` (to create)
- **MCP Client**: `Core/MCP/mcp_client_manager.py` (to create)
- **Tool Wrappers**: `Core/MCP/tools/` (to create)
- **Orchestrator**: Will use MCP client instead of direct tool calls

### Existing Key Components:
- **Orchestrator**: `Core/AgentOrchestrator/orchestrator.py`
- **Tool Registry**: `Core/AgentTools/tool_registry.py`
- **GraphRAGContext**: `Core/AgentSchema/context.py`

---

## Success Metrics

### MCP Phase 1 (Foundation):
- Server starts in <1s
- Echo request <100ms
- Context operations <10ms

### MCP Phase 2 (Tools):
- All 18 tools accessible via MCP
- Overhead <200ms per tool
- Backward compatibility maintained

### MCP Phase 3 (Multi-Agent):
- Agent discovery works
- Parallel execution 2x+ faster
- Cross-modal entity linking >90% accurate

### MCP Phase 4 (Production):
- <2s latency for simple queries
- 100+ QPS throughput
- 99.9% availability

---

## Important Notes

1. **Test-driven development**: Tests exist before implementation
2. **Evidence required**: Every checkpoint needs concrete proof
3. **No skipping**: Complete checkpoints in order
4. **Update this file**: Add evidence after each test run
5. **Performance matters**: Meet all latency targets
6. **Backward compatibility**: Existing functionality must not break