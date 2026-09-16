# MCP Integration Detailed Plan — Historical

**Status:** Superseded  
**Original date:** 2025-06-06  
**Superseded:** 2026-09-16

This document originally specified a staged WebSocket MCP implementation with server/client/context checkpoints, tool migration, multi-agent coordination, and production/performance phases.

That plan no longer describes the active DIGIMON architecture and must not be used as an implementation mandate.

## What exists now

The public snapshot includes a modern stdio MCP surface:

- `digimon_mcp_stdio_server.py`

It uses `FastMCP` and exposes the current capability model: graph construction, typed retrieval/operator calls, reference method execution, optional auto selection, resource/config inspection, graph/community helpers, and cross-modal analysis.

The current architecture is **harness-first**:

- a capable external harness owns adaptive orchestration;
- DIGIMON owns capabilities, typed contracts, resources/prerequisites, bounded model-assisted operations, and evidence/provenance;
- AoT/GoT/ReAct and routing prompts are optional heuristics rather than mandatory runtimes;
- multi-agent coordination is not a prerequisite for ordinary DIGIMON use.

## Active implementation plan

Use the canonical documentation instead:

- `docs/CURRENT_STATE.md`
- `docs/ARCHITECTURE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`
- `docs/adr/002-harness-first-capability-architecture.md`

The immediate architecture work is capability-contract stabilization, resource/prerequisite lifecycle, provenance/evidence propagation, harness-boundary cleanup, legacy consolidation, cross-modal normalization, error semantics, and contract-test/CI hardening.

## Historical provenance

The original checkpoint-by-checkpoint plan remains available in Git history for project lineage and may still contain useful implementation ideas. Its dates, targets, file-creation instructions, WebSocket assumptions, multi-agent sequence, and performance goals are historical rather than current requirements.