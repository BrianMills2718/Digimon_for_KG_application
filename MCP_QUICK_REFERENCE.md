# MCP Quick Reference — Historical

**Status:** Superseded  
**Original planning era:** 2025  
**Superseded:** 2026-09-16

This file originally served as a quick reference for a planned WebSocket MCP migration, including files to create and checkpoint-oriented implementation steps.

That is no longer the active MCP architecture.

## Current MCP surface

The public snapshot's current external harness surface is:

- `digimon_mcp_stdio_server.py`

It uses `FastMCP` over stdio and exposes three useful levels of execution:

1. **individual capabilities/operators** — preferred conceptual mode for capable external harnesses;
2. **reference method execution** — execute one of the known operator compositions;
3. **optional auto selection** — a prompt/model selects a reference method when desired.

The same server also exposes corpus/graph construction, resource/config inspection, community/prerequisite helpers, graph analysis, and cross-modal tools.

## Current architecture references

Use these instead of the old MCP migration checklist:

- `docs/CURRENT_STATE.md`
- `docs/ARCHITECTURE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`
- `docs/adr/002-harness-first-capability-architecture.md`
- `AGENTS.md`
- `CLAUDE.md`

## Current design rule

The external intelligent harness owns adaptive orchestration. DIGIMON owns composable capabilities, typed contracts, resource/prerequisite facts, bounded model-assisted operations, and evidence/provenance.

The original quick-reference checklist remains available in Git history for project provenance.