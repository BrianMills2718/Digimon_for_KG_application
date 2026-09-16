# ADR-001: Agent Orchestration Architecture

**Status:** Superseded by [ADR-002](002-harness-first-capability-architecture.md)  
**Original date:** 2026-02-15  
**Superseded:** 2026-09-16

## Historical decision

ADR-001 established a **dual orchestration model**:

1. a DIGIMON-internal “brain” using `agentic_model` for orchestration and mid-pipeline reasoning; and
2. a capable external client such as Claude Code or Codex that could bypass some of that orchestration and drive DIGIMON directly.

The decision was useful while the typed operator/composition and MCP layers were still being established. It led to work such as separate model roles, config inspection/override tools, reference-method execution and a client-driven operator mode.

## Why it was superseded

By September 2026, the codebase had a clearer architectural center:

- typed slot/dataflow records;
- a 26-operator machine-readable registry;
- chain validation and pipeline execution;
- 10 reference method plans;
- a stdio MCP surface exposing individual capabilities;
- an external harness capable of adaptive planning with full user/conversation/tool context.

At that point, continuing to treat a broad internal agent brain as a co-equal architectural requirement would duplicate reasoning policy and blur ownership.

ADR-002 therefore changes the preferred boundary:

> **The external intelligent harness owns adaptive end-to-end orchestration by default. DIGIMON owns capabilities, contracts, resources/prerequisites, bounded model-assisted operations, and evidence/provenance.**

Internal model calls remain valid inside documented capabilities, and reference/auto execution modes remain useful conveniences. They are no longer the reason to maintain a second general-purpose cognitive architecture inside DIGIMON.

## Current guidance

Do not use ADR-001 as the active design authority.

Use:

- `docs/adr/002-harness-first-capability-architecture.md`
- `docs/ARCHITECTURE.md`
- `docs/CURRENT_STATE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`

The original full ADR-001 text remains available in Git history for project provenance.