# ADR-002: Harness-First Capability Architecture

**Status:** Accepted  
**Date:** 2026-09-16  
**Supersedes:** ADR-001 as the preferred orchestration architecture

## Decision

DIGIMON will treat the **external intelligent harness as the default owner of adaptive orchestration**.

DIGIMON's architectural responsibility is to expose strong, inspectable and composable capabilities with stable contracts, resource/prerequisite semantics and source-evidence boundaries.

Reasoning techniques such as Atom-of-Thought, Graph-of-Thought, ReAct, modality routing and method selection may be supplied as **optional heuristics or reference policies**, but they are not a mandatory programmed cognitive runtime.

## Why this decision

The repository now has a typed operator system, registry, composition engine and MCP surface that allow capable clients to choose and sequence operations directly. This makes it unnecessary—and architecturally undesirable—to keep expanding a broad internal `AgentBrain` as the primary place where intelligence lives.

A capable harness already has advantages that an internal planner cannot reliably reproduce:

- the complete user goal and conversational context;
- awareness of other tools/systems outside DIGIMON;
- ability to inspect observations and adapt freely;
- native planning and reasoning capability;
- ability to decide that graph reasoning is unnecessary.

DIGIMON adds value by making knowledge-graph and retrieval operations trustworthy, typed, composable and evidence-grounded.

## Consequences

### New work should prioritize

- capability descriptors and typed I/O;
- resource identity/lifecycle/prerequisites;
- provenance/evidence propagation;
- MCP/tool discovery and execution consistency;
- machine-actionable errors;
- reference method plans as optional compositions;
- prompt heuristics that remain overridable.

### New work should not prioritize

- additional general-purpose internal planner/orchestrator variants;
- a mandatory AoT/GoT state machine;
- encoding the full reasoning graph before execution;
- multi-agent coordination as a prerequisite for ordinary DIGIMON use;
- forcing all clients through `auto_compose` or one named retrieval method.

## Internal LLM calls

This decision does **not** prohibit model-assisted operators.

A bounded capability may internally use an LLM when its operation intrinsically requires semantic judgment—for example entity extraction, path ranking, query-time reasoning within a reference method, or answer synthesis.

The distinction is ownership:

- the **operator** owns its documented local transformation;
- the **external harness** owns the adaptive end-to-end policy by default.

A simple client may still choose a reference method or auto-selection convenience when it wants DIGIMON to perform more of the orchestration. Those are supported modes, not the architectural default.

## AoT/GoT consequence

Dependency-aware decomposition is retained as a useful prompt heuristic.

Example:

```text
q1: identify an intermediate entity
q2: retrieve facts about <q1.entity>
q3: verify the resulting claim against source evidence
```

This representation is advisory. The harness may merge, branch, reorder, parallelize, revise or skip it.

A formal dependency DAG should only be introduced where it enables a concrete system function such as scheduling, resumability, caching, provenance or auditing.

## Relationship to ADR-001

ADR-001 described a dual orchestration model in which DIGIMON maintained a capable internal brain alongside a smart external client and asserted that some mid-pipeline reasoning had to remain internal.

That record was useful during development, but it no longer expresses the preferred boundary.

ADR-002 changes the emphasis:

- external harness orchestration is the default target;
- internal/reference execution remains a compatibility/convenience mode;
- model-assisted operations can remain bounded capabilities;
- the architecture should not assume that DIGIMON itself must reproduce a general agent brain.

## Implementation status

The architecture is **partially realized**:

- the typed 26-operator registry/composition core is implemented;
- the stdio MCP server exposes individual tools and reference/auto modes;
- dependency-aware decomposition/synthesis prompts are implemented;
- legacy internal planner/orchestrator and `Core/AOT` code still exists;
- resource lifecycle, uniform provenance and capability/MCP parity remain incomplete.

See:

- `docs/CURRENT_STATE.md`
- `docs/ARCHITECTURE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`

for the current reconciliation.