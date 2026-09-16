# DIGIMON Architecture

**Status:** canonical target architecture for the public snapshot  
**Updated:** 2026-09-16

## Architectural thesis

DIGIMON should be a **capability and evidence system for intelligent harnesses**, not a hand-programmed replacement for an intelligent harness.

> **Program the capabilities, contracts, resource lifecycle, and evidence boundaries. Prompt useful reasoning heuristics. Let the harness remain intelligent.**

The codebase already has many of the pieces required for this direction: typed operator slots, an operator registry, composition/validation, multiple retrieval structures and an MCP tool surface. The architectural work now is to make those pieces coherent and dependable while reducing ambiguity created by older internal-agent layers.

## System boundary

```text
User / application goal
        ↓
Intelligent external harness
(Claude Code, Codex, another capable agent, or application policy)
        ↓
DIGIMON MCP / typed capability facade
        ↓
┌────────────────────────────────────────────────────────────┐
│ Capability layer                                           │
│ corpus | graph build | entity | relationship | chunk      │
│ subgraph | community | analysis | modality conversion     │
└────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────┐
│ Contract + resource layer                                  │
│ slot types | operator descriptors | prerequisites          │
│ resource identities | lifecycle | errors | provenance      │
└────────────────────────────────────────────────────────────┘
        ↓
Graphs | VDBs | chunks | communities | matrices | tables/vectors
        ↓
Original source material
```

The **harness owns the control policy**. DIGIMON owns the **capability/data/evidence plane**.

## Responsibilities

### The external harness owns

- interpreting the user's actual goal and conversation context;
- deciding whether a question needs decomposition at all;
- choosing graph, vector, text, table, community or hybrid retrieval;
- selecting, sequencing, retrying, parallelizing and stopping tool calls;
- deciding when a prior method/profile is useful versus when to build a custom chain;
- revising its plan after observations;
- deciding how to communicate uncertainty and remaining gaps to the user.

### DIGIMON owns

- corpus ingestion and normalization;
- construction and loading of retrieval resources;
- typed capability descriptions;
- stable operator input/output contracts;
- compatibility and prerequisite information;
- graph/vector/text/community/structured retrieval operations;
- source/evidence identifiers and traceability;
- deterministic resource/error semantics where possible;
- bounded internal LLM operations when a capability intrinsically calls for them;
- reference plans that are inspectable conveniences rather than mandatory reasoning policy.

## Canonical capability core

### Typed slots

`Core/Schema/SlotTypes.py` defines the current retrieval dataflow vocabulary:

- query text;
- entity sets;
- relationship sets;
- chunk sets;
- subgraphs;
- community sets;
- score vectors.

Entity and relationship records already include `source_id`; chunk records include `chunk_id`. These identifiers are the starting point for a stronger evidence contract.

### Operator descriptors and registry

`Core/Operators/registry.py` is the machine-readable capability catalog for the current 26-operator core. Descriptors specify:

- operator identity/category;
- required and produced slot kinds;
- cost tier;
- prerequisite flags such as entity VDB, relationship VDB, communities and sparse matrices;
- whether an LLM is required;
- usage guidance and limitations;
- implementation binding.

This is closer to the desired architecture than a monolithic planner because it describes **what can be done** without hard-coding **what must be done for every question**.

### Composition

`Core/Composition/` provides validation and execution infrastructure. `OperatorComposer` profiles ten reference methods, builds plans and executes them through the pipeline. Its design intentionally leaves method selection to the caller.

Reference methods are useful for:

- known-good compositions;
- compatibility/regression tests;
- quick execution for simple clients;
- optional routing heuristics;
- later benchmarking.

They are **not** the architectural unit of intelligence. The operator/capability layer is.

## MCP as the preferred harness boundary

`digimon_mcp_stdio_server.py` is the strongest current external-harness surface because it exposes both low-level capabilities and convenience abstractions.

The preferred hierarchy is:

### Mode A — harness-composed capabilities

The harness inspects operators/resources and calls the tools it needs. This is the conceptual default for a capable harness.

### Mode B — reference method execution

The harness chooses a named reference plan and lets DIGIMON execute that composition. Useful when the harness wants a compact known pattern.

### Mode C — heuristic auto-selection

DIGIMON can use an LLM/prompt to select a reference method. This remains optional. It should not become a hidden global reasoning policy.

A future architecture should preserve all three, but Mode A defines the boundary.

## Internal LLM operations are capabilities, not a second mandatory brain

Some operations are naturally model-assisted: entity extraction, relation/path relevance, answer synthesis or iterative reasoning inside a reference method. Those operations may call an internal LLM.

That does **not** imply DIGIMON needs a global internal cognitive architecture.

The distinction is:

- **acceptable internal intelligence:** a bounded operator performs a documented model-assisted transformation;
- **target orchestration:** the external harness decides which capability to invoke and what to do with its observation;
- **legacy/transitional orchestration:** `PlanningAgent`, multiple orchestrators and old cognitive/AoT modules making broad end-to-end decisions inside DIGIMON.

The architecture may retain reference internal execution for compatibility/simple clients, but it should not drive new complexity.

## AoT / GoT / ReAct policy

Atom-of-Thought, Graph-of-Thought and ReAct ideas are **prompt-level reasoning heuristics**, not mandatory runtime state machines.

A dependency-aware suggestion may look like:

```text
q1: identify the performer who portrayed Corliss Archer in Kiss and Tell
q2: find government positions held by <q1.entity>
q3: determine which position is supported by retrieved source evidence
```

The harness may follow, merge, reorder, branch, parallelize or ignore this structure. DIGIMON should only formalize a dependency graph when a concrete system capability requires it—for example explicit parallel scheduling, resumability, caching, provenance or auditability.

Do not add a graph-of-thought executor merely because the prompt can express dependencies.

## Resource architecture

The target resource model should make every derived artifact inspectable through one consistent abstraction.

A resource should eventually expose at least:

- stable resource identifier;
- resource kind (corpus, graph, VDB, community set, sparse matrix, table/vector conversion, etc.);
- dataset/source identity;
- build/configuration fingerprint;
- availability/state;
- dependencies/prerequisites;
- producer/version metadata;
- source/evidence lineage where applicable;
- whether it can be reused, rebuilt, incrementally updated or invalidated.

`GraphRAGContext` plus the MCP filesystem/resource inspection logic are the current foundation, but the model is not yet unified.

## Evidence and provenance invariant

The architecture should make this question answerable for every material final claim:

> **What source evidence caused the system to assert this?**

The desired chain is:

```text
final claim
   ↓
retrieved evidence item / inference step
   ↓
entity / relationship / path / community / chunk
   ↓
source chunk identifier
   ↓
original document/source metadata
```

Current `source_id`/`chunk_id` fields are useful but insufficient as a universal contract. Operators should preserve lineage rather than forcing the final synthesizer to reconstruct it heuristically.

Important rules:

- missing graph evidence is not proof of absence;
- inferred claims must be distinguishable from directly retrieved facts;
- conflicting evidence should remain visible;
- source identifiers should survive operator composition;
- transformations/conversions should record what information was dropped or derived.

## Prerequisites and lifecycle

A capable harness should not need hidden implementation knowledge to discover that an operator requires a VDB, communities or sparse matrices.

The operator descriptors already model several prerequisite flags. The target is to make prerequisite behavior systematic:

1. inspect capability requirements;
2. inspect available resources;
3. choose reuse/build/fallback;
4. execute;
5. register the produced artifact consistently;
6. invalidate downstream resources when their dependencies change.

`auto_build=True` can remain a convenience, but its behavior should be a thin policy over explicit resource/prerequisite contracts.

## Cross-modal architecture

Graph, table and vector representations are complementary. Existing conversion code is useful, but the target is to normalize cross-modal transformations into the same resource/evidence model.

A conversion should declare:

- input resource kind/schema;
- output kind/schema;
- transformation parameters;
- whether it is lossy;
- provenance mapping where possible;
- reusable resource identity/fingerprint.

Modality selection prompts may suggest a path; the harness decides whether to follow it.

## Error model

Failures should be machine-actionable. The target tool result/error model should distinguish at least:

- resource not found;
- prerequisite absent;
- invalid/incompatible input;
- provider/model failure;
- empty retrieval result;
- graph extraction incompleteness;
- conversion unsupported/lossy;
- execution timeout/internal failure.

This lets the harness decide whether to build, retry, fall back, reformulate or stop.

## Legacy and compatibility policy

The repository contains valuable older implementations. They should be classified rather than silently mixed into the target architecture.

### Legacy/transitional examples

- `Core/AOT/` programmed atomic-state/transition logic;
- `Core/AgentBrain/` broad internal planning logic;
- multiple `Core/AgentOrchestrator/` implementations;
- old WebSocket MCP implementation plans and checkpoint trackers;
- old UKRF/multi-agent roadmaps;
- CLI paths that still instantiate the internal planner/orchestrator.

Legacy code may remain while it supports tests, compatibility or historical understanding. New architectural work should not deepen dependencies on it unless a deliberate ADR says otherwise.

## Design invariants

1. **Harness-first:** adaptive reasoning belongs to the capable caller by default.
2. **Capability-first:** operators describe reusable actions, not hidden end-to-end policies.
3. **Typed composition:** inputs/outputs and prerequisites should be discoverable and machine-checkable.
4. **Evidence preservation:** transformations should preserve or explicitly describe lineage.
5. **Resource transparency:** the harness can inspect what exists and what is required.
6. **Graceful incompleteness:** missing KG structure can trigger text/vector fallback rather than false conclusions.
7. **Heuristics remain overridable:** routing/decomposition prompts are priors, not laws.
8. **Reference methods remain optional:** named pipelines are conveniences and baselines, not the system identity.
9. **One canonical documentation hierarchy:** current status and plans do not live in competing checkpoint documents.
10. **No benchmark-driven architecture for now:** benchmarking is deferred until the capability/resource/evidence architecture is coherent.

## Non-goals for the current architecture phase

The current phase is **not** primarily about:

- maximizing benchmark scores;
- claiming research novelty;
- adding more dashboards or agent shells;
- building a general multi-agent society/coordination framework;
- creating a mandatory cognitive architecture;
- forcing every question through graph reasoning;
- optimizing latency targets before the resource/contracts model is stable.

Those may be revisited later if they serve concrete use cases.

## Relationship to other canonical docs

- **[CURRENT_STATE.md](CURRENT_STATE.md)** — how much of this architecture exists now.
- **[GAP_ANALYSIS.md](GAP_ANALYSIS.md)** — exact distance from target.
- **[ROADMAP.md](ROADMAP.md)** — planned closure order.
- **[AGENT_INTELLIGENCE_ENHANCEMENTS.md](AGENT_INTELLIGENCE_ENHANCEMENTS.md)** — more reasoning-policy detail.
- **[FUTURE_EVALUATION_QUESTIONS.md](FUTURE_EVALUATION_QUESTIONS.md)** — later validation questions.