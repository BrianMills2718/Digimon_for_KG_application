# DIGIMON Architecture

**Status:** canonical target architecture for the public snapshot  
**Updated:** 2026-09-16

## Architectural thesis

DIGIMON should be a **capability, resource, and evidence system for intelligent harnesses**, not a hand-programmed replacement for an intelligent harness.

> **Program the capabilities, contracts, resource lifecycle, and evidence boundaries. Prompt useful reasoning heuristics. Let the harness remain intelligent.**

The codebase already has many required pieces: typed operator slots, a registry, composition/validation, multiple retrieval structures and an MCP surface. The architectural work now is to make those pieces coherent, explicit and dependable while reducing ambiguity from older internal-agent layers.

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
│ Contract + resource + evidence layer                       │
│ typed I/O | prerequisites | resources | errors             │
│ provenance | lifecycle | compatibility | lossiness         │
└────────────────────────────────────────────────────────────┘
        ↓
Graphs | VDBs | chunks | communities | matrices | tables/vectors
        ↓
Original source material
```

The **harness owns adaptive control policy**. DIGIMON owns the **capability/data/resource/evidence plane**.

## Responsibilities

### The external harness owns

- interpreting the user's goal and conversation context;
- deciding whether decomposition is useful at all;
- choosing graph, vector, text, table, community or hybrid operations;
- selecting, sequencing, retrying, branching, parallelizing and stopping tool calls;
- deciding when a reference method is useful versus a custom chain;
- revising strategy after observations;
- communicating uncertainty and remaining gaps to the user.

### DIGIMON owns

- corpus ingestion and normalization;
- construction/loading of retrieval resources;
- typed capability descriptions;
- stable input/output contracts;
- compatibility/prerequisite/resource facts;
- graph/vector/text/community/structured operations;
- source/evidence identifiers and traceability;
- explicit execution/error semantics;
- bounded internal LLM operations when an individual capability intrinsically requires semantic judgment;
- reference plans as inspectable conveniences rather than mandatory reasoning policy.

## Canonical capability model

### Typed values

The current operator core uses seven `SlotKind` values:

- query text;
- entity set;
- relationship set;
- chunk set;
- subgraph;
- community set;
- score vector.

This is a useful current vocabulary, not a claim that seven kinds are permanently sufficient. New kinds should be added only when they represent reusable capability semantics—not to encode an entire thought process.

### Capability descriptors

A canonical harness-facing capability should eventually describe at least:

- stable capability ID and version;
- human-readable purpose;
- typed inputs/outputs;
- resource prerequisites;
- produced/modified resources;
- deterministic vs model-assisted behavior;
- cost/side-effect/lossiness characteristics;
- failure classes;
- implementation binding;
- evidence/provenance behavior.

`Core/Schema/OperatorDescriptor.py` and `Core/Operators/registry.py` are the current foundation. Build/config/analysis/cross-modal MCP tools should map into the same conceptual model rather than forming a second undocumented capability universe.

## Composition contract

Composition should remain a way to connect capabilities, not a hidden global planner.

The target contract is:

1. every connection is explicitly typed;
2. every prerequisite is inspectable;
3. static validation returns machine-readable errors/warnings;
4. execution policy is explicit;
5. runtime dispatch rechecks required invariants;
6. output lineage/errors remain structured.

### Strict vs best-effort execution

The current code has permissive static validation in places and `OperatorComposer` can proceed after validation failure. The target architecture should make this a deliberate caller-visible choice:

- **strict mode** — invalid plans do not execute; preferred default for canonical harness-facing execution;
- **best-effort mode** — execution may proceed with explicit warnings when the caller intentionally requests it.

Best-effort behavior should never be mistaken for successful validation.

### Discovery is not proof of executability

Compatibility helpers may suggest chains based on slot kinds, but a discovered chain is not executable until resource prerequisites, field requirements, configuration and explicit wiring also validate.

The architecture should preserve this distinction in names, metadata and tests.

## Reference methods

The ten named retrieval methods remain useful for:

- known reference compositions;
- regression/compatibility tests;
- simple-client shortcuts;
- optional routing heuristics;
- later benchmarking.

They are **not** the unit of system intelligence. A capable harness may call a reference method, compose operators itself, or avoid graph reasoning entirely.

## MCP as the preferred harness boundary

`digimon_mcp_stdio_server.py` is currently the strongest external-harness surface.

The preferred hierarchy is:

### Mode A — harness-composed capabilities

The harness inspects resources/capabilities and calls what it needs. This is the conceptual default.

### Mode B — reference method execution

The harness selects a named reference plan and DIGIMON executes it.

### Mode C — heuristic auto-selection

DIGIMON uses a prompt/model to select a reference method. This is optional convenience behavior, not hidden global policy.

Future MCP work should make these modes share capability/resource/error/evidence contracts rather than maintaining separate metadata systems.

## Internal LLM operations are bounded capabilities

Some operations naturally require semantic judgment: entity extraction, relation/path relevance, reranking, answer synthesis or iterative steps inside a reference method.

That does **not** imply a global internal agent brain.

The distinction is:

- an **operator/capability** owns its documented local model-assisted transformation;
- the **external harness** owns the adaptive end-to-end policy by default;
- legacy `PlanningAgent`/orchestrator/AoT modules are compatibility/history unless deliberately retained for a concrete supported path.

## AoT / GoT / ReAct policy

Atom-of-Thought, Graph-of-Thought and ReAct are **reasoning heuristics**, not mandatory runtime state machines.

A dependency-aware suggestion may look like:

```text
q1: identify the performer who portrayed Corliss Archer in Kiss and Tell
q2: find government positions held by <q1.entity>
q3: determine which position is supported by retrieved source evidence
```

The harness may follow, merge, reorder, branch, parallelize, revise or ignore the suggestion.

### Do not confuse dependency expression with a required DAG runtime

A formal dependency graph should only be introduced where it enables a concrete system capability such as:

- parallel scheduling;
- resumability;
- caching;
- provenance/auditing;
- persisted work-state coordination.

If a generic typed task/text-list output is sufficient, prefer that over a cognitive-runtime abstraction.

### Prompt ownership

Equivalent prompt policy currently exists in YAML files and typed meta-operator code. The target architecture should eliminate silent drift by either:

- centralizing prompt/template loading, or
- explicitly mapping each execution path to its authoritative prompt and testing semantic parity.

Prompt version/provenance should be discoverable when prompt behavior materially affects an output.

## Resource architecture

Every derived artifact should eventually be inspectable through one consistent abstraction.

A resource should expose at least:

- stable resource ID;
- resource kind (corpus, graph, VDB, communities, sparse matrices, table/vector artifact, etc.);
- dataset/source identity;
- build/configuration fingerprint;
- availability/state;
- dependencies/prerequisites;
- producer/version metadata;
- source/evidence lineage where applicable;
- reuse/rebuild/incremental-update/invalidation semantics.

`GraphRAGContext` plus MCP/filesystem resource logic are the current foundation, not the finished resource model.

### Runtime/session scope

The current stdio server uses process-level state. That is acceptable for the present single-process model, but supported session/concurrency semantics should be explicit. Do not add distributed/multi-session machinery until a concrete deployment requirement exists.

## Evidence and provenance invariant

The architecture should make this question answerable for every material final claim:

> **What source evidence caused the system to assert this?**

Desired lineage:

```text
final claim
   ↓
retrieved evidence item / explicit inference
   ↓
entity / relationship / path / community / chunk
   ↓
source chunk identifier
   ↓
original document/source metadata
```

Current `source_id`, `chunk_id`, `SlotValue.producer` and metadata are foundations, not a universal contract.

Important rules:

- missing graph evidence is not proof of absence;
- inferred claims must be distinguishable from directly retrieved assertions;
- conflicting evidence should remain visible;
- source identifiers should survive composition where possible;
- aggregations/conversions should state when lineage was combined or lost;
- a synthesizer should consume structured evidence instead of reconstructing provenance from prose where practical.

## Prerequisites and lifecycle

A harness should not need hidden implementation knowledge to discover that a capability requires a VDB, community structure or sparse matrices.

Target flow:

1. inspect capability requirements;
2. inspect available resources;
3. determine compatibility/currentness;
4. choose reuse/build/fallback;
5. execute;
6. register produced resources;
7. invalidate dependent resources when inputs change.

`auto_build=True` can remain a convenience policy over these explicit facts.

## Cross-modal architecture

Graph, table and vector forms are complementary. Existing conversion code is useful; the target is to normalize it into the same capability/resource/evidence system.

A conversion should declare:

- input resource/schema;
- output resource/schema;
- parameters/provider;
- whether it is lossy;
- provenance mapping where possible;
- reusable resource identity/fingerprint.

Modality-selection prompts are advisory; the harness owns the decision.

## Error model

Failures must be machine-actionable. At minimum distinguish:

- resource not found;
- prerequisite absent;
- invalid plan/wiring/type;
- incompatible configuration/resource;
- provider/model failure;
- empty retrieval result;
- likely extraction incompleteness/unsupported evidence;
- unsupported/lossy conversion;
- timeout/internal failure.

A returned empty set and an execution failure are not equivalent.

The target result/error convention should let the harness decide whether to build, retry, fall back, reformulate or stop without parsing arbitrary prose.

## Legacy and compatibility policy

The repository contains valuable older implementations. They should be classified rather than silently mixed into the target architecture.

### Legacy/transitional examples

- `Core/AOT/` programmed atomic-state/transition logic;
- `Core/AgentBrain/` broad internal planning logic;
- multiple `Core/AgentOrchestrator/` implementations;
- older `Core/MCP/` server/client/coordination experiments;
- old WebSocket MCP/checkpoint and UKRF/multi-agent plans;
- CLI paths that still instantiate the internal planner/orchestrator.

Legacy code may remain while it supports compatibility/tests/history. New architecture work should not deepen dependencies on it unless a deliberate ADR says otherwise.

## Design invariants

1. **Harness-first:** adaptive user-level reasoning belongs to the capable caller by default.
2. **Capability-first:** tools describe reusable actions, not hidden global policies.
3. **Typed composition:** I/O and prerequisites are discoverable and machine-checkable.
4. **Explicit validation policy:** invalid, warning, strict and best-effort states are distinguishable.
5. **Evidence preservation:** transformations preserve or explicitly characterize lineage.
6. **Resource transparency:** the harness can inspect what exists, what is stale and what is required.
7. **Graceful incompleteness:** missing KG structure can trigger text/vector fallback rather than false conclusions.
8. **Heuristics remain overridable:** routing/decomposition prompts are priors, not laws.
9. **Reference methods remain optional:** named pipelines are conveniences/baselines, not system identity.
10. **Prompt behavior is traceable:** prompt sources/versions do not silently drift across execution paths.
11. **One canonical documentation hierarchy:** current status and plans do not live in competing trackers.
12. **No benchmark-driven architecture for now:** evaluation follows architecture stabilization.

## Non-goals for the current architecture phase

The current phase is not primarily about:

- maximizing benchmark scores;
- claiming research novelty;
- adding more dashboards or agent shells;
- building a general multi-agent society/coordination framework;
- creating a mandatory cognitive architecture;
- forcing every question through graph reasoning;
- optimizing production-scale latency before resource/contracts are stable.

## Relationship to other canonical docs

- [CURRENT_STATE.md](CURRENT_STATE.md) — how much of this architecture exists now.
- [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md) — exact module classification and current implementation caveats.
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — distance from target.
- [ROADMAP.md](ROADMAP.md) — closure order and exit criteria.
- [AGENT_INTELLIGENCE_ENHANCEMENTS.md](AGENT_INTELLIGENCE_ENHANCEMENTS.md) — reasoning-policy detail.
- [FUTURE_EVALUATION_QUESTIONS.md](FUTURE_EVALUATION_QUESTIONS.md) — later validation questions.