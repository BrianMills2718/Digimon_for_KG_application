# DIGIMON: Composable Knowledge-Graph RAG

DIGIMON is a research/application codebase for turning document collections into reusable **graph, vector, text, community, and structured retrieval capabilities** that an intelligent agent harness can compose to answer questions.

> **Program the capabilities, contracts, resources, and evidence boundaries. Prompt useful reasoning heuristics. Let the harness remain intelligent.**

## Current status

As of **2026-09-16**, this public snapshot is a **hybrid/transitional architecture**.

Its strongest modern core is already implemented in code:

- a typed slot/record system;
- a machine-readable registry of **26 composable operators**;
- chain validation and pipeline execution;
- **10 reference retrieval methods** represented as operator plans;
- corpus and five graph-build surfaces;
- entity, relationship, chunk, subgraph, community and meta operations;
- a stdio MCP server that exposes individual capabilities plus reference/auto execution modes;
- graph/table/vector cross-modal conversion code;
- evaluation and end-to-end testing infrastructure.

The remaining architecture work is not “invent an agent brain.” It is to make the existing capability system coherent and dependable: **capability/MCP parity, explicit validation semantics, resource lifecycle/prerequisites, provenance/evidence propagation, prompt ownership, machine-actionable errors, and cleanup of legacy internal planning layers**.

For the authoritative reconciliation, start here:

1. **[docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)** — what the code materially contains now.
2. **[docs/IMPLEMENTATION_MAP.md](docs/IMPLEMENTATION_MAP.md)** — module-by-module classification and exact code caveats.
3. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the target harness-first architecture.
4. **[docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md)** — current → target gaps.
5. **[docs/ROADMAP.md](docs/ROADMAP.md)** — architecture-completion sequence and exit criteria.
6. **[docs/README.md](docs/README.md)** — canonical documentation index and maintenance rules.

## Architecture in one diagram

```text
User goal
   ↓
Intelligent external harness
   ↓
DIGIMON MCP / capability facade
   ↓
Typed retrieval/build/analysis capabilities
   ↓
Resource + prerequisite + evidence contracts
   ↓
Graphs | VDBs | chunks | communities | matrices | tables/vectors
   ↓
Original source material
```

The **harness owns adaptive reasoning**: interpreting the goal, deciding whether to decompose it, choosing/ordering tools, observing results, retrying/falling back, and deciding when to stop.

DIGIMON owns the **capability/data/evidence plane**: ingestion, graph construction, typed retrieval/analysis operations, resource/prerequisite facts, source identifiers, and bounded model-assisted transformations when an individual capability requires semantic judgment.

## The operator core

`Core/Operators/registry.py` describes 26 current operators across six categories:

```text
entity        7
relationship  4
chunk         3
subgraph      3
community     2
meta          7
```

`Core/Schema/SlotTypes.py` supplies typed query/entity/relationship/chunk/subgraph/community/score-vector dataflow. `Core/Composition/` validates and executes compositions.

The typed composition layer is substantial, but it is **not yet a closed safety/resource contract**: slot-kind chain discovery does not prove prerequisites are available, static validation is permissive in places, and `OperatorComposer` currently has a best-effort execution path after validation failure. These are explicit Stage-1 architecture gaps rather than hidden limitations.

The 10 named retrieval methods are **reference compositions**, not DIGIMON's identity. A capable harness can compose operators directly or use a named method when useful.

## MCP execution modes

`digimon_mcp_stdio_server.py` currently supports three useful levels:

1. **Individual capabilities/operators** — preferred conceptual mode for capable harnesses.
2. **Reference methods** — execute a known operator composition.
3. **Auto selection** — let a prompt/model choose a reference method as an optional convenience.

Mode 1 defines the target architectural boundary. Modes 2 and 3 remain useful shortcuts, compatibility paths, and later evaluation baselines.

## AoT / GoT is a heuristic, not an executor

Complex questions can benefit from dependency-aware decomposition:

```text
q1: identify the performer who portrayed Corliss Archer in Kiss and Tell
q2: find government positions held by <q1.entity>
q3: determine which position is supported by retrieved source evidence
```

This is guidance, not a mandatory reasoning graph. The harness may merge, reorder, branch, parallelize, revise, or skip the decomposition.

Current heuristic surfaces include:

- [`prompts/decompose_question.yaml`](prompts/decompose_question.yaml)
- [`prompts/synthesize_answers.yaml`](prompts/synthesize_answers.yaml)
- typed meta operators under `Core/Operators/meta/`

The YAML and operator-local policies are aligned in this snapshot, but they are still duplicated prompt sources; consolidating or testing prompt parity is part of the roadmap.

`meta.decompose_question` currently carries sub-question text through the generic `ENTITY_SET` slot. That is a transitional representation, not a reason to introduce a mandatory Graph-of-Thought runtime.

The older `Core/AOT` implementation that programs atomic states and transitions is **legacy project lineage**, not the target reasoning architecture.

## Evidence and provenance

Current typed records already carry useful source identifiers (`source_id` for entities/relationships and `chunk_id` for chunks), and `SlotValue` records producer/metadata. Retrieval paths can return source text and evidence-aware synthesis now preserves available chunk/source markers in its model context.

However, **end-to-end provenance is still Partial**, not finished: the roadmap calls for a universal evidence contract that propagates lineage consistently through retrieval, aggregation, paths, communities and modality conversions.

## Resources and prerequisites

`GraphRAGContext` directly tracks graph and VDB instances, while MCP/server code manages or discovers additional artifacts such as communities, sparse structures and converted data.

That works today, but it is not yet a uniform resource catalog. The target model adds stable resource identity, build/config fingerprints, dependency links, staleness/invalidation, and explicit producer capabilities so the harness can decide whether to reuse, build or fall back.

## Error semantics

Execution errors are currently represented differently across layers: `PipelineExecutor` raises explicit pipeline errors, while some operators return empty/failure-valued slots and build/MCP tools may use status objects or exceptions.

The target is to distinguish **empty evidence, missing prerequisite, invalid plan, provider failure, extraction incompleteness and internal failure** in a machine-actionable way.

## Core retrieval structures

The repository supports multiple structures rather than one fixed GraphRAG pipeline:

- Entity-Relationship graphs;
- Relationship-Keyword graphs;
- hierarchical tree representations;
- balanced hierarchical trees;
- passage graphs;
- vector indexes;
- graph communities and sparse structures;
- graph/table/vector conversion and analysis.

A harness can choose graph reasoning when structure matters and use simpler text/vector paths when it does not.

## Repository map

```text
Core/                       Typed operators, composition, graph/index/provider and legacy agent layers
Config/                     Configuration models and ontology material
Option/                     Runtime/method configuration
prompts/                    Decomposition, synthesis, routing and modality heuristics
Data/                       Example/evaluation datasets
eval/                       Benchmark/evaluation infrastructure
tests/ + test_*.py          Unit/integration/E2E/experimental tests
docs/                       Canonical architecture plus historical/supporting material
examples/                   Example workflows
api.py                      Secondary HTTP/API surface
digimon_cli.py              Transitional CLI using internal planner/orchestrator
digimon_mcp_stdio_server.py Preferred external harness/tool surface
```

The repository intentionally retains historical/experimental code and documents. **File existence does not imply canonical architecture.** See `docs/IMPLEMENTATION_MAP.md` for classification.

## Current development priority

The current order of work is:

1. inventory/audit canonical capabilities, descriptors and MCP parity;
2. make validation strict-vs-best-effort semantics explicit;
3. make prompt ownership/parity explicit;
4. unify resource identities/lifecycle/prerequisites;
5. make provenance/evidence an end-to-end contract;
6. make the harness-first boundary operationally clean;
7. consolidate legacy planners/orchestrators/AoT/MCP layers;
8. normalize cross-modal capabilities;
9. standardize errors/recovery semantics;
10. harden architectural contract tests and CI.

Benchmarking, ablations, router calibration and novelty comparisons are intentionally deferred until those architecture boundaries are stable. Future evaluation questions are preserved in **[docs/FUTURE_EVALUATION_QUESTIONS.md](docs/FUTURE_EVALUATION_QUESTIONS.md)**.

## Getting started

- **[FUNCTIONALITY.md](FUNCTIONALITY.md)** — concise implemented-capability view.
- **[docs/QUICK_START.md](docs/QUICK_START.md)** — current setup/entry points.
- **[docs/PLANNING_SUMMARY.md](docs/PLANNING_SUMMARY.md)** — concise current implementation plan.
- **[docs/AGENT_INTELLIGENCE_ENHANCEMENTS.md](docs/AGENT_INTELLIGENCE_ENHANCEMENTS.md)** — reasoning-policy detail.
- **[docs/adr/002-harness-first-capability-architecture.md](docs/adr/002-harness-first-capability-architecture.md)** — current orchestration decision.

## Public snapshot note

This repository is retained as a public application/architecture snapshot and provenance record. Ongoing private DIGIMON development may contain changes not represented here. The public snapshot remains useful for inspecting the architecture, implementation lineage, experiments and tool design.

For the public portfolio-level project description, see [Brian Mills' portfolio](https://brianmills.dev/portfolio/).

## Lineage

DIGIMON's development includes work derived from and inspired by the GraphRAG research ecosystem, including [JayLZhou/GraphRAG](https://github.com/JayLZhou/GraphRAG) and *In-depth Analysis of Graph-based RAG in a Unified Framework* (Zhou et al., arXiv:2503.04338, 2025).