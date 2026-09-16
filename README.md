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

The remaining architecture work is not “invent an agent brain.” It is to make the existing capability system more coherent: **uniform resource lifecycle, prerequisites, provenance/evidence propagation, capability/MCP parity, errors/recovery, and cleanup of legacy internal planning layers**.

For the authoritative status, target design, gaps and plan, start here:

1. **[docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)** — what the code materially contains now.
2. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the target harness-first architecture.
3. **[docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md)** — current → target gaps.
4. **[docs/ROADMAP.md](docs/ROADMAP.md)** — architecture-completion sequence and exit criteria.
5. **[docs/README.md](docs/README.md)** — canonical documentation index and status vocabulary.

## Architecture in one diagram

```text
User goal
   ↓
Intelligent external harness
   ↓
DIGIMON MCP / capability facade
   ↓
Typed operators + build/analysis/conversion capabilities
   ↓
Resource/prerequisite/evidence layer
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

The current prompts live in:

- [`prompts/decompose_question.yaml`](prompts/decompose_question.yaml)
- [`prompts/synthesize_answers.yaml`](prompts/synthesize_answers.yaml)

The older `Core/AOT` implementation that programs atomic states and transitions is **legacy project lineage**, not the target reasoning architecture.

## Evidence and provenance

The current typed records already carry useful source identifiers (`source_id` for entities/relationships and `chunk_id` for chunks), and retrieval paths can return source text. The synthesis prompt is designed to preserve supplied provenance and expose conflicts/unresolved dependencies.

However, **end-to-end provenance is still Partial**, not finished: the roadmap calls for a universal evidence contract that propagates lineage consistently through retrieval, aggregation, paths, communities and modality conversions.

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

The repository intentionally retains historical/experimental code and documents. **File existence does not imply canonical architecture.** See `docs/CURRENT_STATE.md` for classifications.

## Current development priority

The current order of work is:

1. stabilize the canonical capability contract;
2. unify resource identities/lifecycle/prerequisites;
3. make provenance/evidence an end-to-end contract;
4. make the harness-first boundary operationally clean;
5. consolidate legacy planners/orchestrators/AoT code;
6. normalize cross-modal capabilities;
7. standardize errors/recovery semantics;
8. harden architectural contract tests and CI.

Benchmarking, ablations, router calibration and novelty comparisons are intentionally deferred until those architecture boundaries are stable. Future evaluation questions are preserved in **[docs/FUTURE_EVALUATION_QUESTIONS.md](docs/FUTURE_EVALUATION_QUESTIONS.md)**.

## Getting started

- **[FUNCTIONALITY.md](FUNCTIONALITY.md)** — concise implemented-capability view.
- **[docs/QUICK_START.md](docs/QUICK_START.md)** — current setup/entry points.
- **[docs/AGENT_INTELLIGENCE_ENHANCEMENTS.md](docs/AGENT_INTELLIGENCE_ENHANCEMENTS.md)** — reasoning-policy detail.
- **[docs/adr/002-harness-first-capability-architecture.md](docs/adr/002-harness-first-capability-architecture.md)** — current orchestration decision.

## Public snapshot note

This repository is retained as a public application/architecture snapshot and provenance record. Ongoing private DIGIMON development may contain changes not represented here. The public snapshot remains useful for inspecting the architecture, implementation lineage, experiments and tool design.

For the public portfolio-level project description, see [Brian Mills' portfolio](https://brianmills.dev/portfolio/).

## Lineage

DIGIMON's development includes work derived from and inspired by the GraphRAG research ecosystem, including [JayLZhou/GraphRAG](https://github.com/JayLZhou/GraphRAG) and *In-depth Analysis of Graph-based RAG in a Unified Framework* (Zhou et al., arXiv:2503.04338, 2025).
