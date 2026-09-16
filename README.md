# DIGIMON: Composable Knowledge-Graph RAG

DIGIMON is a research and application architecture for turning document collections into reusable **text, vector, graph, community, and structured retrieval resources** that an intelligent agent harness can compose to answer complex questions.

The central idea is simple:

> **Program the capabilities, contracts, and evidence boundaries. Give the harness useful reasoning heuristics. Let the harness remain intelligent.**

Rather than forcing every question through one fixed GraphRAG pipeline—or encoding a complete agent brain as a hand-built state machine—DIGIMON exposes typed operations that a capable harness can select, sequence, revise, and combine as the question requires.

## What the system does

A typical workflow is:

```text
Documents
   ↓
Corpus preparation
   ↓
Graph / vector / text / community / structured resources
   ↓
Intelligent harness chooses and composes retrieval operations
   ↓
Source evidence + graph structure
   ↓
Evidence-aware answer synthesis
```

Depending on the question, the harness may use semantic entity discovery, direct text retrieval, one-hop relationships, multi-hop paths, Personalized PageRank, community structure, table-style aggregation, or a cross-modal combination. It can also decide that graph reasoning is unnecessary.

See **[FUNCTIONALITY.md](FUNCTIONALITY.md)** for the capability-level overview.

## Harness-first agent architecture

DIGIMON deliberately separates **reasoning policy** from **retrieval capability**.

### DIGIMON owns

- corpus ingestion and normalization;
- graph construction and graph resources;
- vector indexes and semantic search;
- entity, relationship, subgraph, path, and community operations;
- structured/table and cross-modal operations where available;
- resource discovery and prerequisite handling;
- typed tool contracts;
- links from retrieved graph evidence back to source text;
- evidence-aware synthesis constraints.

### The intelligent harness owns

- interpreting the user's goal;
- deciding whether decomposition is useful;
- selecting and sequencing tools;
- pursuing independent branches in parallel when useful;
- revising the approach after observations;
- stopping when enough evidence has been gathered;
- deciding how much reasoning structure is actually necessary.

This keeps DIGIMON useful across different capable agent harnesses instead of coupling the architecture to one programmed planner.

## AoT / GoT as a soft reasoning heuristic

Complex questions often contain dependencies that should not be flattened into falsely independent searches.

For example:

```text
q1: identify the performer who portrayed Corliss Archer in Kiss and Tell
q2: find government positions held by {{q1.entity}}
q3: determine which position is supported by the retrieved source evidence
```

That dependency structure is useful, but it is **not a mandatory execution graph**. A capable harness can merge steps, branch into candidates, run independent work concurrently, reorder the plan, or bypass the decomposition if a more direct retrieval route appears.

The prompt in [`prompts/decompose_question.yaml`](prompts/decompose_question.yaml) implements this as a lightweight Atom-of-Thought / Graph-of-Thought heuristic while retaining a simple interface. The architectural rationale is documented in **[docs/AGENT_INTELLIGENCE_ENHANCEMENTS.md](docs/AGENT_INTELLIGENCE_ENHANCEMENTS.md)**.

The repository also contains an earlier `Core/AOT` implementation that encodes atomic states and transitions directly in code. It is retained as project history and an experimental implementation, but it is not the required reasoning architecture for the current harness-first direction.

## Core capabilities

### Corpus preparation

Convert `.txt`, `.md`, `.json`, `.jsonl`, `.csv`, and `.pdf` document collections into DIGIMON corpus resources.

### Graph construction

The project supports multiple retrieval structures, including:

- **Entity-Relationship graphs** for named entities and explicit relationships;
- **Relationship-Keyword graphs** for richer relationship retrieval;
- **Passage graphs** connecting source passages through shared entities;
- **hierarchical summary trees** for multilevel retrieval;
- associated vector indexes and community structures.

### Retrieval and analysis

Representative operations include:

- semantic entity search;
- direct graph-neighbor and relationship lookup;
- Personalized PageRank;
- K-hop path and connected-subgraph retrieval;
- community detection/access;
- source chunk retrieval;
- graph statistics and visualization/export;
- resource discovery;
- graph/vector/table cross-modal workflows.

The repository includes method configurations and operator compositions inspired by or implementing ideas from GraphRAG-family systems such as ToG, HippoRAG, LightRAG, RAPTOR, DALK, KGP, and related approaches.

## Evidence-aware synthesis

Retrieval is only useful if the final response preserves what the system actually knows.

[`prompts/synthesize_answers.yaml`](prompts/synthesize_answers.yaml) instructs synthesis to:

- only make claims supported by supplied evidence/sub-results;
- preserve citation/provenance markers when present;
- distinguish retrieved evidence from inference;
- surface material conflicts;
- expose unresolved information dependencies;
- avoid treating missing evidence as proof that a claim is false;
- avoid inventing certainty or unsupported bridges between facts.

## Example question shape

Multi-hop questions such as the following are useful architectural tests:

> What government position was held by the woman who portrayed Corliss Archer in the film *Kiss and Tell*?

A harness might discover the performer, use that entity as the input to another retrieval step, retrieve source evidence for government roles, and then synthesize only the position supported by the corpus. Another capable harness may solve the same question with a different sequence. DIGIMON's responsibility is to make the necessary capabilities and evidence available without dictating one universal route.

## Repository map

```text
Core/                       Core graph, retrieval, agent-tool, and provider modules
Config/                     Configuration models and ontology material
Option/                     Runtime and method configuration
prompts/                    Harness-facing reasoning/routing/synthesis heuristics
Data/                       Example and evaluation datasets
eval/                       Benchmark/evaluation infrastructure
tests/ + test_*.py          Unit/integration/end-to-end and experimental tests
docs/                       Architecture, integration, planning, and usage documentation
examples/                   Example workflows
api.py                      API surface
digimon_cli.py              CLI surface
digimon_mcp_stdio_server.py MCP/tool surface
```

This repository reflects an active research lineage and contains experimental and historical material in addition to the current architectural direction.

## Current development priority

The immediate priority is **finishing and clarifying the architecture**, especially:

- stable typed tool contracts;
- resource discovery and prerequisite handling;
- clean data flow among graph, text, vector, table, and community representations;
- provenance from graph evidence to original source text;
- robust harness/tool interaction;
- useful reasoning prompts that guide without replacing harness intelligence;
- predictable behavior when evidence or required resources are missing.

Benchmarking, ablations, novelty comparisons, router calibration, and broader question-class evaluation are intentionally deferred until the architecture is stable. The questions worth revisiting later are preserved in **[docs/FUTURE_EVALUATION_QUESTIONS.md](docs/FUTURE_EVALUATION_QUESTIONS.md)**.

## Getting started

The repository includes minimal and full dependency sets plus API, CLI, and MCP-style access surfaces. Start with **[docs/QUICK_START.md](docs/QUICK_START.md)** and **[FUNCTIONALITY.md](FUNCTIONALITY.md)**.

Representative configuration lives under `Option/`, and method-specific configurations live under `Option/Method/`.

## Repository status

**Public snapshot status (September 2026):** this repository is retained as a public application/architecture snapshot and provenance record. Canonical ongoing DIGIMON development has moved to a maintained private repository. The public snapshot remains useful for inspecting the project's architecture, implementation lineage, experiments, and agent-tool design, but it should not be assumed to contain every current private implementation detail.

For the public portfolio-level project description, see [Brian Mills' portfolio](https://brianmills.dev/portfolio/).

## Lineage and acknowledgement

DIGIMON's development includes work derived from and inspired by the GraphRAG research ecosystem. The original repository lineage referenced by this project includes [JayLZhou/GraphRAG](https://github.com/JayLZhou/GraphRAG) and the paper:

> *In-depth Analysis of Graph-based RAG in a Unified Framework* — Zhou et al., arXiv:2503.04338 (2025).

The current DIGIMON direction focuses on exposing graph and retrieval capabilities as composable tools for intelligent harnesses rather than treating one fixed GraphRAG method as the system itself.
