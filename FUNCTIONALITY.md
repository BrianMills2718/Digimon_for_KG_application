# DIGIMON KG-RAG: What It Does

## One-liner

DIGIMON gives an intelligent agent harness typed tools for turning documents into graph, vector, text, community, and structured retrieval resources, then using those resources to answer questions with source-grounded evidence.

## Architectural Principle

DIGIMON does **not** need to hard-code a complete agent brain.

The system is split deliberately:

- **DIGIMON provides capabilities and contracts**: ingestion, graph construction, indexing, retrieval, analysis, resource discovery, provenance, and typed tool interfaces.
- **The harness provides adaptive reasoning**: it interprets the user goal, decides whether decomposition is useful, chooses tools, observes results, revises its approach, and stops when it has enough evidence.
- **Prompts provide soft heuristics**: AoT/GoT-style decomposition, modality selection, and retrieval-method profiles can help the harness reason without becoming a mandatory execution graph.

A harness may follow a suggested dependency chain, pursue independent branches in parallel, skip unnecessary steps, or discover a better route after seeing intermediate evidence.

See `docs/AGENT_INTELLIGENCE_ENHANCEMENTS.md` for the current reasoning architecture.

## Core Flow

Given a goal and a folder of documents (`.txt`, `.md`, `.json`, `.jsonl`, `.csv`, `.pdf`), an MCP-capable or tool-capable harness can use DIGIMON to:

1. **Ingest** documents into a structured corpus.
2. **Build one or more retrieval structures** — entity/relationship graphs, passage graphs, hierarchical trees, vector indexes, or community structures.
3. **Inspect available resources** and choose an appropriate retrieval strategy.
4. **Retrieve evidence** through semantic search, graph traversal, relationship retrieval, source-chunk lookup, or cross-modal operations.
5. **Adapt** after intermediate observations rather than committing to a fixed pipeline in advance.
6. **Synthesize** a final answer while preserving evidence/provenance boundaries.

The user states what they want to know; the harness determines how to use the available DIGIMON capabilities.

## Dependency-Aware Reasoning Heuristic

For complex questions, the harness may use a lightweight Atom-of-Thought / Graph-of-Thought decomposition heuristic. The heuristic exposes dependencies without programming a fixed reasoning runtime.

Example:

```text
q1: identify the performer who portrayed Corliss Archer in Kiss and Tell
q2: find government positions held by {{q1.entity}}
q3: determine which position is supported by the retrieved source evidence
```

This is guidance, not an execution contract. A capable harness can merge steps, branch, parallelize, revise, or bypass the decomposition entirely.

The corresponding prompt is `prompts/decompose_question.yaml`.

## What's in the Toolbox

### Corpus Preparation

- **corpus_prepare** — Turns a directory of documents into a structured corpus. Supports `.txt`, `.md`, `.json`, `.jsonl`, `.csv`, and `.pdf`, with text/title field detection for structured formats.

### Graph Construction

Graph-building tools can accept an `input_directory`; when appropriate, corpus preparation can happen before graph construction.

- **graph_build_er** — Entity-Relationship graph for named entities and explicit relationships.
- **graph_build_rk** — Relationship-Keyword graph with richer edge descriptions/keywords.
- **graph_build_tree** — Hierarchical summary tree (RAPTOR-style).
- **graph_build_tree_balanced** — Balanced hierarchical tree using K-Means-style partitioning.
- **graph_build_passage** — Passage graph linking text passages through shared entities.

### Search and Retrieval

- **entity_vdb_build** — Build a vector index over graph entities.
- **entity_vdb_search** — Find entities relevant to a natural-language query.
- **entity_onehop** — Retrieve direct graph neighbors.
- **entity_ppr** — Personalized PageRank over seed entities.
- **relationship_onehop** — Retrieve relationships attached to entities.
- **chunk_get_text** — Retrieve original source text associated with graph evidence.
- **subgraph/path operators** — Explore multi-hop paths or compact connected subgraphs when relational structure matters.
- **community operators** — Work with graph communities and higher-level structure.

### Analysis and Resource Discovery

- **graph_analyze** — Graph statistics and structural analysis.
- **graph_visualize** — Export graph structure for inspection or visualization.
- **list_available_resources** — Inspect graphs, indexes, and other artifacts currently available to the harness.

### Cross-Modal Analysis

DIGIMON can treat graph, table, vector, and text representations as complementary tools rather than mutually exclusive modes. A harness can move between them when a question calls for aggregation, similarity, relationship traversal, or source verification.

## Typical Session

```text
User: "I have a folder of articles about defense contracting.
       Who are the key players and how are they connected?"

Harness:
  1. Inspects available DIGIMON resources.
  2. Builds or reuses an entity/relationship graph.
  3. Builds or reuses an entity vector index if semantic discovery is useful.
  4. Searches for relevant entities.
  5. Traverses relationships / graph neighborhoods for connection structure.
  6. Retrieves source chunks supporting the important entities and edges.
  7. Revises or expands retrieval if evidence is incomplete.
  8. Produces an answer grounded in the retrieved source material.
```

The exact sequence is intentionally not fixed. The harness chooses the path that fits the question and the resources already available.

## Evidence-Aware Synthesis

The final synthesis step should preserve rather than erase evidence boundaries. The synthesis heuristic in `prompts/synthesize_answers.yaml` is designed to:

- retain provenance/citation markers when supplied;
- distinguish retrieved evidence from inferred conclusions;
- surface material conflicts;
- identify unresolved information dependencies;
- avoid converting missing evidence into a false negative claim;
- avoid invented certainty or unsupported bridges between facts.

## Current Priority

The current goal is to finish and clarify the architecture:

- typed tool contracts;
- reliable resource discovery and prerequisite handling;
- clean data flow among graph, text, vector, table, and community representations;
- provenance from graph evidence back to original text;
- robust harness interaction;
- useful but non-prescriptive reasoning prompts.

Benchmarking, method ablations, novelty comparisons, router calibration, and question-class evaluation are intentionally deferred. The questions to revisit later are preserved in `docs/FUTURE_EVALUATION_QUESTIONS.md`.
