# Agent Intelligence Architecture

## Current Direction: Harness-First Reasoning

DIGIMON should expose strong, typed retrieval and graph operations to an intelligent agent harness rather than attempting to encode a complete "agent brain" directly in Python.

The architectural boundary is intentional:

- **DIGIMON owns capabilities and constraints**: corpus preparation, graph construction, retrieval operators, modality conversion, resource discovery, evidence/provenance handling, and typed tool contracts.
- **The harness owns adaptive reasoning**: understanding the user goal, deciding whether to decompose it, choosing and sequencing tools, revising a plan after observations, and deciding when enough evidence has been gathered.
- **Prompts may provide heuristics**: Atom-of-Thought (AoT), Graph-of-Thought (GoT), ReAct-style planning, modality selection, and method profiles can help the harness reason, but they should not become a rigid state machine that replaces model intelligence.

This direction avoids over-programming the reasoning policy while keeping the system inspectable and grounded.

## AoT / GoT as a Heuristic, Not an Executor

A useful decomposition should expose information dependencies without requiring DIGIMON to precompute the entire reasoning path.

For example:

```text
q1: identify the performer who portrayed Corliss Archer in Kiss and Tell
q2: find government positions held by {{q1.entity}}
q3: determine which position is supported by the retrieved source evidence
```

The important property is not the exact three-step sequence. It is that `q2` depends on a discovery from `q1`, while `q3` is an evidence-resolution step. An intelligent harness may instead resolve the entity and office in one retrieval operation, branch into multiple candidate entities, run independent searches in parallel, or abandon this decomposition if a more direct route is available.

Therefore the decomposition prompt should:

1. suggest the smallest useful set of sub-goals;
2. distinguish independent work from dependency-linked work;
3. make dependencies explicit when useful;
4. avoid inventing intermediate answers;
5. avoid prescribing specific tools unless the harness explicitly asks for a tool-level plan;
6. permit revision, merging, reordering, parallelization, and early termination.

The prompt in `prompts/decompose_question.yaml` follows this policy while retaining a simple JSON-array interface.

## Relationship to the Legacy `Core/AOT` Code

The repository contains an earlier `Core/AOT` implementation that represents atomic states, dependencies, transition probabilities, and heuristic entity/relationship/action extraction in code. That work remains useful as project history and as a source of ideas, but it should not be treated as the required reasoning architecture for the current harness-first direction.

The preferred architecture is lighter:

```text
User goal
   ↓
Intelligent harness
   ├─ optionally applies AoT/GoT decomposition heuristic
   ├─ inspects available DIGIMON resources/tools
   ├─ chooses actions adaptively
   ├─ observes results and revises
   └─ synthesizes an evidence-grounded answer
          ↓
DIGIMON typed tools + data/resource layer
```

The harness can reason over a graph of sub-goals without DIGIMON itself maintaining a mandatory graph-of-thought runtime.

## Evidence-Aware Synthesis

Answer synthesis is part of the architectural boundary. The synthesis layer should not erase the distinction between retrieval and inference.

The current synthesis heuristic should:

- preserve source/citation/provenance markers when available;
- only assert claims supported by retrieved evidence or explicit sub-results;
- expose material conflicts rather than silently resolving them;
- identify unresolved dependencies;
- avoid treating missing evidence as proof that a claim is false;
- avoid manufacturing confidence scores that have not been calibrated.

This keeps the final response grounded even when the harness takes an adaptive path through the available tools.

## Tool and Method Selection

DIGIMON can provide method profiles and lightweight routing heuristics, but the harness should remain free to override them based on observations and resource availability.

Useful guidance includes:

- graph operations for relationships, paths, communities, and multi-hop structure;
- table operations for aggregation, filtering, counts, and explicit comparison;
- vector operations for semantic similarity, nearest neighbors, clustering, and discovery;
- cross-modal workflows when the question spans more than one analytical form.

These mappings are priors, not hard rules. The agent may start with vector/entity discovery and then move into graph traversal, or retrieve text first to determine whether graph reasoning is warranted.

## Architectural Priorities

The current priority is to finish and clarify the architecture rather than optimize benchmark scores or claim novelty.

Near-term work should emphasize:

- stable typed tool contracts;
- clear data flow between graph, text, vector, community, and structured representations;
- resource discovery and prerequisite handling;
- robust provenance from graph entities/relationships back to source text;
- prompt heuristics that support intelligent harnesses without replacing them;
- predictable failure behavior when resources or evidence are missing;
- concise documentation of the intended execution model.

Benchmarking, router calibration, ablations, and novelty comparisons are valuable later-stage validation tasks and are documented separately in `docs/FUTURE_EVALUATION_QUESTIONS.md`.

## Design Principle

> Program the capabilities, contracts, and evidence boundaries. Prompt useful reasoning heuristics. Let the harness remain intelligent.
