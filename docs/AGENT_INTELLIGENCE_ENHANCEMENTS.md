# Agent Intelligence Architecture

**Updated:** 2026-09-16

## Current Direction: Harness-First Reasoning

DIGIMON should expose strong, typed retrieval and graph operations to an intelligent agent harness rather than attempting to encode a complete "agent brain" directly in Python.

The architectural boundary is intentional:

- **DIGIMON owns capabilities and constraints**: corpus preparation, graph construction, retrieval operators, modality conversion, resource discovery, evidence/provenance handling, and typed tool contracts.
- **The harness owns adaptive reasoning**: understanding the user goal, deciding whether to decompose it, choosing and sequencing tools, revising a plan after observations, and deciding when enough evidence has been gathered.
- **Prompts may provide heuristics**: Atom-of-Thought (AoT), Graph-of-Thought (GoT), ReAct-style planning, modality selection, and method profiles can help the harness reason, but they should not become a rigid state machine that replaces model intelligence.

This direction avoids over-programming the reasoning policy while keeping the system inspectable and grounded.

For the exact current implementation status, including prompt duplication and the transitional sub-question slot representation, see `IMPLEMENTATION_MAP.md`.

## AoT / GoT as a Heuristic, Not an Executor

A useful decomposition should expose information dependencies without requiring DIGIMON to precompute the entire reasoning path.

For example:

```text
q1: identify the performer who portrayed Corliss Archer in Kiss and Tell
q2: find government positions held by <q1.entity>
q3: determine which position is supported by the retrieved source evidence
```

The important property is not the exact three-step sequence. It is that `q2` depends on a discovery from `q1`, while `q3` is an evidence-resolution step. An intelligent harness may instead resolve the entity and office in one retrieval operation, branch into multiple candidate entities, run independent searches in parallel, or abandon this decomposition if a more direct route is available.

Therefore decomposition guidance should:

1. suggest the smallest useful set of sub-goals;
2. distinguish independent work from dependency-linked work;
3. make dependencies explicit when useful;
4. avoid inventing intermediate answers;
5. avoid prescribing specific tools unless the harness explicitly asks for a tool-level plan;
6. permit revision, merging, reordering, branching, parallelization, and early termination.

The current policy is implemented in both:

- `prompts/decompose_question.yaml`; and
- `Core/Operators/meta/decompose_question.py`.

The typed meta operator now uses the same dependency-aware advisory policy rather than the older “independent sub-questions” instruction.

### Current implementation limitation: prompt duplication

The YAML prompt and operator-local prompt are **not yet one runtime source of truth**. The meta operator currently contains its own aligned prompt text rather than loading the YAML template.

That means a future edit to only one surface could reintroduce drift. The architecture roadmap therefore calls for either centralized prompt loading or explicit prompt-source mapping plus semantic parity tests.

### Current implementation limitation: generic sub-question type

`meta.decompose_question` currently returns suggested sub-questions inside the existing typed operator system by storing each question in `EntityRecord.entity_name` and returning an `ENTITY_SET`.

That is a transitional compatibility representation. It does **not** mean a sub-question is conceptually an entity.

A future cleanup may justify a general text/task-list record if it is useful across multiple capabilities. Do not introduce a formal Graph-of-Thought dependency DAG solely to fix this type mismatch. A DAG should exist only if a concrete capability needs it—for example scheduling, resumability, caching, provenance, or auditing.

## Relationship to the Legacy `Core/AOT` Code

The repository contains an earlier `Core/AOT` implementation that represents atomic states, dependencies, transition probabilities, and heuristic entity/relationship/action extraction directly in code.

That work remains useful as project history and as a source of ideas, but it should not be treated as the required reasoning architecture for the current harness-first direction.

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
DIGIMON typed capabilities + resource/evidence layer
```

The harness can reason over a graph of sub-goals without DIGIMON itself maintaining a mandatory graph-of-thought runtime.

## Evidence-Aware Synthesis

Answer synthesis is part of the architectural boundary. The synthesis layer should not erase the distinction between retrieval and inference.

The current synthesis policy is implemented in both:

- `prompts/synthesize_answers.yaml`; and
- `Core/Operators/meta/synthesize_answers.py`.

The typed synthesis operator now includes available chunk/source markers in its model context and instructs the model to:

- preserve source/citation/provenance markers when available;
- only assert claims supported by supplied evidence;
- expose material conflicts rather than silently resolving them;
- identify unresolved dependencies;
- avoid inventing a bridge between otherwise unsupported facts;
- avoid treating missing evidence as proof that a claim is false;
- avoid manufacturing confidence scores that the evidence does not justify.

This is an improvement in synthesis behavior, but it is **not yet the end-to-end provenance architecture**. The operator can only preserve evidence markers that upstream operations actually retained. A universal structured evidence/assertion contract remains planned.

## Tool and Method Selection

DIGIMON can provide method profiles and lightweight routing heuristics, but the harness should remain free to override them based on observations and resource availability.

Useful priors include:

- graph operations for relationships, paths, communities, and multi-hop structure;
- table operations for aggregation, filtering, counts, and explicit comparison;
- vector operations for semantic similarity, nearest neighbors, clustering, and discovery;
- cross-modal workflows when the question spans more than one analytical form.

These mappings are priors, not hard rules. The harness may start with vector/entity discovery and then move into graph traversal, retrieve text first to determine whether graph reasoning is warranted, or avoid graph operations entirely.

Reference methods and `auto_compose` are optional conveniences. They should not become a hidden mandatory reasoning policy.

## What DIGIMON Should Program vs What It Should Leave to the Harness

### Program directly in DIGIMON

- typed capability input/output contracts;
- resource prerequisites and producer relationships;
- graph/index/chunk/community operations;
- deterministic compatibility facts;
- machine-readable errors;
- source/evidence lineage;
- bounded semantic transformations inside individual model-assisted capabilities;
- inspectable reference compositions.

### Leave adaptive by default

- whether the user request should be decomposed;
- how many reasoning branches to pursue;
- when to use graph versus vector/text/table retrieval;
- whether to follow an AoT/GoT suggestion literally;
- when to retry, fall back, merge branches, or stop;
- which evidence is sufficient to answer the user's actual goal.

This distinction is the reason the project should invest in stronger contracts rather than another general-purpose internal planner.

## Architectural Priorities

The current priority is to finish and clarify the architecture rather than optimize benchmark scores or claim novelty.

Near-term work should emphasize:

1. capability/descriptor/MCP parity;
2. explicit strict-versus-best-effort composition semantics;
3. prompt source-of-truth/parity;
4. resource discovery, identity, lifecycle and prerequisites;
5. robust provenance from graph entities/relationships back to source text;
6. a clean external-harness execution boundary;
7. consolidation of legacy planner/orchestrator/AoT/MCP layers;
8. predictable machine-actionable failure behavior;
9. contract tests that keep those boundaries stable.

Benchmarking, router calibration, ablations, and novelty comparisons are valuable later-stage validation tasks and are documented separately in `FUTURE_EVALUATION_QUESTIONS.md`.

## Design Principle

> **Program the capabilities, contracts, resources, validation semantics, and evidence boundaries. Prompt useful reasoning heuristics. Let the harness remain intelligent.**