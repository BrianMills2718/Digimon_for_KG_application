# DIGIMON Architecture Completion Roadmap

**Updated:** 2026-09-16  
**Scope:** finish and consolidate the architecture before making benchmarking, novelty, production-scale optimization, or additional UI surfaces the primary focus.

This roadmap is ordered by dependency, not by calendar estimates.

## Guiding principle

> **Make capabilities, resources, prerequisites, errors and evidence explicit enough that a capable harness can reason with DIGIMON without DIGIMON hard-coding the harness's reasoning policy.**

## Stage 0 — Canonical documentation baseline

**Goal:** establish one accurate description of code reality and target architecture.

### Work

- maintain `docs/CURRENT_STATE.md` as code-truth status;
- maintain `docs/ARCHITECTURE.md` as target design;
- maintain `docs/GAP_ANALYSIS.md` as the distance between them;
- use this roadmap for active architectural sequencing;
- keep `README.md`, `FUNCTIONALITY.md`, `AGENTS.md` and `CLAUDE.md` aligned with the canonical set;
- mark older checkpoint/UKRF/MCP/AoT plans historical rather than letting them remain competing instructions.

### Exit criteria

- a contributor can identify the canonical docs in under a minute;
- no active agent instruction file tells the agent to follow a superseded checkpoint plan;
- old plans are clearly labeled historical/superseded;
- architecture terminology uses the same `Implemented / Partial / Legacy / Planned` meanings.

## Stage 1 — Stabilize the capability contract

**Goal:** make the typed operator system the unequivocal canonical capability core.

### Work

- inventory the 26 registered operators and all additional MCP tools;
- define a common capability descriptor or a documented mapping between operator descriptors and non-operator tools;
- standardize names, input/output schemas, prerequisite descriptions and error expectations;
- ensure external discovery can tell which capability is deterministic, model-assisted, destructive/building, expensive, or lossy;
- add contract tests for registry metadata and MCP exposure.

### Exit criteria

- every canonical harness-facing capability has a machine-readable description;
- the harness can discover required inputs/produced outputs without reading implementation source;
- MCP/operator metadata cannot silently drift without a test failure;
- reference methods are explicitly compositions of capabilities, not a separate architectural universe.

## Stage 2 — Unify resources and prerequisites

**Goal:** let the harness inspect what exists, what is missing, and how artifacts depend on each other.

### Work

- introduce a `ResourceDescriptor`/resource-catalog abstraction;
- represent corpus, graph, VDB, community, sparse-matrix and converted table/vector artifacts consistently;
- define stable resource IDs/namespaces and dataset association;
- record producer/config/build fingerprints;
- record prerequisite/dependency links;
- expose resource state through the canonical harness surface;
- make `auto_build` use the same explicit prerequisite model rather than hidden special cases;
- define reuse, rebuild and invalidation behavior.

### Exit criteria

For any capability request, the harness can determine:

1. which resources it requires;
2. whether they already exist;
3. whether they are compatible/current;
4. which capability can build them;
5. whether it should build, reuse or choose a fallback.

## Stage 3 — Make provenance an end-to-end contract

**Goal:** preserve source lineage through retrieval and composition.

### Work

- define a first-class evidence/provenance record;
- connect entity/relationship `source_id` and chunk IDs to source-document metadata;
- propagate evidence references through subgraphs, communities, scoring/reranking and aggregation;
- define provenance behavior for cross-modal conversions;
- mark derived/inferred outputs separately from directly retrieved facts;
- make answer synthesis consume structured evidence rather than relying only on free-form sub-answer text;
- preserve conflicting evidence rather than silently collapsing it.

### Exit criteria

- a material final claim can be traced to one or more source chunks/documents;
- relationship/path evidence can identify the source material that justified important edges;
- aggregation/conversion states whether lineage was preserved, combined or lost;
- missing evidence is distinguishable from negative evidence;
- synthesis can surface unresolved/conflicting evidence using structured inputs.

## Stage 4 — Make the harness-first boundary operationally clean

**Goal:** ensure the preferred execution path matches the documented architecture.

### Work

- treat stdio MCP/capability access as the canonical external orchestration surface;
- keep individual-operator composition as the primary conceptual mode;
- retain `execute_method` and `auto_compose` as optional conveniences;
- review internal LLM-assisted operators and document their bounded role;
- adapt CLI/API paths to use the canonical capability layer where practical, or label them compatibility/experimental;
- remove assumptions that the internal `PlanningAgent` must control all queries;
- ensure AoT/GoT prompts remain advisory and do not require a new executor.

### Exit criteria

- a capable external harness can complete the supported build/retrieve/evidence flow without invoking the legacy internal planner;
- optional internal/reference execution remains available without defining the core architecture;
- no canonical document describes a mandatory two-brain or programmed-cognitive architecture;
- CLI/API documentation accurately identifies whether each entry point is canonical, compatibility or experimental.

## Stage 5 — Consolidate legacy and duplicate architecture

**Goal:** reduce ambiguity and maintenance cost without discarding useful lineage prematurely.

### Work

- identify live callers of `Core/AgentBrain`, each `Core/AgentOrchestrator` variant and `Core/AOT`;
- mark unused modules deprecated and remove them when safe;
- move historical planning/report material under an archive convention over time;
- stop maintaining duplicate tool registries/planners when the operator registry can serve the need;
- remove generated/cache/vendor artifacts from source control where safe;
- reconcile dependency/environment files around supported paths.

### Exit criteria

- every major orchestration module is classified as canonical, compatibility, experimental or legacy;
- unused orchestration/AoT code is removed or clearly isolated;
- new contributors are not presented with multiple equally authoritative architectures;
- the root/docs directories no longer use old status trackers as active instructions.

## Stage 6 — Normalize cross-modal capabilities

**Goal:** bring graph/table/vector transformations under the same capability/resource/evidence rules.

### Work

- describe conversion operations with capability metadata;
- represent converted outputs as resources with fingerprints and schemas;
- record lossiness/transformation parameters;
- preserve provenance mappings where feasible;
- use explicit typed adapters rather than ad hoc payload conventions at harness boundaries;
- keep modality-selection prompts advisory.

### Exit criteria

- the harness can discover conversion paths and their input/output schemas;
- converted artifacts participate in resource discovery/lifecycle;
- lossy conversions are explicitly marked;
- source/evidence lineage survives where technically possible.

## Stage 7 — Standardize failure and recovery semantics

**Goal:** make tool failures actionable to an intelligent harness.

### Work

- define error categories for missing resources/prerequisites, incompatible inputs, empty retrieval, provider failure, extraction incompleteness, unsupported conversion and internal failure;
- standardize result/error envelopes or MCP exceptions;
- distinguish zero results from execution failure;
- provide structured recovery hints only where they reflect capability facts, not hard-coded reasoning policy.

### Exit criteria

- the harness can programmatically distinguish build/retry/fallback/reformulate/stop cases;
- canonical contract tests cover error classes;
- empty or incomplete KG results do not automatically become negative factual conclusions.

## Stage 8 — Architectural reliability and CI

**Goal:** prove the contracts and boundaries remain stable as implementation changes.

### Work

Create a test matrix around:

- operator descriptor/slot compatibility;
- custom composition execution;
- MCP discovery/execution parity;
- resource registration/prerequisite/invalidation;
- provenance propagation;
- graph-build → retrieve → evidence E2E flow;
- cross-modal conversion contracts;
- failure/recovery semantics;
- legacy-entry-point compatibility where intentionally retained.

Tighten CI so canonical contract tests are blocking. Keep live-provider/expensive tests separately classified.

### Exit criteria

- architectural contract regressions fail CI;
- optional provider/LLM tests are clearly separated from deterministic core tests;
- supported Python/package/build paths are documented and exercised;
- the status docs can reference test categories instead of anecdotal past runs.

## Stage 9 — Incremental and temporal/conflict semantics

**Goal:** add more sophisticated lifecycle/evidence behavior only after the underlying resource/provenance model exists.

### Work

- define supported incremental corpus/graph/index updates;
- specify entity identity behavior across updates;
- propagate invalidation to communities/matrices/VDBs;
- add assertion validity/time metadata where required;
- represent source disagreement explicitly;
- define query-time filtering/selection behavior as capabilities, not hidden global policy.

### Exit criteria

- new documents can be incorporated with documented consequences for derived artifacts;
- stale resources are detectable;
- conflicting/time-bounded claims can coexist and be surfaced to the harness.

## Stage 10 — Later evaluation and research validation

**Goal:** measure the stabilized architecture rather than allowing a benchmark to define it.

This stage is intentionally deferred until the earlier architecture exit criteria are substantially met.

Use [FUTURE_EVALUATION_QUESTIONS.md](FUTURE_EVALUATION_QUESTIONS.md) to evaluate:

- when KG structure helps versus lexical/vector retrieval;
- adaptive composition versus fixed methods;
- extraction/entity-resolution error propagation;
- provenance quality;
- routing quality and calibration;
- path-expansion control;
- latency/token/quality tradeoffs;
- incomplete-graph fallback behavior;
- incremental update behavior.

The existing evaluation infrastructure should be extended rather than used as a reason to redesign the capability architecture prematurely.

## What not to add before it solves a documented gap

Avoid expanding scope with:

- another general-purpose internal planner;
- another orchestrator variant;
- a mandatory Graph-of-Thought/AoT executor;
- another UI shell;
- a multi-agent coordination framework without a concrete capability need;
- benchmark-specific special cases in the core resource model;
- self-reported confidence as a substitute for evidence/provenance.

## Immediate implementation focus

If work begins directly from this roadmap, the next code-oriented sequence is:

1. capability inventory/parity;
2. resource descriptor/catalog;
3. prerequisite/lifecycle semantics;
4. evidence/provenance contract;
5. harness-first entry-point cleanup;
6. legacy consolidation;
7. contract tests/CI hardening.

That sequence closes the largest architectural gaps without programming the agent's reasoning for it.