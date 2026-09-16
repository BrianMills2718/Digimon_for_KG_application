# DIGIMON Architecture Completion Roadmap

**Updated:** 2026-09-16  
**Scope:** finish and consolidate the architecture before making benchmarking, novelty, production-scale optimization, or additional UI surfaces the primary focus.

This roadmap is ordered by architectural dependency, not by calendar estimates.

## Guiding principle

> **Make capabilities, resources, prerequisites, validation, errors, prompts, and evidence explicit enough that a capable harness can reason with DIGIMON without DIGIMON hard-coding the harness's reasoning policy.**

## Stage 0 — Canonical documentation baseline

**Goal:** keep one accurate description of code reality and target architecture.

### Work

- maintain `docs/CURRENT_STATE.md` as code-truth status;
- maintain `docs/IMPLEMENTATION_MAP.md` as the module/capability map;
- maintain `docs/ARCHITECTURE.md` as target design;
- maintain `docs/GAP_ANALYSIS.md` as current→target distance;
- use this roadmap for active architectural sequencing;
- keep `README.md`, `FUNCTIONALITY.md`, `AGENTS.md` and `CLAUDE.md` aligned;
- convert high-authority obsolete trackers to explicit historical stubs rather than leaving competing “IN PROGRESS” plans.

### Exit criteria

- a contributor can identify canonical docs in under a minute;
- no active instruction/status file mandates a superseded planner/AoT/WebSocket checkpoint program;
- old plans are clearly labeled historical/superseded;
- `Implemented / Partial / Legacy / Planned` terminology is used consistently;
- current docs distinguish source inspection from fresh runtime certification.

**Current status:** substantially complete; maintain continuously.

## Stage 1 — Stabilize the capability and composition contract

**Goal:** make the typed capability system the unequivocal canonical core and make plan validation semantics unambiguous.

### Work

- inventory all 26 registered operators against their actual implementations;
- inventory additional MCP-facing corpus/build/resource/config/analysis/cross-modal tools;
- define one common capability descriptor model or an explicit mapping for non-operator tools;
- audit every operator descriptor for exact input/output semantics, field requirements, LLM/cost/prerequisite claims and limitations;
- resolve known semantic mismatches such as generic `ENTITY_SET` use for sub-questions and broader rerank behavior than its current descriptor expresses;
- distinguish compatibility **discovery** from proof of **executability**;
- make validation policy caller-visible:
  - strict mode: invalid plan does not execute;
  - explicit best-effort mode: proceed with structured warnings when intentionally requested;
- remove accidental fail-open ambiguity from `OperatorComposer.execute()`;
- decide prompt-source ownership for typed meta operators versus YAML templates and prevent semantic drift;
- add deterministic contract tests for registry metadata, reference plans and MCP parity.

### Exit criteria

- every canonical harness-facing capability has machine-readable metadata or a documented adapter into the canonical model;
- descriptor behavior matches implementation behavior;
- static validation produces structured errors/warnings;
- strict execution cannot silently proceed after failed validation;
- best-effort execution is explicit and distinguishable;
- compatibility/chain discovery does not present resource-incomplete suggestions as validated execution plans;
- decomposition/synthesis prompt execution paths cannot silently drift from their documented policy;
- reference methods are visibly compositions of capabilities, not a separate architecture.

## Stage 2 — Unify resources and prerequisites

**Goal:** let the harness inspect what exists, what is missing, and how derived artifacts depend on each other.

### Work

- introduce a typed `ResourceDescriptor`/resource-catalog abstraction;
- represent corpus, graph, VDB, community, sparse-matrix and converted table/vector artifacts consistently;
- define stable resource IDs/namespaces and dataset association;
- record producer/config/build fingerprints;
- record dependency/prerequisite links;
- expose resource state through the canonical harness surface;
- replace/augment boolean prerequisite flags with explicit resource requirements;
- link missing prerequisites to capabilities that can produce them;
- make `auto_build` a thin convenience policy over explicit resource facts;
- define reuse, rebuild, staleness and invalidation behavior;
- document the supported current stdio session/process scope before considering more complex multi-session state.

### Exit criteria

For any capability request, the harness can determine:

1. which resources it requires;
2. whether compatible resources already exist;
3. whether they are current/stale;
4. which capability can build them;
5. what downstream resources depend on them;
6. whether to build, reuse, rebuild or choose a fallback.

## Stage 3 — Make provenance an end-to-end contract

**Goal:** preserve source lineage through retrieval, transformation and composition.

### Work

- define a first-class evidence/provenance/assertion record;
- connect entity/relationship `source_id` and chunk IDs to stable source-document metadata;
- propagate evidence references through subgraphs, communities, scoring/reranking and aggregation;
- use `SlotValue.metadata` only as an interim bridge where a stronger typed record is not yet available;
- define provenance behavior for graph/table/vector conversions;
- mark derived/inferred outputs separately from directly retrieved assertions;
- represent when lineage has been combined, summarized or lost;
- make answer synthesis consume structured evidence rather than relying only on free-form text;
- preserve conflicting evidence rather than silently collapsing it.

### Exit criteria

- a material final claim can be traced to one or more source chunks/documents;
- important relationship/path evidence can identify source material that justified it;
- aggregation/conversion states whether lineage was preserved, combined or lost;
- missing evidence is distinguishable from negative evidence;
- synthesis can surface unresolved/conflicting evidence using structured inputs.

## Stage 4 — Make the harness-first boundary operationally clean

**Goal:** ensure the preferred execution path matches the documented architecture.

### Work

- treat stdio MCP/capability access as the canonical external orchestration surface;
- keep individual capability composition as the primary conceptual mode;
- retain `execute_method` and `auto_compose` as optional conveniences;
- document bounded internal model-assisted operators clearly;
- adapt CLI/API paths to use the canonical capability layer where practical, or label them compatibility/experimental;
- remove assumptions that `PlanningAgent` must control ordinary queries;
- keep AoT/GoT/ReAct decomposition advisory;
- if sub-question typing is improved, prefer a reusable text/task-list abstraction before introducing a formal reasoning DAG;
- introduce a dependency graph only when a concrete feature such as scheduling, resumability, caching, provenance or auditing requires it.

### Exit criteria

- a capable external harness can complete supported build→retrieve→evidence flows without invoking the legacy internal planner;
- optional internal/reference execution remains available without defining the core architecture;
- no canonical document or entry point description implies a mandatory two-brain/cognitive architecture;
- CLI/API documentation accurately identifies canonical, compatibility and experimental paths.

## Stage 5 — Consolidate legacy and duplicate architecture

**Goal:** reduce ambiguity and maintenance cost without prematurely deleting useful lineage.

### Work

- identify live callers of `Core/AgentBrain`, each `Core/AgentOrchestrator` variant, `Core/AOT`, `Core/Memory`, and older `Core/MCP` components;
- classify each module as canonical, compatibility, experimental or legacy;
- deprecate/remove unused variants when safe;
- retain historical behavior in Git instead of maintaining competing active trackers;
- stop maintaining duplicate tool registries/planners where the canonical descriptor system can serve the requirement;
- remove generated/cache/vendor artifacts from source control where safe;
- reconcile dependency/environment files around supported paths.

### Exit criteria

- every major orchestration/MCP/memory/AoT module is classified;
- unused architecture is removed or clearly isolated;
- new contributors are not presented with several equally authoritative “brains”;
- root/docs directories no longer contain apparently active superseded implementation trackers.

## Stage 6 — Normalize cross-modal capabilities

**Goal:** bring graph/table/vector transformations under the same capability/resource/evidence rules.

### Work

- describe conversion operations with canonical capability metadata;
- represent converted outputs as resources with fingerprints and schemas;
- record transformation parameters/provider and lossiness;
- preserve provenance mappings where feasible;
- use explicit typed adapters rather than ad hoc DataFrame/ndarray/dictionary payload conventions at harness boundaries;
- keep modality-selection prompts advisory.

### Exit criteria

- the harness can discover valid conversion paths and their schemas;
- converted artifacts participate in resource lifecycle/discovery;
- lossy conversions are explicitly marked;
- source/evidence lineage survives where technically possible.

## Stage 7 — Standardize failure and recovery semantics

**Goal:** make tool failures actionable to an intelligent harness.

### Work

- define error categories for missing resources/prerequisites, invalid plan/wiring, incompatible resource/config, empty retrieval, provider failure, likely extraction incompleteness, unsupported conversion and internal failure;
- standardize result/error envelopes or MCP exception conventions;
- distinguish zero results from execution failure;
- stop converting provider/operator exceptions into ordinary empty results unless the result explicitly carries failure state;
- provide structured recovery hints only where they express capability/resource facts, not hard-coded user-level reasoning policy.

### Exit criteria

- the harness can programmatically distinguish build/retry/fallback/reformulate/stop cases;
- canonical contract tests cover error classes;
- empty/incomplete KG results do not automatically become negative factual conclusions;
- no failure path masquerades as a successful answer payload.

## Stage 8 — Architectural reliability and CI

**Goal:** prove the contracts remain stable as implementation changes.

### Work

Create a deterministic test matrix for:

- operator descriptor↔implementation parity;
- slot and field compatibility;
- strict/best-effort validation semantics;
- custom composition execution;
- MCP discovery/execution parity;
- prompt-policy parity where duplicate prompt surfaces remain;
- resource registration/prerequisite/invalidation;
- provenance propagation;
- graph-build→retrieve→evidence E2E flow;
- cross-modal conversion contracts;
- standardized failures/recovery;
- intentionally retained legacy-entry-point compatibility.

Tighten CI so canonical contract tests are blocking. Keep live-provider/expensive tests separately classified.

### Exit criteria

- architecture contract regressions fail CI;
- MyPy/integration permissiveness is either tightened for supported paths or clearly scoped as non-blocking optional coverage;
- provider/LLM tests are separated from deterministic core tests;
- supported Python/package/build paths are documented and exercised;
- status docs can reference test categories rather than anecdotal past runs.

## Stage 9 — Incremental and temporal/conflict semantics

**Goal:** add sophisticated lifecycle/evidence behavior after the resource/provenance foundation exists.

### Work

- define supported incremental corpus/graph/index updates;
- specify entity identity behavior across updates;
- propagate invalidation to communities/matrices/VDBs/conversions;
- add assertion validity/time metadata where required;
- represent source disagreement explicitly;
- define query-time filtering/selection as capabilities, not hidden global policy.

### Exit criteria

- new documents can be incorporated with documented consequences for derived artifacts;
- stale resources are detectable;
- conflicting/time-bounded claims can coexist and be surfaced to the harness.

## Stage 10 — Later evaluation and research validation

**Goal:** measure the stabilized architecture rather than allowing a benchmark to define it.

This stage is intentionally deferred until earlier architecture exit criteria are substantially met.

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

## What not to add before it solves a documented gap

Avoid expanding scope with:

- another general-purpose internal planner;
- another orchestrator variant;
- a mandatory Graph-of-Thought/AoT executor;
- a formal reasoning DAG without a concrete scheduling/resume/cache/provenance use case;
- another UI shell;
- a multi-agent coordination framework without a concrete capability need;
- benchmark-specific special cases in the core resource model;
- self-reported confidence as a substitute for evidence/provenance.

## Immediate implementation focus

If code work begins directly from this roadmap, the next sequence is:

1. **capability/descriptor/MCP inventory and parity audit**;
2. **validation policy cleanup** — strict vs explicit best-effort;
3. **prompt-source ownership/parity cleanup**;
4. **resource descriptor/catalog**;
5. **prerequisite/lifecycle semantics**;
6. **evidence/provenance contract**;
7. **harness-first entry-point cleanup**;
8. **legacy consolidation**;
9. **cross-modal/error normalization**;
10. **blocking contract tests/CI hardening**.

That sequence closes concrete implementation gaps without programming the harness's intelligence for it.