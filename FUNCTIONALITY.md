# DIGIMON Functionality

**Snapshot:** 2026-09-16  
**Purpose:** concise capability inventory. For architectural status and gaps, see `docs/CURRENT_STATE.md`, `docs/IMPLEMENTATION_MAP.md`, and `docs/GAP_ANALYSIS.md`.

## Status vocabulary

- **Implemented** — substantive code exists and is wired into a current surface.
- **Partial** — substantive code exists but lifecycle/integration/contracts/reliability are incomplete.
- **Legacy** — present for compatibility/history, not the target architecture.
- **Planned** — target behavior is not materially complete.

This page describes code presence and wiring; it is not a fresh certification of every provider-dependent runtime path.

## One-liner

DIGIMON exposes composable document, graph, vector, text, community and structured-analysis capabilities to an intelligent harness. The harness decides how to reason; DIGIMON provides retrieval/analysis machinery and should preserve the evidence needed to justify answers.

## Implemented capability core

### Typed operator system — **Implemented**

The canonical retrieval core contains **26 registered operators** with typed slot I/O, cost tiers, prerequisite flags and compatibility metadata.

Categories:

- entity — 7;
- relationship — 4;
- chunk — 3;
- subgraph — 3;
- community — 2;
- meta — 7.

Key files:

- `Core/Schema/SlotTypes.py`
- `Core/Schema/OperatorDescriptor.py`
- `Core/Operators/registry.py`
- `Core/Composition/`

### Composition/validation — **Implemented / Partial hardening**

DIGIMON has real typed plan validation and execution, including:

- slot-kind metadata;
- static chain validation;
- pre-dispatch slot-name/type checking;
- fail-fast operator execution by default;
- loops and conditional branches;
- reference-plan profiling/execution;
- compatibility and chain-discovery helpers.

Important limits:

- slot-kind compatibility does not prove that resource prerequisites are available;
- static validation is permissive in places;
- `OperatorComposer` currently has a best-effort path after validation failure;
- some descriptor/slot semantics still need implementation-level parity auditing;
- loop accumulation and certain generic meta operations are not a fully general typed workflow model.

These are explicit architecture-hardening tasks in `docs/ROADMAP.md`.

### Reference retrieval methods — **Implemented**

Ten named methods are represented as operator plans and profiled/executed through `OperatorComposer`. They are convenience/reference compositions rather than the core abstraction:

- `basic_local`
- `basic_global`
- `lightrag`
- `fastgraphrag`
- `hipporag`
- `tog`
- `gr`
- `dalk`
- `kgp`
- `med`

### Corpus preparation — **Implemented**

`corpus_prepare` supports document directories containing:

- `.txt`
- `.md`
- `.json`
- `.jsonl`
- `.csv`
- `.pdf`

Structured parsers include field-detection logic for text/title-like columns/fields.

### Graph construction — **Implemented**

Current MCP graph-build surfaces:

- `graph_build_er`
- `graph_build_rk`
- `graph_build_tree`
- `graph_build_tree_balanced`
- `graph_build_passage`

Graph-build calls can accept an `input_directory`; the MCP server can prepare a missing corpus before building.

### Entity retrieval — **Implemented**

The current operator/tool layers include:

- entity VDB build/search;
- one-hop expansion;
- Personalized PageRank;
- entity linking;
- TF-IDF ranking;
- model-assisted entity scoring/extraction paths.

### Relationship retrieval — **Implemented**

The current operator/tool layers include:

- one-hop relationship retrieval;
- relationship VDB build/search;
- score aggregation;
- model-assisted relation selection.

### Chunk/source retrieval — **Implemented**

Capabilities include:

- chunks from relationships;
- entity-occurrence chunk retrieval;
- score-to-chunk aggregation;
- direct source-text/chunk lookup tools on the MCP surface.

### Subgraph/path retrieval — **Implemented**

Capabilities include:

- K-hop path/neighborhood extraction;
- Steiner-tree extraction;
- model-assisted path relevance filtering.

### Community operations — **Implemented / Partial lifecycle**

Community retrieval/build/access capabilities exist. Their usability depends on community artifacts being available or built as prerequisites.

### Graph analysis/visualization — **Implemented**

The MCP/tool surface includes graph structural analysis and graph export/visualization-oriented capabilities.

### Resource inspection — **Partial**

The current MCP server can inspect graphs/VDBs and other derived resources.

The gap is architectural rather than absence of functionality: resource identity, dependencies, fingerprints, staleness, invalidation and lifecycle are not yet represented by one uniform typed resource catalog.

`GraphRAGContext` directly models graphs and VDBs, while other artifact knowledge is spread across server/filesystem logic.

### Prerequisite auto-build — **Partial**

Reference-method execution can build several missing prerequisites automatically where supported.

The current behavior is useful, but prerequisite semantics are distributed across descriptor booleans, server helpers and artifact conventions rather than one canonical resource requirement contract.

## Harness-facing execution

### Stdio MCP server — **Implemented in code / preferred external surface**

`digimon_mcp_stdio_server.py` uses `FastMCP` and exposes three levels:

1. **individual capability/operator calls** — preferred conceptual mode for a capable harness;
2. **reference method execution** — run a named composition;
3. **auto selection** — optional prompt/model chooses a reference method.

It also exposes corpus/graph construction, resource/config inspection, analysis and cross-modal tools.

The server currently maintains process-level lazy `_state` and one `GraphRAGContext`; broader session/resource isolation is not yet a canonical architecture feature.

### CLI — **Implemented / Transitional**

`digimon_cli.py` is a working project entry point, but it still instantiates the older internal `PlanningAgent` and `AgentOrchestrator`, including experimental ReAct mode.

It should be treated as a compatibility/transitional surface rather than the target definition of orchestration.

### API/UI surfaces — **Partial / Secondary**

HTTP API, dashboard, Streamlit and React-era surfaces remain in the repository. They are not the architectural center of the current reconciliation.

## Cross-modal analysis — **Implemented / Experimental integration**

`Core/AgentTools/cross_modal_tools.py` contains graph/table/vector conversions, embedding-provider adapters and conversion validation/selection support.

The code uses NetworkX, pandas, NumPy and embedding adapters (including a deterministic hash provider for testing). The gap is normalization: DataFrame/array/dictionary payloads and provenance/lossiness semantics are not yet fully represented through the same typed capability/resource model as the 26-operator core.

## Reasoning heuristics

### Dependency-aware decomposition — **Implemented / Transitional typing**

Both the YAML prompt and typed meta operator now use advisory dependency-aware guidance such as:

```text
q1: identify an intermediate entity
q2: retrieve facts about <q1.entity>
q3: resolve the answer against retrieved source evidence
```

The harness may ignore, merge, reorder, branch, parallelize or revise this structure. It is not a mandatory executor.

`meta.decompose_question` currently stores sub-question text inside `EntityRecord.entity_name` values in an `ENTITY_SET` slot. That is a compatibility shortcut, not an ideal long-term semantic type.

### Evidence-aware synthesis — **Implemented / Partial provenance inputs**

Both the YAML prompt and typed meta operator instruct synthesis to:

- preserve available source/chunk/provenance markers;
- distinguish evidence from unsupported inference;
- expose material conflicts/unresolved dependencies;
- avoid treating missing evidence as proof of falsity;
- avoid manufacturing confidence.

The operator now includes available chunk/source markers in model context, but upstream lineage is not yet universal.

### Prompt source of truth — **Partial**

The YAML templates and typed meta-operator prompt text are aligned in this snapshot but are separate sources. Centralized loading or parity tests are still needed to prevent drift.

### Legacy programmed AoT — **Legacy**

`Core/AOT/` contains an older programmed atomic-state/transition approach. It is historical/experimental code, not the target reasoning architecture.

## Evidence/provenance — **Partial**

Current typed records already carry useful identifiers:

- entities: `source_id`;
- relationships: `source_id`;
- chunks: `chunk_id`;
- slot values: producer + metadata.

Retrieval can move from graph evidence back to chunks/source text, and synthesis can preserve supplied provenance.

What is **not yet complete** is a universal end-to-end evidence contract that guarantees lineage propagation across every operator, aggregation, subgraph/community operation and cross-modal conversion.

## Error/recovery semantics — **Partial**

Current layers use different conventions:

- `PipelineExecutor` raises explicit pipeline errors for several invalid-input/upstream-failure cases;
- some individual operators catch exceptions and return empty/failure-valued slots;
- MCP/build tools may raise exceptions or return structured status objects.

The architecture target is to make empty evidence, missing prerequisites, invalid plans, provider failures, extraction incompleteness and internal failures machine-distinguishable.

## Evaluation — **Implemented infrastructure / Deferred priority**

The repository includes an evaluation runner capable of tracking exact match, token-level F1/precision/recall, latency, LLM calls and token usage, plus benchmark/test datasets including HotpotQA material.

Evaluation is not the current architecture priority. Deferred questions are documented in `docs/FUTURE_EVALUATION_QUESTIONS.md`.

## Current architecture priorities

The active implementation sequence is documented in `docs/ROADMAP.md`. In short:

1. capability/descriptor/MCP inventory and parity;
2. explicit strict-vs-best-effort validation semantics;
3. prompt ownership/parity;
4. resource catalog/lifecycle/prerequisites;
5. end-to-end evidence/provenance;
6. clean harness-first boundary;
7. legacy orchestration/AoT/MCP consolidation;
8. cross-modal normalization;
9. error/recovery standardization;
10. architectural contract tests/CI hardening.

For the full code-vs-goal reconciliation, use:

- `docs/CURRENT_STATE.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/ARCHITECTURE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`