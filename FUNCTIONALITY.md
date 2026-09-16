# DIGIMON Functionality

**Snapshot:** 2026-09-16  
**Purpose:** concise capability inventory. For architectural status and gaps, see `docs/CURRENT_STATE.md` and `docs/GAP_ANALYSIS.md`.

## Status vocabulary

- **Implemented** — substantive code exists and is wired into a current surface.
- **Partial** — substantive code exists but lifecycle/integration/contracts/reliability are incomplete.
- **Legacy** — present for compatibility/history, not the target architecture.
- **Planned** — target behavior is not materially complete.

This page describes code presence and wiring; it is not a fresh certification of every provider-dependent runtime path.

## One-liner

DIGIMON exposes composable document, graph, vector, text, community and structured-analysis capabilities to an intelligent harness. The harness decides how to reason; DIGIMON provides the retrieval/analysis machinery and should preserve the evidence needed to justify answers.

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

### Reference retrieval methods — **Implemented**

Ten named methods are represented as operator plans and profiled/executed through `OperatorComposer`. They are convenience/reference compositions rather than the core abstraction.

Current method profiles include:

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

`list_available_resources` exposes graphs, VDBs, communities, sparse matrices and datasets through the current MCP server.

The gap is architectural rather than absence of functionality: resource identity, dependencies, fingerprints, invalidation and lifecycle are not yet represented by one uniform typed resource catalog.

### Prerequisite auto-build — **Partial**

Reference-method execution can build several missing prerequisites automatically, including VDB/community/sparse resources where supported.

The current behavior is useful, but prerequisite semantics are distributed across descriptors/server helpers rather than one canonical resource contract.

## Harness-facing execution

### Stdio MCP server — **Implemented in code / preferred external surface**

`digimon_mcp_stdio_server.py` uses `FastMCP` and exposes three levels of execution:

1. **individual capability/operator calls** — preferred conceptual mode for a capable harness;
2. **reference method execution** — run one named composition;
3. **auto selection** — optional LLM/prompt chooses a reference method.

It also exposes resource/config inspection, graph construction, analysis and cross-modal tools.

### CLI — **Implemented / transitional**

`digimon_cli.py` is a working project entry point, but it still instantiates the older internal `PlanningAgent` and `AgentOrchestrator`, including an experimental ReAct mode.

It should be treated as a compatibility/transitional surface rather than the target definition of orchestration.

### API/UI surfaces — **Partial / secondary**

HTTP API, dashboard, Streamlit and React-era surfaces remain in the repository. They are not the architectural center of the current reconciliation.

## Cross-modal analysis — **Implemented / experimental integration**

`Core/AgentTools/cross_modal_tools.py` contains graph/table/vector conversions, embedding-provider adapters and conversion validation/selection support.

The current gap is normalization: these conversions are not yet fully represented through the same typed operator/resource/provenance model as the 26-operator retrieval core.

## Reasoning heuristics

### Dependency-aware decomposition — **Implemented as heuristic**

`prompts/decompose_question.yaml` can suggest a dependency-aware structure such as:

```text
q1: identify an intermediate entity
q2: retrieve facts about <q1.entity>
q3: resolve the answer against retrieved source evidence
```

The harness may ignore, merge, reorder, branch or revise this structure. It is not a mandatory executor.

### Evidence-aware synthesis — **Implemented as heuristic**

`prompts/synthesize_answers.yaml` instructs synthesis to preserve supplied source/provenance markers, distinguish evidence from inference, expose material conflicts/unresolved dependencies, and avoid treating missing evidence as proof of falsity.

### Legacy programmed AoT — **Legacy**

`Core/AOT/` contains an older programmed atomic-state/transition approach. It is historical/experimental code, not the target reasoning architecture.

## Evidence/provenance — **Partial**

Current typed records already carry useful identifiers:

- entities: `source_id`;
- relationships: `source_id`;
- chunks: `chunk_id`.

Retrieval can move from graph evidence back to chunks/source text, and synthesis can preserve supplied provenance.

What is **not yet complete** is a universal end-to-end evidence contract that guarantees lineage propagation across every operator, aggregation, subgraph/community operation and cross-modal conversion.

## Evaluation — **Implemented infrastructure / deferred priority**

The repository includes an evaluation runner capable of tracking exact match, token-level F1/precision/recall, latency, LLM calls and token usage, plus benchmark/test datasets including HotpotQA material.

Evaluation is not the current architecture priority. Deferred questions are documented in `docs/FUTURE_EVALUATION_QUESTIONS.md`.

## Current architecture priorities

The active implementation sequence is documented in `docs/ROADMAP.md`. In short:

1. canonical capability contract;
2. resource catalog/lifecycle/prerequisites;
3. end-to-end evidence/provenance;
4. clean harness-first boundary;
5. legacy orchestration/AoT consolidation;
6. cross-modal normalization;
7. error/recovery semantics;
8. contract tests/CI hardening.

For the full code-vs-goal reconciliation, use:

- `docs/CURRENT_STATE.md`
- `docs/ARCHITECTURE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`
