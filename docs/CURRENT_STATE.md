# DIGIMON Current State

**Snapshot:** 2026-09-16  
**Repository:** `BrianMills2718/Digimon_for_KG_application`  
**Purpose:** describe what is materially present in the public codebase now, without treating historical plans as current implementation.

## Executive summary

DIGIMON is currently a **hybrid/transitional codebase** whose strongest architectural center is the typed operator/composition layer exposed through the stdio MCP server.

The repository already contains substantial working structure: typed operator slots, a machine-readable registry of 26 operators, plan validation/execution, 10 reference method profiles, multiple graph builders, entity/relationship/chunk/subgraph/community operations, resource inspection, prerequisite-building helpers, cross-modal conversion code, CLI/API/MCP surfaces, evaluation infrastructure, and end-to-end tests.

At the same time, the repository still contains earlier internal planning/orchestration systems, legacy AoT code, multiple orchestrators, old WebSocket-MCP plans, and documentation from earlier architectural directions. The current target is **harness-first**: DIGIMON should make retrieval/analysis capabilities safe and composable while a capable external harness owns adaptive reasoning.

This review is a **code/documentation inspection**, not a fresh full runtime certification. “Implemented” below means that substantive code is present and wired into a current surface; it does not claim every live LLM/provider path was re-executed on 2026-09-16.

## Status vocabulary

- **Implemented** — meaningful code exists and is wired into a current execution surface.
- **Partial** — meaningful code exists, but integration, consistency, lifecycle, contracts, or reliability are incomplete.
- **Legacy** — retained code or documentation that is not the preferred target architecture.
- **Planned** — target behavior is not materially complete in this snapshot.

## Current capability map

| Area | Status | What exists now | Main evidence |
|---|---|---|---|
| Typed operator dataflow | **Implemented** | Seven slot kinds plus typed entity, relationship, chunk, subgraph, community and score records; `SlotValue` records producer and metadata. | `Core/Schema/SlotTypes.py` |
| Operator registry | **Implemented** | Machine-readable registry for 26 operators across entity, relationship, chunk, subgraph, community and meta categories, with cost/prerequisite metadata and compatibility helpers. | `Core/Operators/registry.py` |
| Composition engine | **Implemented** | Plan validation, pipeline execution, reference-method profiling and execution. `OperatorComposer` intentionally contains no LLM method-selection policy. | `Core/Composition/`, especially `OperatorComposer.py` |
| Reference retrieval methods | **Implemented** | Ten named method plans are profiled/composed as convenience pipelines rather than the core abstraction. | `Core/Methods/`, `OperatorComposer.py` |
| Corpus preparation | **Implemented** | MCP-facing preparation for `.txt`, `.md`, `.json`, `.jsonl`, `.csv`, and `.pdf`; structured-field parsing helpers exist. | `digimon_mcp_stdio_server.py`, `Core/AgentTools/corpus_*` |
| Graph construction | **Implemented** | ER, RK, tree, balanced-tree and passage graph build tools are wired into MCP; auto corpus preparation is supported when an input directory is supplied. | `digimon_mcp_stdio_server.py`, `Core/AgentTools/graph_construction_tools.py` |
| Entity/vector retrieval | **Implemented** | Entity VDB build/search, one-hop expansion, PPR, linking, TF-IDF and LLM-assisted entity operators are represented in the current operator/tool layers. | `Core/Operators/entity/`, MCP server |
| Relationship retrieval | **Implemented** | One-hop, VDB, score aggregation and LLM-assisted relationship operators; MCP wrappers include VDB build/search surfaces. | `Core/Operators/relationship/`, MCP server |
| Chunk/source retrieval | **Implemented** | Chunk retrieval from relationships/entity occurrence and score aggregation are in the typed operator system; source/chunk lookup tools are also exposed by MCP. | `Core/Operators/chunk/`, MCP server |
| Subgraph/path retrieval | **Implemented** | K-hop paths, Steiner-tree extraction and LLM-assisted path filtering exist as operators/tools. | `Core/Operators/subgraph/`, MCP server |
| Community operations | **Implemented / Partial lifecycle** | Community operators and community-building/access tools exist; availability depends on built community artifacts. | `Core/Operators/community/`, MCP server |
| Resource discovery | **Partial** | MCP can list available graphs, VDBs, communities, sparse matrices and datasets; `GraphRAGContext` itself primarily models graphs/VDBs plus providers/config. Resource lifecycle is not yet one uniform typed catalog. | `digimon_mcp_stdio_server.py`, `Core/AgentSchema/context.py` |
| Prerequisite handling | **Partial** | `auto_build` and helper logic can build several missing prerequisites for reference methods, but prerequisite semantics are distributed between operator descriptors, MCP helpers and context/artifact conventions. | operator descriptors, MCP server, composition code |
| MCP harness surface | **Implemented in code** | `FastMCP` stdio server exposes individual operators, reference methods, auto composition, resource/config inspection and cross-modal tools. It is the clearest current external-harness surface. | `digimon_mcp_stdio_server.py` |
| CLI | **Implemented, transitional** | CLI still instantiates `PlanningAgent`/`AgentOrchestrator`, with normal and experimental ReAct processing. This is usable project lineage but not the preferred conceptual center. | `digimon_cli.py` |
| HTTP/API and UI surfaces | **Partial / secondary** | API, dashboard, Streamlit and React-era surfaces remain in the repository, but this documentation pass does not treat them as the canonical orchestration interface. | `api.py`, dashboard/UI files |
| Cross-modal graph/table/vector conversion | **Implemented, experimental** | Conversion functions exist for graph↔table/vector paths with validation/selection MCP tools. Some paths use heuristics/fallback embeddings and do not yet share the same typed slot model as the 26 retrieval operators. | `Core/AgentTools/cross_modal_tools.py`, MCP server |
| AoT/GoT heuristic prompts | **Implemented as heuristic** | Question decomposition and answer synthesis prompts are dependency/evidence aware and intentionally non-binding. | `prompts/decompose_question.yaml`, `prompts/synthesize_answers.yaml` |
| Legacy programmed AoT runtime | **Legacy** | `Core/AOT` encodes atomic states, heuristic extraction and transition probabilities directly in code. It is not the target reasoning architecture. | `Core/AOT/` |
| Internal agent brain / multiple orchestrators | **Legacy / transitional** | Substantial `Core/AgentBrain` and `Core/AgentOrchestrator` code remains and is used by some entry points. It should not define the future capability boundary. | `Core/AgentBrain/`, `Core/AgentOrchestrator/`, CLI |
| Provenance/evidence representation | **Partial** | Typed entity/relationship records carry `source_id`; chunks have `chunk_id`; retrieval operators can map relationships/entities to source chunks; synthesis prompts preserve supplied markers. A universal claim→evidence→source contract is not yet enforced across every path. | `Core/Schema/SlotTypes.py`, chunk operators, synthesis prompt |
| Conflict/temporal evidence semantics | **Planned / uneven** | Individual graph properties may carry metadata, but there is no canonical cross-system conflict/valid-time evidence model documented or enforced as an invariant. | gap identified in architecture review |
| Incremental resource updates | **Planned / not canonical** | The canonical docs do not currently define stable incremental-update semantics for graph identity, indexes, communities and derived artifacts. | gap identified in architecture review |
| Evaluation framework | **Implemented, deferred priority** | Benchmark runner records EM, token F1/precision/recall, latency, LLM calls and tokens; HotpotQA and other tests exist. Architecture completion is the present priority rather than benchmark optimization. | `eval/benchmark.py`, `test_hotpotqa.py` |
| Automated CI | **Partial** | Workflow includes formatting/lint, unit tests, integration tests, build and Docker jobs, but some checks are explicitly non-blocking and this pass did not verify the latest workflow run. | `.github/workflows/ci.yml` |

## What is architecturally strongest today

### 1. Typed composable operators

The clearest modern core is:

```text
Slot types / records
      ↓
Operator descriptors + registry
      ↓
Chain validation / execution
      ↓
Reference plans or harness-built compositions
```

This layer makes capabilities inspectable without deciding the user's reasoning strategy for them.

### 2. External harness access through MCP

The stdio MCP server already describes three useful levels of abstraction:

1. **individual operators** — the client/harness composes;
2. **reference methods** — known operator chains execute as shortcuts;
3. **auto composition** — an internal heuristic/model can pick a reference method when desired.

For the target architecture, mode 1 is the conceptual center. Modes 2 and 3 remain useful conveniences, fallbacks and testable reference behavior.

### 3. Multiple retrieval structures

DIGIMON is not limited to one graph representation. The code contains ER/RK graph builders, hierarchical trees, passage graphs, vector indexes, graph traversal, communities and cross-modal conversion code. The target architecture should preserve that plurality while making the lifecycle and evidence contracts more uniform.

## Where the codebase is transitional

### Internal orchestration remains embedded in some entry points

The CLI and older code still instantiate `PlanningAgent` and `AgentOrchestrator`. Multiple orchestrator variants and planner modules remain. These are real parts of the repository, but the target is not to keep expanding a bespoke internal cognitive architecture.

### Resource state is split across abstractions

`GraphRAGContext` directly tracks graphs and VDBs, while other artifacts/resources are discovered or managed through additional MCP/server logic and filesystem conventions. This works, but the target should make resources, capabilities, prerequisites and lifecycle states easier for a harness to inspect uniformly.

### Evidence is present but not yet a universal contract

The typed records already preserve important identifiers (`source_id`, `chunk_id`) and retrieval paths can return text evidence. The remaining architectural work is to make provenance a first-class, uniform contract rather than something each operator/tool can represent differently.

### Cross-modal code is broader than the typed operator core

Graph/table/vector conversion is substantive code, but it currently sits mainly under agent tools/MCP rather than being fully normalized into the same slot/descriptor/resource model as the 26 registered retrieval operators.

## Historical material that is not current truth

Several older documents describe priorities that no longer match the code or target architecture. Examples include:

- the June 2025 WebSocket MCP checkpoint plan;
- older multi-agent/UKRF roadmaps;
- mandatory programmed AoT/Markov preprocessing;
- documents that describe 18-tool or pre-operator registries as current;
- old action items claiming MCP/config/operator composition has not yet been built.

Git history preserves those decisions. The canonical status is this document plus `ARCHITECTURE.md`, `GAP_ANALYSIS.md`, and `ROADMAP.md`.

## Verification boundary

This documentation reconciliation inspected repository structure and relevant implementation files. It did **not** rerun every provider-dependent graph build, all LLM-backed operators, every MCP tool, the full CI matrix, or every UI/API path. Runtime claims in older documentation should therefore not be promoted into current guarantees unless tests or fresh executions support them.

The next architecture work should close the gaps documented in [GAP_ANALYSIS.md](GAP_ANALYSIS.md), not add more orchestration surfaces.