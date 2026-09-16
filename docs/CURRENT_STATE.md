# DIGIMON Current State

**Snapshot:** 2026-09-16  
**Repository:** `BrianMills2718/Digimon_for_KG_application`  
**Purpose:** describe what is materially present in the public codebase now, without treating historical plans as current implementation.

## Executive summary

DIGIMON is currently a **hybrid/transitional codebase** whose strongest architectural center is the typed operator/composition layer exposed through the stdio MCP server.

The repository already contains substantial working structure: typed operator slots, a machine-readable registry of 26 operators, plan validation/execution, 10 reference method profiles, multiple graph builders, entity/relationship/chunk/subgraph/community operations, resource inspection, prerequisite-building helpers, cross-modal conversion code, CLI/API/MCP surfaces, evaluation infrastructure, and end-to-end tests.

At the same time, the repository still contains earlier internal planning/orchestration systems, programmed AoT code, multiple orchestrators, old MCP/multi-agent experiments, and documentation from earlier architectural directions. The current target is **harness-first**: DIGIMON should make retrieval/analysis capabilities safe, discoverable, composable, resource-aware, and evidence-grounded while a capable external harness owns adaptive reasoning.

This reconciliation is a **source/documentation inspection**, not a fresh full runtime certification. “Implemented” below means substantive code is present and wired into a current surface; it does not claim every live LLM/provider path was rerun on 2026-09-16.

For module-by-module classification and implementation caveats, see [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md).

## Status vocabulary

- **Implemented** — meaningful code exists and is wired into a current execution surface.
- **Partial** — meaningful code exists, but integration, consistency, lifecycle, contracts, or reliability are incomplete.
- **Legacy** — retained code/documentation that is not the preferred target architecture.
- **Planned** — target behavior is not materially complete in this snapshot.

## Current capability map

| Area | Status | What exists now | Main evidence |
|---|---|---|---|
| Typed operator dataflow | **Implemented** | Seven slot kinds plus typed entity, relationship, chunk, subgraph, community and score records; `SlotValue` records producer and metadata. | `Core/Schema/SlotTypes.py` |
| Operator registry | **Implemented** | 26 operators across entity, relationship, chunk, subgraph, community and meta categories, with typed I/O, cost and prerequisite metadata. | `Core/Operators/registry.py` |
| Composition engine | **Implemented / Partial hardening** | Static chain validation, pre-dispatch slot checks, pipeline execution, loops/conditionals, reference-method profiling and execution. | `Core/Composition/` |
| Reference retrieval methods | **Implemented** | Ten named method plans: `basic_local`, `basic_global`, `lightrag`, `fastgraphrag`, `hipporag`, `tog`, `gr`, `dalk`, `kgp`, `med`. | `Core/Methods/`, `OperatorComposer.py` |
| Corpus preparation | **Implemented** | MCP-facing preparation for `.txt`, `.md`, `.json`, `.jsonl`, `.csv`, and `.pdf`; structured-field parsers exist. | `digimon_mcp_stdio_server.py`, `Core/AgentTools/corpus_*` |
| Graph construction | **Implemented** | ER, RK, tree, balanced-tree and passage graph build tools are wired into MCP; corpus can be auto-prepared from an input directory. | `digimon_mcp_stdio_server.py`, `Core/AgentTools/graph_construction_tools.py` |
| Entity/vector retrieval | **Implemented** | Entity VDB search, one-hop expansion, PPR, linking, TF-IDF and model-assisted entity paths exist in current operator/tool layers. | `Core/Operators/entity/`, MCP server |
| Relationship retrieval | **Implemented** | One-hop, VDB, score aggregation and model-assisted relationship operators; MCP also exposes relationship-resource helpers. | `Core/Operators/relationship/`, MCP server |
| Chunk/source retrieval | **Implemented** | Relationship→chunk, entity-occurrence and score→chunk operations plus direct chunk/source lookup surfaces. | `Core/Operators/chunk/`, MCP server |
| Subgraph/path retrieval | **Implemented** | K-hop paths/neighborhoods, Steiner-tree extraction and model-assisted path filtering. | `Core/Operators/subgraph/`, MCP server |
| Community operations | **Implemented / Partial lifecycle** | Community operators and community-building/access tools exist; usability depends on derived artifacts. | `Core/Operators/community/`, MCP server |
| Resource discovery | **Partial** | MCP can inspect graphs, VDBs and other derived artifacts; `GraphRAGContext` itself models graphs/VDBs plus providers/config. | `digimon_mcp_stdio_server.py`, `Core/AgentSchema/context.py` |
| Prerequisite handling | **Partial** | Descriptor flags and `auto_build` helpers cover several prerequisites. | descriptors, MCP server, composition code |
| MCP harness surface | **Implemented in code** | `FastMCP` stdio server exposes individual capabilities, reference methods, auto-selection, config/resource tools and cross-modal operations. | `digimon_mcp_stdio_server.py` |
| CLI | **Implemented / Transitional** | CLI still instantiates `PlanningAgent`/`AgentOrchestrator`, including experimental ReAct behavior. | `digimon_cli.py` |
| HTTP/API and UI surfaces | **Partial / Secondary** | API, dashboard, Streamlit and React-era surfaces remain, but are not the canonical orchestration boundary. | `api.py`, UI files |
| Cross-modal graph/table/vector conversion | **Implemented / Experimental integration** | Conversion code uses NetworkX/pandas/NumPy plus embedding adapters and validation helpers. | `Core/AgentTools/cross_modal_tools.py` |
| Dependency-aware decomposition heuristic | **Implemented / Transitional representation** | YAML and typed meta-operator prompts now use advisory dependency-aware AoT/GoT guidance; sub-questions are still carried as `EntityRecord` values in `ENTITY_SET`. | `prompts/decompose_question.yaml`, `Core/Operators/meta/decompose_question.py` |
| Evidence-aware synthesis heuristic | **Implemented / Partial provenance inputs** | YAML and typed meta-operator prompts now preserve available evidence markers, surface conflicts/unresolved dependencies, and avoid unsupported bridges. | `prompts/synthesize_answers.yaml`, `Core/Operators/meta/synthesize_answers.py` |
| Legacy programmed AoT runtime | **Legacy** | `Core/AOT` encodes atomic states, heuristic extraction and transition probabilities directly in code. | `Core/AOT/` |
| Internal agent brain / multiple orchestrators | **Legacy / Transitional** | Substantial internal planner/orchestrator code remains and is used by some entry points. | `Core/AgentBrain/`, `Core/AgentOrchestrator/` |
| Provenance/evidence representation | **Partial** | Entity/relationship records carry `source_id`; chunks carry `chunk_id`; `SlotValue` can carry producer/metadata; retrieval can return source text. | `Core/Schema/SlotTypes.py`, chunk operators |
| Error/recovery semantics | **Partial** | Pipeline dispatch has explicit errors, but individual operators/build tools can also return empty/failure-valued results or status objects. | `PipelineExecutor.py`, operator/tool implementations |
| Conflict/temporal evidence semantics | **Planned / Uneven** | No canonical cross-system assertion/conflict/valid-time evidence model. | architecture gap |
| Incremental resource updates | **Planned / Not canonical** | No uniform update/invalidation semantics across graphs, VDBs, communities and conversions. | architecture gap |
| Evaluation framework | **Implemented / Deferred priority** | Benchmark runner records EM, token F1/precision/recall, latency, LLM calls and token usage; benchmark/E2E tests exist. | `eval/benchmark.py`, tests |
| Automated CI | **Partial** | Black/Flake8 and unit tests are blocking; MyPy and integration tests are currently permissive/non-blocking. | `.github/workflows/ci.yml` |

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

This layer describes reusable capabilities without requiring DIGIMON to own the user-level reasoning policy.

### 2. External harness access through MCP

The stdio MCP server exposes three useful levels:

1. **individual capabilities/operators** — conceptual default for a capable harness;
2. **reference methods** — known operator compositions;
3. **auto selection** — optional prompt/model selection of a reference method.

Modes 2 and 3 are useful conveniences and future baselines. Mode 1 defines the target architectural boundary.

### 3. Multiple retrieval structures

The codebase contains ER/RK graphs, hierarchical trees, passage graphs, vector indexes, graph traversal, communities, sparse operations and graph/table/vector conversions. DIGIMON is therefore already broader than a single fixed GraphRAG pipeline.

## Important implementation caveats

### Composition is typed, but not yet a closed safety contract

The current composition code is substantive, but several details matter:

- registry successor/chain helpers reason mainly over `SlotKind`; they do not prove resource prerequisites, field requirements, semantic applicability or cost constraints;
- `ChainValidator` may accept an unwired input when another prior output of the same kind exists, emitting a warning;
- `OperatorComposer.execute()` currently logs static validation failures and proceeds best-effort;
- `PipelineExecutor` then performs stricter input-name/type checks before dispatch and defaults to fail-fast operator execution;
- loop accumulation currently wraps carried outputs as `ENTITY_SET`, which is not a fully general typed-loop model;
- some descriptors represent semantically broader inputs than their declared slot type, such as the current rerank descriptor.

The composition layer should therefore be described as **implemented with contract-hardening gaps**, not as a fully solved planner/type system.

### Prompt policy is aligned, but prompt ownership is duplicated

The decomposition and synthesis YAML prompts now match the operator-local prompts conceptually. However, the typed meta operators do not automatically load those YAML files. Equivalent instructions therefore live in more than one place and can drift again.

This is a documentation/architecture gap to solve through centralized prompt loading or explicit parity tests—not by building a new reasoning executor.

### Sub-questions use a generic carrier type

`meta.decompose_question` currently stores suggested sub-question text in `EntityRecord.entity_name` values under an `ENTITY_SET` slot. This is compatible with the current seven-slot system but semantically awkward.

A more general text/task-list slot may eventually be useful. A formal dependency DAG should still only be introduced when a concrete runtime function—scheduling, resumability, caching, provenance, auditing—requires it.

### Resource state is split across abstractions

`GraphRAGContext` directly stores graphs and VDBs. Other resources are discovered or built through additional MCP/server logic and filesystem conventions. There is no one typed resource catalog with stable IDs, build fingerprints, dependencies, staleness and invalidation semantics.

### MCP state is process-level

The stdio server lazily initializes process-level `_state`, stores providers/context there, and changes the process working directory to the project root. That is adequate for the present tool-server model but is not yet a general multi-session resource architecture.

### Evidence exists, but not as an invariant

Current records preserve useful IDs and the synthesis operator can pass available chunk/source markers to the model. The missing piece is enforced lineage propagation through every aggregation, path/community operation and cross-modal transformation.

### Errors are not uniform

`PipelineExecutor` can raise actionable `PipelineExecutionError`, while some operators catch failures and return empty/failure-valued slots and MCP/build tools may return structured status objects or raise exceptions. Empty retrieval, missing prerequisite and actual execution failure therefore remain too easy to conflate.

## Historical material that is not current truth

Several older implementations/documents describe priorities that no longer match the target architecture, including:

- internal general-purpose planner/orchestrator expansion;
- mandatory programmed AoT/Markov preprocessing;
- old WebSocket MCP checkpoint sequences;
- older UKRF/multi-agent roadmaps;
- historical tool registries/counts and performance targets.

`docs/CHECKPOINT_PROGRESS.md` and the root MCP planning files are explicitly marked historical. Git history preserves the original details.

## Current development priority

The active order is:

1. stabilize the canonical capability contract and MCP parity;
2. unify resource identities/lifecycle/prerequisites;
3. make provenance/evidence an end-to-end contract;
4. make the harness-first boundary operationally clean;
5. consolidate legacy planners/orchestrators/AoT code;
6. normalize cross-modal capabilities;
7. standardize machine-actionable errors/recovery;
8. harden architectural contract tests and CI;
9. add incremental/temporal/conflict semantics after the foundations exist;
10. perform benchmarking/research validation later.

## Verification boundary

This pass inspected repository structure and key implementation files. It did **not** rerun every graph builder, provider-dependent operator, MCP tool, UI/API path, benchmark, Docker build or CI job.

Runtime claims in historical documentation should not be promoted into current guarantees unless a current test or fresh execution supports them.

Continue with:

- [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md) — exact module classification and implementation caveats;
- [ARCHITECTURE.md](ARCHITECTURE.md) — target design;
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — current → target gaps;
- [ROADMAP.md](ROADMAP.md) — ordered closure plan.