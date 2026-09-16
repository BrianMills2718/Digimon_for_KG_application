# DIGIMON Implementation Map

**Reconciled:** 2026-09-16  
**Purpose:** map the current repository to the canonical harness-first architecture so a contributor can tell what to extend, what to treat as transitional, and what not to mistake for the target design.

This document complements, rather than replaces:

- `CURRENT_STATE.md` — status of the system now;
- `ARCHITECTURE.md` — target architecture;
- `GAP_ANALYSIS.md` — current → target gaps;
- `ROADMAP.md` — ordered closure plan.

It is based on source inspection. It is **not** a claim that every provider-dependent path was rerun during this reconciliation.

## Architectural center of gravity

The current preferred execution model is:

```text
external intelligent harness
        ↓
digimon_mcp_stdio_server.py
        ↓
capability/build/analysis surface
        ↓
Core/Operators + Core/Composition + Core/Methods
        ↓
GraphRAGContext / graph / index / chunk / provider implementations
        ↓
derived resources + original source material
```

The external harness owns adaptive orchestration. DIGIMON owns capabilities, typed contracts, resource/prerequisite facts, bounded model-assisted operations, and evidence boundaries.

## Module classification

| Path / surface | Classification | Current role | Guidance |
|---|---|---|---|
| `Core/Schema/SlotTypes.py` | **Canonical / Implemented** | Seven typed operator dataflow kinds and record classes | Extend carefully; provenance/resource work will likely add structure around these records |
| `Core/Schema/OperatorDescriptor.py` | **Canonical / Implemented, limited** | Machine-readable operator metadata, cost and prerequisite flags | Preferred metadata foundation; current boolean prerequisites are too narrow for the target resource model |
| `Core/Operators/registry.py` | **Canonical / Implemented** | Registers the current 26 retrieval/meta operators | Keep synchronized with implementations and MCP exposure; descriptor drift is a Stage-1 concern |
| `Core/Operators/` | **Canonical / Implemented** | Entity, relationship, chunk, subgraph, community and meta operations | Preferred home for reusable typed retrieval operations |
| `Core/Composition/ChainValidator.py` | **Canonical / Partial hardening** | Static slot-compatibility checks for plans | Useful but currently permissive; not a complete prerequisite/resource validator |
| `Core/Composition/PipelineExecutor.py` | **Canonical / Implemented, Partial semantics** | Resolves typed inputs and executes plans, loops and conditionals | Has stricter pre-dispatch slot checks; error/result semantics still need normalization |
| `Core/Composition/OperatorComposer.py` | **Canonical / Implemented, Partial validation policy** | Profiles/builds/executes ten reference methods | Does not own global LLM routing; currently logs validation failures and may execute best-effort |
| `Core/Methods/` | **Canonical reference layer / Implemented** | Ten named reference operator plans | Keep as shortcuts, compatibility paths and future baselines; do not make them the system identity |
| `Core/AgentSchema/context.py` | **Canonical foundation / Partial resource model** | Runtime context for providers plus graph/VDB instances | Useful foundation, but only graphs/VDBs are first-class resource collections today |
| `Core/AgentTools/graph_construction_tools.py` and corpus/build tools | **Canonical capability implementations / Implemented** | Corpus and graph construction used by MCP | Map into the eventual common capability/resource descriptor model |
| `Core/AgentTools/cross_modal_tools.py` | **Implemented / Experimental integration** | Graph↔table↔vector transformations and embedding adapters | Substantive code, but outside the seven-slot operator/resource model and provenance contract |
| `Core/AgentTools/*planner*` and older planner utilities | **Legacy / Experimental** | Earlier internal planning strategies | Do not expand as the primary reasoning architecture |
| `digimon_mcp_stdio_server.py` | **Canonical external facade / Implemented in code** | FastMCP stdio surface for build, operators, methods, resources, analysis and cross-modal tools | Preferred harness boundary; future work should reduce manual metadata/resource drift |
| `Core/MCP/` | **Mixed legacy/experimental MCP lineage** | Older MCP clients/servers, coordination and integration experiments | Do not assume these modules define the current MCP architecture; inspect callers before modifying |
| `Core/AgentBrain/` | **Legacy / Transitional** | Broad internal planning and synthesis logic used by some older entry points | Maintain only for compatibility while live callers remain; do not deepen as default architecture |
| `Core/AgentOrchestrator/` | **Legacy / Transitional** | Multiple generations of internal orchestration | Identify callers and consolidate over time |
| `Core/AOT/` | **Legacy** | Programmed atomic states, dependency sets and transition probabilities | Not the target AoT/GoT architecture; current policy is advisory prompting |
| `Core/Memory/` | **Experimental / Legacy lineage** | Earlier memory/strategy-learning architecture | Not a current architecture priority unless a concrete use case reintroduces it |
| `Core/Graph/`, `Core/Index/`, `Core/Chunk/`, `Core/Community/`, `Core/Provider/` | **Foundational implementation** | Underlying graph/index/chunk/community/provider machinery | Keep behind stable capability/resource contracts where possible |
| `eval/` | **Implemented infrastructure / Deferred priority** | Benchmark execution and quality/cost measurements | Preserve; architecture work currently takes precedence over benchmark optimization |
| `tests/`, `testing/`, root `test_*.py` | **Mixed active/experimental test estate** | Unit, integration, E2E and historical test scripts | Build a canonical test taxonomy before treating file presence as support status |
| `digimon_cli.py` | **Implemented / Transitional entry point** | Uses internal `PlanningAgent`/`AgentOrchestrator`, optional ReAct mode | Compatibility surface, not preferred orchestration boundary |
| `api.py`, dashboards, Streamlit, React UI | **Secondary / Mixed-generation surfaces** | Alternate user/application interfaces | Do not add new architectural policy here; eventually adapt to canonical capabilities or label clearly |

## Canonical typed operator core

### Slot kinds

`Core/Schema/SlotTypes.py` currently defines:

1. `QUERY_TEXT`
2. `ENTITY_SET`
3. `RELATIONSHIP_SET`
4. `CHUNK_SET`
5. `SUBGRAPH`
6. `COMMUNITY_SET`
7. `SCORE_VECTOR`

Current records include useful provenance foundations:

- `EntityRecord.source_id`;
- `RelationshipRecord.source_id`;
- `ChunkRecord.chunk_id`;
- `SlotValue.producer` and free-form `metadata`.

These are useful primitives, but they are not yet a universal evidence contract.

### Registered operators

The registry currently contains 26 operators:

| Category | Count | Operators |
|---|---:|---|
| entity | 7 | `entity.vdb`, `entity.ppr`, `entity.onehop`, `entity.link`, `entity.tfidf`, `entity.agent`, `entity.rel_node` |
| relationship | 4 | `relationship.onehop`, `relationship.vdb`, `relationship.score_agg`, `relationship.agent` |
| chunk | 3 | `chunk.from_relation`, `chunk.occurrence`, `chunk.aggregator` |
| subgraph | 3 | `subgraph.khop_paths`, `subgraph.steiner_tree`, `subgraph.agent_path` |
| community | 2 | `community.from_entity`, `community.from_level` |
| meta | 7 | `meta.extract_entities`, `meta.reason_step`, `meta.rerank`, `meta.generate_answer`, `meta.pcst_optimize`, `meta.decompose_question`, `meta.synthesize_answers` |

### Reference methods

`Core/Methods/__init__.py` currently exposes ten plans:

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

`OperatorComposer` intentionally profiles/builds/executes these plans without making the method-selection decision itself.

## Composition: what is implemented and what is not

The composition layer is real, but it should not be described as a fully closed type/resource system yet.

### Implemented

- typed slot descriptors;
- plan wiring checks;
- named-output tracking;
- pre-dispatch slot-name/type validation in `PipelineExecutor`;
- fail-fast operator execution by default;
- loops and conditional branches;
- method plan profiling/execution;
- basic compatibility/chain-discovery helpers.

### Important limitations

1. **Registry compatibility helpers are slot-kind heuristics.** `get_compatible_successors()` looks for overlapping kinds, not proof that every required input/prerequisite is satisfied.
2. **Chain discovery is not resource-aware.** `find_chains_to_goal()` reasons about available slot kinds, not VDB/community/matrix/resource availability, field requirements, cost or semantic applicability.
3. **Static validation is permissive.** `ChainValidator` can treat any prior output of the same kind as satisfying an unwired required input and emits a warning rather than requiring explicit wiring.
4. **Composer validation is currently fail-open.** `OperatorComposer.execute()` logs validation errors and proceeds best-effort; `PipelineExecutor` may then reject bad inputs at dispatch time.
5. **Loop accumulation is not fully typed.** Accumulated loop outputs are currently wrapped as `ENTITY_SET`, regardless of the conceptual output being accumulated.
6. **Some descriptors are semantically broader than their slot type.** For example, `meta.rerank` documents that it can rerank chunks while its descriptor is expressed as `ENTITY_SET`.

These are architecture-hardening gaps, not reasons to discard the composition layer.

## Prompt and reasoning surfaces

### Current policy

AoT/GoT/ReAct are optional heuristics. They may suggest a reasoning shape; the harness remains free to choose a different path.

### Current implementations

- `prompts/decompose_question.yaml` — dependency-aware advisory decomposition;
- `prompts/synthesize_answers.yaml` — evidence-aware synthesis guidance;
- `Core/Operators/meta/decompose_question.py` — typed operator-local advisory decomposition prompt;
- `Core/Operators/meta/synthesize_answers.py` — typed operator-local evidence-aware synthesis prompt.

As of this reconciliation, the operator-local prompts have been aligned with the same policy as the YAML prompts.

### Remaining prompt gap

There are still **multiple prompt sources of truth**. The typed meta operators do not automatically load the YAML templates; equivalent instructions exist in two places. This can drift again.

A future cleanup should either:

- centralize prompt loading/templates, or
- explicitly define which prompt surface is authoritative for each execution path and test semantic parity.

### Transitional sub-question representation

`meta.decompose_question` currently returns suggested sub-questions in `EntityRecord.entity_name` inside an `ENTITY_SET` slot. This preserves compatibility with the existing seven-slot model but is semantically awkward.

Do not introduce a formal reasoning DAG solely to fix this. First decide whether a more general typed text/task-list slot is useful across capabilities. A dependency object is justified only if it serves scheduling, resumability, caching, provenance or another concrete system function.

## Resource/runtime state

### Current context

`GraphRAGContext` directly models:

- target dataset name;
- config;
- LLM provider;
- embedding provider;
- chunk factory/storage manager;
- graph instances;
- VDB instances;
- resolved configuration values.

It does **not** yet provide one common typed catalog for corpus artifacts, communities, sparse matrices, converted tables/vectors, build fingerprints, staleness or dependency edges.

### MCP state

`digimon_mcp_stdio_server.py` maintains process-level `_state` initialized lazily and stores configuration/providers/context there. It also changes the working directory to the project root during initialization.

This is adequate for the current single-process tool-server model, but session/multi-client isolation and explicit resource identity are not first-class architecture yet.

### Prerequisites

Operator descriptors currently use booleans such as:

- `requires_entity_vdb`;
- `requires_relationship_vdb`;
- `requires_community`;
- `requires_sparse_matrices`;
- `requires_llm`.

MCP helpers and reference-method execution add additional build/reuse behavior. The target is a resource requirement model that tells the harness *which compatible resource* is needed and *which capability can produce it*, rather than only a boolean flag.

## Evidence/provenance map

### Present now

- entity/relationship `source_id` fields;
- chunk IDs and chunk text;
- graph→chunk retrieval paths;
- `SlotValue.producer`/metadata;
- evidence-aware synthesis instructions;
- operator-local synthesis now includes available chunk/source markers in the prompt context.

### Still missing

- one universal evidence/assertion type;
- source-document identity/metadata contract across all loaders;
- enforced lineage propagation through score aggregation, subgraphs, communities and conversions;
- explicit distinction between retrieved assertion and derived inference at every layer;
- canonical conflict/time-validity representation;
- typed lossiness/provenance behavior for graph/table/vector conversion.

## Error semantics

Current error behavior varies by layer:

- `PipelineExecutor` raises `PipelineExecutionError` for several plan/dispatch failures and defaults to fail-fast execution;
- some individual operators catch exceptions and return an empty or failure-valued `SlotValue` with logs/metadata;
- MCP/build tools may raise runtime exceptions or return structured status objects;
- an empty retrieval and a failed retrieval are therefore not represented uniformly today.

The target error model should make these cases machine-distinguishable so a harness can choose build, retry, fallback, reformulation or stop behavior.

## Cross-modal implementation

`Core/AgentTools/cross_modal_tools.py` implements substantive graph/table/vector conversions using NetworkX, pandas and NumPy, with embedding provider adapters including a deterministic hash provider for testing.

It is real functionality, but its current payloads (`DataFrame`, `ndarray`, dictionaries) are not normalized into the seven-slot operator system or the target resource/evidence model. Treat it as **implemented functionality with partial architectural integration**.

## Testing and CI interpretation

The repository has meaningful tests and a CI workflow, but support claims should remain bounded:

- CI runs Black and Flake8 as blocking checks;
- MyPy is currently non-blocking (`|| true`);
- unit tests are blocking in the workflow;
- integration tests are currently non-blocking (`|| true`);
- build and Docker jobs depend on lint/test jobs;
- provider/LLM-dependent behavior is not equivalent to deterministic contract coverage.

The target is a clear test matrix where capability/slot/resource/provenance/MCP contract tests are blocking and live-provider tests are separately classified.

## Documentation classification

### Canonical current docs

- `README.md`
- `FUNCTIONALITY.md`
- `docs/README.md`
- `docs/CURRENT_STATE.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/ARCHITECTURE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`
- `docs/PLANNING_SUMMARY.md`
- `docs/AGENT_INTELLIGENCE_ENHANCEMENTS.md`
- `docs/FUTURE_EVALUATION_QUESTIONS.md`
- `docs/adr/002-harness-first-capability-architecture.md`
- `AGENTS.md`
- `CLAUDE.md`

### Historical or superseded examples

- `docs/CHECKPOINT_PROGRESS.md` — earlier internal-orchestrator/AoT checkpoint program;
- `docs/adr/001-agent-orchestration-architecture.md` — superseded orchestration decision;
- root WebSocket MCP plans/trackers — retained as historical stubs/provenance;
- older UKRF/multi-agent/cognitive-architecture reports and handoffs unless restated by canonical docs.

Historical material may still contain useful implementation ideas. It does not define current priority.

## Contributor decision rule

Before adding a new abstraction, ask:

1. Is this a reusable capability or resource fact that DIGIMON should own?
2. Can it fit the typed capability/resource/evidence model?
3. Is it actually orchestration policy that a capable harness can own instead?
4. Does an existing legacy planner/orchestrator already attempt the same thing?
5. Will the change preserve source evidence and produce machine-actionable failures?

Prefer improving the capability/resource/evidence plane over creating another internal reasoning layer.

## Where to work next

The implementation map points directly to the current roadmap order:

1. capability/descriptor/MCP inventory and parity;
2. explicit strict-vs-best-effort validation semantics;
3. prompt source-of-truth/parity;
4. typed resource catalog and prerequisite links;
5. provenance/evidence record and propagation;
6. clean harness-first entry points;
7. legacy planner/orchestrator/AoT/MCP consolidation;
8. cross-modal normalization;
9. standardized errors/recovery;
10. blocking architecture contract tests and CI;
11. incremental/temporal/conflict semantics later;
12. benchmarking/research validation after the architecture is coherent.
