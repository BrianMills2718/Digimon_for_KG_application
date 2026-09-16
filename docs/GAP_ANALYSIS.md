# DIGIMON Gap Analysis

**Updated:** 2026-09-16  
**Purpose:** identify the concrete distance between the code described in [CURRENT_STATE.md](CURRENT_STATE.md) and the target in [ARCHITECTURE.md](ARCHITECTURE.md).

The current priority is **architecture completion**, not benchmark optimization or novelty claims.

## Priority model

- **A — foundational:** closes ambiguity in the capability/resource/evidence boundary. Later work depends on it.
- **B — consolidation:** reduces duplicate/legacy architecture and makes the harness surface dependable.
- **C — hardening:** improves reliability, test coverage and maintainability after the architecture is coherent.
- **Later:** valuable validation/research work intentionally deferred.

## Gap matrix

| Priority | Area | Target | Current state | Gap | Next architectural action |
|---|---|---|---|---|---|
| **A** | Canonical capability surface | One discoverable typed capability model used consistently by external harnesses | 26-operator registry/composition core is strong; MCP also exposes build/config/analysis/cross-modal tools outside that registry | The harness sees more capabilities than the canonical operator registry describes, and not all tools share one descriptor/result model | Define the canonical capability descriptor and decide which non-operator tools belong in it; make MCP discovery derive from or map to that model |
| **A** | Resource model | Uniform identities/state for corpus, graph, VDB, communities, sparse matrices and converted artifacts | `GraphRAGContext` directly tracks graphs/VDBs; MCP/server logic discovers/manages additional artifacts | Resource identity, dependency, reuse, invalidation and lifecycle are distributed | Introduce a typed `ResourceDescriptor`/catalog abstraction and adapters for existing artifacts |
| **A** | Prerequisite semantics | Every capability declares requirements and the harness can inspect/build/fallback explicitly | Operator descriptors have prerequisite flags; `auto_build` helpers exist | Requirements are not one extensible contract; build behavior is partly hidden in server/composer helpers | Replace booleans/policy-specific checks over time with explicit prerequisite/resource requirements and build-capability links |
| **A** | Provenance/evidence | Every material output can retain traceability to source evidence | `source_id` on entity/relationship records, `chunk_id` on chunks, relationship→chunk retrieval, synthesis prompt preservation | No universal evidence object or enforced lineage propagation across all operators/conversions | Define evidence/provenance schema and propagation rules; attach lineage to typed outputs rather than reconstructing at synthesis time |
| **A** | Harness-first boundary | External harness is default adaptive planner; DIGIMON supplies capabilities and bounded model-assisted operators | MCP supports this well, but CLI and older modules still center `PlanningAgent`/orchestrators; older ADR/docs describe a dual-brain system | Architectural ownership is inconsistent across entry points and docs | Make MCP/capability layer canonical; treat internal planning as compatibility/reference behavior and stop expanding it |
| **A** | Stable error semantics | Harness can distinguish missing resource, missing prerequisite, empty result, incompatible input, provider failure and internal failure | Error handling exists locally but result/error shapes vary across tool families | Recovery policy requires tool-specific knowledge | Define a small typed error taxonomy and standard result envelope or MCP error conventions |
| **B** | MCP ↔ operator parity | Discovery, schemas and execution semantics stay synchronized | MCP server manually wraps many tools while operator registry describes the 26 retrieval/meta operators | Manual wrappers can drift from registry metadata and return schemas | Generate/centralize descriptions where practical; add parity tests for operator IDs, schemas, prerequisites and outputs |
| **B** | Internal agent/orchestrator cleanup | No competing cognitive architecture defines core behavior | `Core/AgentBrain`, multiple `Core/AgentOrchestrator` variants and planner utilities remain | New contributors can mistake legacy/transitional orchestration for the target | Classify modules, identify actual callers, deprecate unused variants, isolate compatibility entry points |
| **B** | Legacy AoT cleanup | AoT/GoT is an optional heuristic prompt unless a concrete runtime feature needs structure | `Core/AOT` implements atomic states, heuristic extraction and transition probabilities in code | Old implementation conflicts conceptually with current policy | Mark legacy in code/docs; remove from active paths if unused; retain history in git rather than current architecture |
| **B** | Cross-modal normalization | Conversions participate in typed capabilities/resources/provenance | Substantive graph/table/vector conversion code and MCP tools exist | Uses DataFrame/ndarray/intermediate dictionaries and separate conversion conventions; lineage/lossiness not unified | Define conversion descriptors/output resources and provenance/lossiness metadata; then integrate with capability discovery |
| **B** | Resource invalidation/update semantics | Derived resources know when source corpus/graph changes make them stale | Build/reuse behavior is mostly artifact/path/context based | No canonical dependency graph or invalidation policy | Add build fingerprints/dependency links to resource descriptors; define rebuild vs reuse behavior |
| **B** | Conflict and temporal evidence | System can preserve incompatible claims and validity windows instead of flattening them | Metadata may exist in graph data, but no canonical evidence semantics | Synthesis can be instructed to surface conflicts only if upstream representation preserves them | Extend evidence model with assertion/source/time/status fields before adding sophisticated conflict reasoning |
| **B** | Incremental updates | New documents can update identities, relations and derived resources predictably | No canonical incremental update contract in current architecture docs | Rebuild semantics dominate; identity/index/community consequences are underspecified | First define resource dependency/invalidation; then specify supported incremental operations by artifact type |
| **B** | API/CLI surface alignment | Secondary entry points reflect the same capability architecture | CLI uses internal planner; API/UI surfaces come from different generations | User-facing behavior and docs can drift | Either adapt entry points to call the canonical capability/MCP layer or explicitly label them compatibility/experimental |
| **C** | Test taxonomy | Unit/contract/integration/E2E tests correspond to architectural boundaries | Many tests exist, including HotpotQA and operator tests; CI has unit/integration jobs | Coverage/status is difficult to read; some CI checks are non-blocking; old checkpoint tests coexist with modern tests | Create a test matrix keyed to capability contracts, resource lifecycle, MCP parity and provenance |
| **C** | CI trustworthiness | Required checks correspond to supported paths and fail when architectural contracts regress | Workflow includes lint, tests, build and Docker; mypy/integration checks are partly permissive | A green workflow may not mean all important contracts passed | Decide supported CI matrix; make canonical contract tests blocking before tightening optional/provider-dependent paths |
| **C** | Dependency/package story | One documented installation model matches supported entry points | Multiple requirements/environment files and broad optional dependencies remain | Setup expectations are difficult to infer | Separate core/MCP, optional research and UI dependencies; verify package/build metadata |
| **C** | Repository hygiene | Source tree clearly distinguishes source, generated artifacts, experiments and history | Research workspace contains outputs/logs/cache/node_modules/results and multiple old plan/test surfaces | Reviewers/contributors face noise and accidental authority from old files | Move or ignore generated artifacts where safe; consolidate historical docs under a clearly marked archive over time |
| **Later** | Benchmark/ablation quality | Demonstrate when graph/composition helps and at what cost | Evaluation infrastructure exists | Not the current architectural bottleneck | Use `FUTURE_EVALUATION_QUESTIONS.md` after architecture exit criteria are met |
| **Later** | Router calibration/novelty | Measure method selection and research contribution | Auto-selection prompt exists | Premature while contracts/resources/provenance are evolving | Defer |

## The five architecture gaps that matter most

### 1. Resource lifecycle is not yet first-class

The operator layer is more mature than the resource layer. Operators can say they require an entity VDB or communities, but the system does not yet have one uniform typed description of every resource, how it was built, what it depends on, whether it is stale, and how the harness should obtain it.

**Why it matters:** a harness can only reason reliably about tools when prerequisites and artifacts are inspectable.

### 2. Provenance exists as identifiers, not yet as an end-to-end invariant

`source_id` and `chunk_id` are valuable foundations. The gap is propagation: every transformation should either preserve lineage or explicitly say that lineage was lost/aggregated.

**Why it matters:** graph reasoning is most useful when the final claim can be traced back through the graph to source text.

### 3. The code has two architectural stories

The modern story is typed operators + MCP + capable harness. The older story is internal `PlanningAgent` + orchestrators + programmed cognitive/AoT layers. Both still exist in code and several entry points/docs.

**Why it matters:** continuing both as equal architectures would duplicate reasoning policy and create unclear ownership.

### 4. Capability discovery is fragmented

The registry is excellent for the 26 operator core, while graph builders, VDB builders, resource/config tools and cross-modal tools are exposed separately through MCP.

**Why it matters:** the harness should not need to learn multiple metadata systems to discover what DIGIMON can do.

### 5. Failure/recovery semantics need standardization

The target harness is intelligent enough to recover, but it needs machine-readable reasons to do so.

**Why it matters:** “no results,” “resource missing,” “KG extraction omitted the relation,” and “provider failed” require very different next actions.

## Architectural decisions already made

The following should **not** be reopened accidentally while closing these gaps:

- do not build a mandatory programmed agent brain;
- do not make legacy AoT state/transition code the reasoning runtime;
- do not force every query into a graph pipeline;
- do not make `auto_compose` the only orchestration path;
- do not optimize architecture around HotpotQA or another benchmark yet;
- do not add more front ends as a substitute for stabilizing contracts/resources/evidence;
- do keep reference methods because they are useful shortcuts, compatibility paths and future baselines.

## What “architecture complete enough to evaluate” means

Before benchmarking becomes a primary activity, the project should be able to answer these implementation questions cleanly:

1. What capabilities are available, and what are their typed inputs/outputs?
2. What resources exist for this dataset/session?
3. What prerequisite is missing for this capability?
4. How can that prerequisite be built or avoided?
5. What source evidence supports this returned entity/relationship/chunk/path?
6. Did a conversion or aggregation lose information?
7. What failure class occurred, and what recovery choices are available?
8. Which modules are canonical versus compatibility/legacy?

The roadmap turns those questions into exit criteria.