# DIGIMON Gap Analysis

**Updated:** 2026-09-16  
**Purpose:** identify the concrete distance between the code described in [CURRENT_STATE.md](CURRENT_STATE.md) and the target in [ARCHITECTURE.md](ARCHITECTURE.md).

The current priority is **architecture completion**, not benchmark optimization or novelty claims.

For exact module-level implementation details, see [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md).

## Priority model

- **A — foundational:** closes ambiguity in the capability/resource/evidence boundary. Later work depends on it.
- **B — consolidation:** reduces duplicate/legacy architecture and makes the harness surface dependable.
- **C — hardening:** improves reliability, test coverage and maintainability after the architecture is coherent.
- **Later:** valuable validation/research work intentionally deferred.

## Gap matrix

| Priority | Area | Target | Current state | Gap | Next architectural action |
|---|---|---|---|---|---|
| **A** | Canonical capability surface | One discoverable typed capability model used consistently by external harnesses | 26-operator registry/composition core is strong; MCP also exposes build/config/analysis/cross-modal tools outside that registry | The harness sees more capabilities than the operator registry describes, and not all tools share one descriptor/result model | Define the canonical capability descriptor and a mapping/adaptor path for non-operator tools; make MCP discovery derive from or verify against it |
| **A** | Composition safety semantics | Validation outcome has a clear execution contract | `ChainValidator` performs static checks; `PipelineExecutor` performs stricter dispatch checks; `OperatorComposer.execute()` logs invalid plans and proceeds best-effort | “Validated composition” can mean different things depending on layer; invalid plans may begin execution | Decide fail-closed vs explicit best-effort modes, make the choice caller-visible, and add tests for invalid wiring/prerequisites |
| **A** | Resource model | Uniform identities/state for corpus, graph, VDB, communities, sparse matrices and converted artifacts | `GraphRAGContext` directly tracks graphs/VDBs; MCP/server logic discovers/manages additional artifacts | Resource identity, dependency, reuse, invalidation and lifecycle are distributed | Introduce a typed `ResourceDescriptor`/catalog abstraction and adapters for existing artifacts |
| **A** | Prerequisite semantics | Every capability declares requirements and the harness can inspect/build/fallback explicitly | Operator descriptors have boolean prerequisite flags; `auto_build` helpers exist | Requirements do not identify concrete compatible resources/builders; build behavior is partly hidden in server/composer helpers | Replace/augment booleans with explicit resource requirements and producer-capability links |
| **A** | Provenance/evidence | Every material output can retain traceability to source evidence | `source_id`/`chunk_id`, producer metadata, graph→chunk paths and evidence-aware synthesis exist | No universal evidence object or enforced lineage propagation across all operators/conversions | Define evidence/provenance schema and propagation rules; attach lineage to typed outputs rather than reconstructing at synthesis time |
| **A** | Harness-first boundary | External harness is default adaptive planner; DIGIMON supplies capabilities and bounded model-assisted operators | MCP supports this well, but CLI/older modules still center `PlanningAgent`/orchestrators | Architectural ownership remains inconsistent across entry points | Make MCP/capability layer canonical; treat internal planning as compatibility/reference behavior and stop expanding it |
| **A** | Stable error semantics | Harness can distinguish missing resource, missing prerequisite, empty result, incompatible input, provider failure and internal failure | Pipeline layer raises explicit errors; some operators return empty/failure slots; build/MCP tools use status objects or exceptions | Recovery policy still requires tool-specific knowledge and empty retrieval can resemble failure | Define a small typed error taxonomy and standard result/error envelope or MCP convention |
| **B** | MCP ↔ operator parity | Discovery, schemas and execution semantics stay synchronized | MCP manually wraps many tools while operator registry describes the 26 retrieval/meta operators | Manual wrappers/descriptions can drift from registry metadata and return schemas | Centralize/generate descriptions where practical; add parity tests for IDs, schemas, prerequisites and outputs |
| **B** | Prompt source of truth | One clear prompt/template authority per execution path, or enforced semantic parity | YAML decomposition/synthesis prompts and operator-local prompt text are now aligned but duplicated | The two surfaces can drift again; the typed meta operators do not automatically load the YAML files | Centralize prompt loading or add explicit mapping/parity tests and document ownership |
| **B** | Sub-question representation | Advisory decomposition has semantically appropriate typed output without forcing a reasoning runtime | `meta.decompose_question` currently stores sub-question strings in `EntityRecord.entity_name` inside `ENTITY_SET` | Type reuse is expedient but semantically misleading and limits clean dependency metadata | Consider a generic text/task-list slot or record; only add a formal dependency DAG if concrete runtime capabilities require it |
| **B** | Registry/descriptor semantic precision | Descriptors exactly match operator behavior and supported payloads | Some metadata is broader/narrower than declared slot types; e.g. reranking comments mention chunks while descriptor uses `ENTITY_SET` | Machine-readable discovery can overpromise or mischaracterize valid payloads | Audit all 26 descriptors against implementations and reference plans; make mismatches blocking contract tests |
| **B** | Internal agent/orchestrator cleanup | No competing cognitive architecture defines core behavior | `Core/AgentBrain`, multiple `Core/AgentOrchestrator` variants and planner utilities remain | New contributors can mistake legacy/transitional orchestration for the target | Classify modules, identify live callers, deprecate unused variants, isolate compatibility entry points |
| **B** | Legacy AoT cleanup | AoT/GoT is an optional heuristic prompt unless a concrete runtime feature needs structure | `Core/AOT` implements atomic states, heuristic extraction and transition probabilities | Old implementation conflicts conceptually with current policy | Keep clearly legacy; remove from active paths if unused; retain history in Git |
| **B** | Cross-modal normalization | Conversions participate in typed capabilities/resources/provenance | Substantive graph/table/vector conversion code and MCP tools exist | Uses DataFrame/ndarray/dictionaries and separate conventions; lineage/lossiness not unified | Define conversion descriptors/output resources and provenance/lossiness metadata; integrate with discovery |
| **B** | Resource invalidation/update semantics | Derived resources know when source changes make them stale | Build/reuse behavior is mostly artifact/path/context based | No canonical dependency graph or invalidation policy | Add build fingerprints/dependency links; define rebuild vs reuse behavior |
| **B** | MCP session/runtime state | Resource/session ownership is explicit and safe for supported deployment model | Stdio server keeps process-level `_state` and one `GraphRAGContext` | Multi-session/client semantics are implicit; process CWD/state are architectural assumptions | Document supported single-session model now; design explicit session/resource scope only if deployment needs it |
| **B** | Conflict and temporal evidence | Incompatible claims and validity windows survive retrieval/synthesis | Metadata may exist in graph data, but no canonical assertion/time semantics | Synthesis can expose conflicts only if upstream representation preserves them | Extend evidence model with assertion/source/time/status fields before sophisticated conflict reasoning |
| **B** | Incremental updates | New documents update identities, relations and derived resources predictably | No canonical incremental update contract | Rebuild semantics dominate; downstream consequences underspecified | First define resource dependency/invalidation; then specify supported incremental operations by artifact type |
| **B** | API/CLI surface alignment | Secondary entry points reflect the same capability architecture | CLI uses internal planner; API/UI surfaces come from different generations | User-facing behavior and docs can drift | Adapt entry points to canonical capabilities where practical or label compatibility/experimental explicitly |
| **C** | Test taxonomy | Unit/contract/integration/E2E tests correspond to architectural boundaries | Many tests exist, including operator and HotpotQA paths | Coverage/support status is hard to infer from file presence | Create a test matrix keyed to capability contracts, resource lifecycle, MCP parity, provenance and supported entry points |
| **C** | CI trustworthiness | Required checks correspond to supported paths and fail on architecture regressions | Black/Flake8 + unit tests block; MyPy and integration tests currently tolerate failure | Green CI does not imply all important integration/type contracts passed | Make canonical deterministic contract tests blocking; separate optional/live-provider suites |
| **C** | Dependency/package story | One documented installation model matches supported entry points | Multiple requirements/environment files and broad optional dependencies remain | Setup expectations are difficult to infer | Separate core/MCP, research and UI dependencies; verify package/build metadata |
| **C** | Repository hygiene | Source tree clearly distinguishes source, generated artifacts, experiments and history | Research workspace contains outputs/logs/cache/node_modules/results and old plan/test surfaces | Reviewers/contributors face noise and accidental authority from old files | Remove/ignore generated artifacts where safe; continue converting high-authority stale docs to historical stubs/archive conventions |
| **Later** | Benchmark/ablation quality | Demonstrate when graph/composition helps and at what cost | Evaluation infrastructure exists | Not the current architectural bottleneck | Use `FUTURE_EVALUATION_QUESTIONS.md` after architecture exit criteria are met |
| **Later** | Router calibration/novelty | Measure method selection and research contribution | Auto-selection prompt exists | Premature while contracts/resources/provenance are evolving | Defer |

## The architecture gaps that matter most

### 1. Capability metadata is not yet the whole harness surface

The operator registry is a strong machine-readable core, but graph builders, corpus preparation, VDB builders, analysis, resource/config tools and cross-modal transformations also appear through MCP with separate contracts.

**Why it matters:** a capable harness should not need multiple discovery systems or implementation-specific knowledge.

### 2. Resource lifecycle is not first-class

The operator layer is more mature than the resource layer. Operators can say they require an entity VDB or communities, but the system does not yet have one uniform typed description of every resource, how it was built, what it depends on, whether it is stale, and how the harness can obtain it.

**Why it matters:** reliable tool selection depends on inspectable resource state.

### 3. Provenance exists as identifiers, not yet as an invariant

`source_id`, `chunk_id`, producer metadata and retrieval paths are useful foundations. The gap is propagation: every transformation should preserve lineage or explicitly state what was combined/lost.

**Why it matters:** graph reasoning is most useful when a final claim can be traced through structure back to source material.

### 4. Composition validation has multiple levels of strictness

Static validation is permissive and `OperatorComposer` currently proceeds after validation errors, while `PipelineExecutor` applies stricter checks at dispatch.

**Why it matters:** the harness needs to know whether a plan was rejected, accepted with warnings, or intentionally executed in best-effort mode.

### 5. The code still tells two orchestration stories

The modern story is typed capabilities + MCP + capable harness. The older story is internal `PlanningAgent` + orchestrators + programmed cognitive/AoT layers. Both still exist in code and some entry points.

**Why it matters:** continuing both as equal architectures duplicates policy and creates unclear ownership.

### 6. Prompt policy is duplicated

The YAML and typed meta-operator prompts are aligned today, but they are separate text sources.

**Why it matters:** documentation can become false again if only one execution path is updated.

### 7. Failure/recovery semantics need standardization

“No result,” “missing prerequisite,” “KG omitted the relation,” “model/provider failed,” and “plan wiring is invalid” require different recovery actions but are not represented uniformly.

**Why it matters:** a harness can recover intelligently only from machine-readable failure facts.

## Architectural decisions already made

The following should **not** be reopened accidentally while closing these gaps:

- do not build a mandatory programmed agent brain;
- do not make legacy AoT state/transition code the reasoning runtime;
- do not force every query into a graph pipeline;
- do not make `auto_compose` the only orchestration path;
- do not formalize a reasoning DAG merely because the decomposition heuristic can express dependencies;
- do not optimize architecture around HotpotQA or another benchmark yet;
- do not add more front ends as a substitute for stabilizing contracts/resources/evidence;
- do keep reference methods because they are useful shortcuts, compatibility paths and future baselines.

## What “architecture complete enough to evaluate” means

Before benchmarking becomes a primary activity, the project should be able to answer these implementation questions cleanly:

1. What capabilities are available, and what are their typed inputs/outputs?
2. Which capability descriptions are authoritative at the MCP boundary?
3. What resources exist for this dataset/session?
4. What prerequisite is missing for this capability?
5. How can that prerequisite be built or avoided?
6. Did static plan validation pass, warn, or fail, and will execution proceed?
7. What source evidence supports this returned entity/relationship/chunk/path?
8. Did a conversion or aggregation lose information or lineage?
9. What failure class occurred, and what recovery choices are available?
10. Which modules are canonical versus compatibility/experimental/legacy?

The roadmap turns these questions into exit criteria.