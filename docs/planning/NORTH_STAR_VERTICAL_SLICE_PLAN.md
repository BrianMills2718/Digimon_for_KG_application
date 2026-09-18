# DIGIMON North-Star Vertical Slice — Batch and Converge

**Updated:** 2026-09-17  
**Revision:** 2 — replaces the calendar-driven execution sequence, not the vision  
**Planning path:** durable_solo; one writer, repository-local reversible implementation  
**Authority:** [VISION](../VISION.md), [ARCHITECTURE](../ARCHITECTURE.md), [ROADMAP](../ROADMAP.md), and the contributor's request to plan rapid implementation with approximately 1,000 authored lines/hour and trace-driven repair.  
**Consumer:** the contributor and the next execution session; this is the sole active execution plan for the north-star slice. [CURRENT_STATE](../CURRENT_STATE.md) retains system-wide status.  
**Method record:** [PlanningPathDecisionV1](supporting/north-star-speedrun-path.json). No work graph, claim, invented reviewer, or repeated approval loop is activated.

## Outcome And Boundaries

For a researcher/analyst working with governed text-derived knowledge, produce one inspectable project in which an external harness can navigate semantic content and representation schemas, retrieve across complementary structures using canonical IDs, analyze a bounded working set, and recover both source evidence and analytic derivation.

The full ambition remains **Represent → Retrieve → Analyze/Transform**, downstream of onto-canon6. This first integrated vertical is not a redefinition of DIGIMON as only graph retrieval, only provenance, or only a demo.

**First milestone:** one coherent prototype of the previously selected internal-product direction. Do not promote it to an internal product solely because tests pass.

### Canonical outcome probe

Use one real, authorized onto-canon6 Foundation export and its exact passage companion. The harness receives a question and a catalog entry point, not a hard-coded reasoning plan. It can:

1. discover an entity/topic and the available schemas in generated Markdown;
2. use its canonical ID for exact SQL lookup and at least one aggregation;
3. use the same identity for graph neighborhood retrieval;
4. perform an actual vector query against an index of this same snapshot;
5. run Leiden and a centrality calculation on an explicitly selected graph view/subgraph;
6. use the analytical result to select entities and retrieve exact supporting passages;
7. produce an inspectable finding with evidence, method, parameters, graph scope, limitations, and lineage.

A fixed script exercises these capabilities for regression; it does not establish autonomous harness composition. A real harness trace must additionally demonstrate a cross-representation move and an analytic-result-to-evidence move using actual available tools. The harness may choose a different order or omit unhelpful methods on other questions.

**Inspectable output:** a reproducibly generated directory containing SQLite, graph and vector artifacts, a Markdown catalog, analytical output, a finding, and machine-readable execution/lineage records. No new dashboard or wiki server is required.

**Negative case:** missing evidence, an unknown entity, or a stale/mismatched projection produces an explicit bounded result/failure, not fabricated evidence or a plausible-looking finding.

**Non-claims:** this vertical does not prove all reference methods, complete IR fidelity, general retrieval superiority, causal/social influence from centrality, production readiness, or every future representation family.

**Authority limits:** no deployment, upstream semantic-contract change, migration of canonical stores, or new provider spending authority is created here. DIGIMON is a public repository: private real exports and full source-bearing traces stay in authorized local storage. Check in synthetic/sanitized regression fixtures only when appropriate; label them accurately.

## Operating Model And Implementation Surface

The contributor's **approximately 1,000 authored code/test lines per active authoring hour** is an experimental throughput target, not a quota, observed rate, correctness claim, or delivery promise. Do not pad code to meet it. A smaller implementation that closes the same boundary is better.

Estimate **3,000–5,000 additional authored implementation/test lines** for the shortest integrated slice, including the bounded repair allowance below. This excludes existing code, generated wiki pages, exported data, lockfiles, formatting-only churn, and planning prose. Record additions/deletions and net growth separately; do not count repeated rewrites as delivered capability. Replace the estimate when actual failures change the surface.

The previous 4–7 week, 5-day, and 12–20 hour figures were unmeasured planning guesses, not runtime evidence. They no longer schedule this execution frontier. Measure generation, execution, diagnosis/repair, and tool/access overhead separately instead of repeatedly compressing guesses.

### Batch rule

```text
brief contract + expected result
→ coherent implementation batch, usually 500–1,500 authored lines including tests
→ compile/import + focused checks
→ rerun the growing end-to-end path
→ inspect first failing boundary and its smallest counterexample
→ repair the implicated cause(s)
→ rerun affected checks + cumulative regression
→ record exact result and commit a coherent change
```

Batch boundaries follow working behavior, not files or a mandatory line count. Do not commit one file at a time when a tested coherent commit is available. Do not create a new plan for each fix. When several failures share one expensive setup, collect/localize them in that run and repair the bounded cluster before repeating it.

**Unverified-work limit:** do not stack another dependent feature batch on an unexecuted batch. If execution is unavailable, keep the current batch explicitly unverified and solve the access/dependency blocker rather than building the remainder source-only. Independent work can advance only when it can itself execute against the stable interface.

**Progress signal:** inspectable capabilities through their intended consumer, not LOC, commit count, or log volume. Record source-present, focused-check-passed, integrated, and harness-observed separately. A test timeout or missing credential is not a pass.

## Architecture And Capability Invariants

- **onto-canon6 owns semantics:** consume its current supported export, preserve IDs/roles/literals/qualifiers/aliases/source scope, and do not re-extract or recanonicalize governed data.
- **Identity is scoped:** preserve canonical entity/assertion/passage IDs; bind projection artifacts to the exact input snapshot, passage companion, and namespace/registry context. Matching counts or matrix shapes are insufficient.
- **Representations are complementary:** SQLite, assertion graph, entity-network graph, vector index, Markdown navigation, and exact evidence serve different operations; none replaces canonical semantic authority.
- **N-ary semantics remain explicit:** no implicit clique expansion or direction inferred from dictionary order. Retain the assertion structure and declare what an entity-network projection omits.
- **Graph analytics have a defined domain:** record predicate/qualifier eligibility, polarity treatment, direction, weights, parallel-edge policy, self-loops, and whole-graph versus subgraph scope. Never call a graph-derived score original evidence or causal influence.
- **Use existing engines and consumers:** extend `Core/Projection`, maintained graph/VDB/context seams, and existing analytical implementations. A second private demo-only retrieval stack does not count as adoption.
- **The harness owns composition:** use native file navigation/search and SQL tooling where available. Add only missing specialized capability access, with a tiny callable/CLI bridge where needed; MCP is optional transport, not the product. Full CLI modernization and polished SDK design are later.
- **Wiki is content plus environment map:** show semantic organization, actual representations, schemas/ontology references, canonical join keys, supported operations, availability, and evidence. Observed predicate/type vocabulary is not a complete ontology unless the producer supplies one.
- **Derived state and evidence are distinct:** source text, retrieved working sets, computed analytics, and findings retain their own identities and origins.
- **Traceability begins with the first build:** lightweight execution/lineage records accompany each real artifact. This does not authorize building a telemetry platform or internal agent brain.
- **Rebuilds preserve usable prior outputs on failure:** write new derived outputs separately and publish only after validation; never destroy the previous project merely because regeneration was requested.

See [FOUNDATION_PROPERTY_GRAPH_DESIGN](FOUNDATION_PROPERTY_GRAPH_DESIGN.md) for the adopted two-graph intent. Its word “lossless” is a requirement to verify, not a completed round-trip guarantee.

## Current Baseline And Reuse

Inspected GitHub base: `d548f0c1f84e3450c252eedea8f3a02abf0f3520`.

| Existing seam | Disposition / intended consumer | Evidence boundary |
|---|---|---|
| `Core/Projection/FoundationIR.py`, `Identity.py` | Reuse and repair as input to every projection | Source exists; current repository tests have not been executed in this planning turn |
| `Core/Projection/Relational.py` | Reuse SQLite for exact queries, aggregation, and evidence joins | Source exists; no new database backend is required |
| `Core/Projection/PropertyGraph.py` | Reuse assertion graph and binary MultiGraph; explicit adapter into maintained entity-network retrieval | Pure projectors exist; runtime integration and fidelity remain unproved |
| `Core/AgentSchema/context.py`, `Core/Graph/`, `Core/Storage/`, `Core/Index/`, `Core/Operators/` | Extend actual graph/vector registration and consumer paths | Existing machinery, not a reason to assume every new graph shape is supported |
| `Core/Community/` and existing graph analytics | Reuse before wrapping or extending | Leiden/centrality must accept the selected working set rather than silently using the whole dataset |
| `tests/core/test_foundation_ir_contract.py`, `test_relational_projection_contract.py`, `test_foundation_property_graph_projection.py` | Execute as the first focused baseline | Test presence is not test success; add independent counterexamples rather than merely restating implementation |

New/not integrated: vectors from this IR, runtime graph adapter, generated catalog, bounded analytic surface, durable derivation records, and an observed harness journey.

The earlier “no runner” claim is not a timeless repository fact. This planning session verified Python exists locally but a direct GitHub Git request failed DNS; no repository test run followed. Recheck available execution routes at Batch 0. GitHub Actions, an authorized host checkout, or a scoped local checkout can each provide bounded evidence. Do not wait for CI if a suitable runner is already usable, and do not claim a partial/local run proves all environments.

## Milestone Horizon And Batch Contracts

The first executable frontier is **Batch 0**. Later acceptance contracts are specified now; file details remain reversible proposals. Ranges include code and focused tests and sum to **3,000–5,000 LOC**. The baseline repair allowance is not a cap on unknown defects.

| Batch | Planning state / prerequisite | Inspectable increment | Likely surfaces; acceptance signal | Additional LOC |
|---|---|---|---|---:|
| **0. Execute and repair existing projections** | Fully specifiable now; execution access not yet verified | Existing IR → SQLite/graphs → exact evidence can be run and inspected | Existing projection modules and their three test files; producer-compatible fixture, import/SQL/graph checks, first real counterexample | 300–600 |
| **1. One project, one runner, minimal trace + lineage** | Conditional on 0 | A saved/reopened project and growing demo initially exercising SQL → evidence; later stages explicitly absent | Proposed `Core/Projection/Project.py`, `Core/Projection/Execution.py`, `scripts/run_foundation_demo.py`; start/success/failure events, artifact hashes, negative case, replay | 400–700 |
| **2. Connect the graph to maintained retrieval** | Conditional on 1 | Canonical entity → real runtime neighborhood/subgraph → source evidence | Existing `PropertyGraph.py`, narrow adapter in `Core/Projection/`, actual context/storage/consumer seams; parallel assertions preserved, scope/loss documented | 500–800 |
| **3. Build and query real vectors** | Conditional on 1; independent of 2 after shared seam is stable | Entity/assertion/passage documents → existing embedding/VDB path → actual retrieval result with canonical IDs | Proposed `Core/Projection/Vector.py`, existing VDB/provider seams and focused tests; persisted reload, dimension/metric/model metadata, genuine query | 400–700 |
| **4. Generate progressive-disclosure catalog** | Conditional on representation metadata; live acceptance after 2 and 3 | Entry point → semantic page → real SQL/graph/vector addresses → source links | Proposed `Core/Projection/Catalog.py` and tests; schemas derived from real artifacts, stable page mappings, link checks, no fake availability | 500–800 |
| **5. Retrieve → analyze → retrieve evidence** | Conditional on 2 and shared execution record | Retrieved subgraph → Leiden + centrality → selected entity IDs → exact passages, plus a SQL aggregation | Existing analytic/Leiden machinery, narrow proposed `Core/Projection/Analytics.py` adapter and tests; finite scores, correct member domain, method/scope/seed, native-library or small-graph oracle | 500–800 |
| **6. Converge and observe the whole workflow** | Conditional on 2–5 | One inspectable finding and traceable artifact chain, executed by a real harness as well as the script | Extend the same runner; proposed `tests/e2e/test_foundation_vertical.py`; clean/reuse runs, source-change rejection, injected failure, intended-consumer evidence | 400–600 |

**Ordering flexibility:** graph and vector work do not require each other; catalog rendering can start from real partial metadata while marking absent resources. Analytics can follow the graph before catalog polish. There is one writer by default, not fictitious parallel agents. If real concurrent writers are introduced, freeze shared interfaces, assign non-overlapping write surfaces, and activate Company Planning coordination only then.

**No integration cliff:** every batch reruns the same growing journey. Batch 6 verifies and demonstrates integration already built; it is not the first time the pieces meet.

## Active Slice — Batch 0

**Visible result:** an executable baseline using the checked-in projection implementations, with a truthful pass/fail record and a minimal counterexample for any failure.

**First commands, from a real checkout:**

```bash
git rev-parse HEAD
git status --short
python --version
python -m compileall -q Core/Projection
python -m pytest \
  tests/core/test_foundation_ir_contract.py \
  tests/core/test_relational_projection_contract.py \
  tests/core/test_foundation_property_graph_projection.py -q
```

Use the repository's declared environment first. Install only missing relevant dependencies; do not spend the first batch installing unrelated optional UI/research stacks. Record the resolved interpreter/dependencies and checkout revision. A normal test invocation that fails during collection is evidence to fix, not permission to silently bypass the repository's test configuration.

### Checks that can invalidate the current implementation

- Compare the consumer with the actual onto-canon6 exporter and a producer-generated fixture. Preserve literal carriers and producer-defined metadata; do not invent SHA-256 semantics for an opaque producer field merely because the consumer currently expects them.
- Exercise duplicate/missing IDs, n-ary roles, qualifiers, aliases, Unicode, absent evidence, and mismatched source scope. Compare to expected input records independently of the projector's own success flags.
- Check the intended lossless assertion graph against a field-preserving reconstruction, including filler-local attributes and null/empty distinctions. Add ID-collision cases. Counts alone cannot establish fidelity.
- Inject a failure during relational overwrite. The existing output must survive; source inspection found that `Relational.py` currently unlinks it before constructing the replacement.
- Verify joins return the original passage bytes/identity, not a reconstructed claim rendered as evidence. Bind both assertion and passage inputs to the project.

These are focused boundary checks, not a whole-repository audit. Repair demonstrated failures, retain explicit limitations, and return to the evolving vertical.

**Failure/containment:** use temporary output directories, never canonical stores. Keep real inputs private when required. An unavailable dependency/provider is recorded with the failed command and exact unblock condition. Do not quietly substitute a mock and report the original criterion passed.

**Exit:** relevant tests execute, the bounded projection/evidence path executes, and remaining failures/limitations have explicit dispositions. This closes the focused baseline only, not a full MCP/provider certificate.

Existing core tests and MCP reuse/clean-rebuild canaries remain regression obligations for affected maintained surfaces. Run them on the integrated candidate with appropriate resources; a failure that affects the vertical blocks its claim. Missing credentials or a partial run must remain visible in final status rather than disappear behind a “green” label.

## Minimal Shared Contracts — Agree Once, Reuse

Do not add a generic framework. Reuse existing typed records wherever they fit. The new information needed across these batches is small:

**Artifact reference:** artifact ID, kind, concrete location, input snapshot/namespace, content digest or explicit non-persisted status, schema/version, and build state. The project manifest lists actual artifacts and capabilities, not every planned family.

**Execution record:** run/execution ID, operation + implementation revision, exact input/output artifact references, parameters/seed/provider/model where applicable, status, declared omissions/loss, and evidence links. Failed executions retain their input and error record but never advertise a successful output.

**Working set / result:** existing typed payload or stable artifact reference, selected canonical IDs, graph/projection scope, producing execution, and recoverable evidence. Reuse `SUBGRAPH`, `SCORE_VECTOR`, and `COMMUNITY_SET` where semantics match rather than inventing cognitive-state types.

**Native-tool boundary:** a small callable runtime and command entry point are sufficient for missing specialized operations. The harness may read Markdown or query SQLite with its existing tools. Record inputs and results actually observed from such calls; mark missing external steps as unobserved rather than fabricate complete lineage.

## Maximum Useful Observability, Not A Telemetry Product

One local JSONL stream plus referenced input/output files is sufficient initially. Instrument the **boundary wrappers**, not every function. Attach the same execution IDs to durable artifacts so diagnostic events and derivation records agree without pretending they are identical products.

Every boundary should expose:

- operation, revision, configuration, input artifact hashes and counts;
- start and terminal status, elapsed duration and provider call/usage facts when applicable;
- output artifact hashes, counts, selected IDs, and omission reasons;
- invariant failures with expected/actual values and the first relevant IDs;
- a reference to a small reproducible failing input and the relevant error/traceback.

Bound diagnostic samples; retain exact full intermediate artifacts only where needed for reproduction. Do not dump the entire corpus, credentials, or secrets into logs. Tracing overhead is part of the measurement, not free work.

Example diagnostic shape (illustrative, not an implemented API):

```json
{"operation":"graph.runtime_adapter","status":"failed","input_artifact":"graph:<digest>","check":"parallel_assertion_identity","expected_count":2,"actual_count":1,"first_mismatch":{"entity_ids":["entity:a","entity:b"],"missing_assertion_id":"assertion:2"},"reproducer_ref":"failures/case-01.json"}
```

**Traces answer what happened; independent checks answer whether it was correct.** Use producer fixtures, exact SQL/ID set comparisons, manually understandable graphs or direct library results, graph reconstruction, and deliberately broken inputs. A self-reported `invariants_ok: true` is not sufficient evidence.

Keep diagnostics and analytical lineage distinct: logs may expire or contain failed attempts; retained analytical outputs must continue to identify their inputs, method, parameters, implementation, and source evidence.

## Integration Acceptance Matrix

| Boundary | Required signal | Failure/control |
|---|---|---|
| Input → all projections | Exact scoped IDs and supported fields preserved or declared omitted | Duplicate/colliding identity and incompatible companion |
| SQL → graph → vectors → catalog | Same entity can be addressed without fuzzy rediscovery | Same label, different canonical IDs must remain distinct |
| Graph → working set | Actual maintained consumer returns only requested scope and retains contributing assertions | Parallel edges, unary/n-ary omission, disconnected and self-loop cases |
| Working set → analysis | Correct input domain, finite scores, valid partition, parameters/seed and graph semantics recorded | Unknown node, empty/disconnected graph, invalid weights, controlled library failure |
| Analysis → evidence | Selected IDs resolve to genuine source passages | Unknown evidence never becomes fabricated source text |
| Artifact → lineage | Real inputs/executions/outputs connect back to source/IR; failures not successful artifacts | Missing parent, changed input, incomplete external-tool observation |
| Whole project | Fresh process opens outputs; clean build and reuse agree on declared semantics | Stale snapshot rejected or explicitly rebuilt, failed rebuild preserves last good output |
| Harness journey | Native navigation + actual cross-representation and analytic composition visible in an execution trace | Fixed demo script alone is not evidence of autonomous composition |

Deterministic fixtures can test vector plumbing without paid calls, but those runs must be labeled accordingly. Actual embedding/index operation with the configured backend is required to claim the vector part observed; a document list or hash embedding substitute does not establish semantic retrieval.

For stochastic analytics, pin the supported seed and implementation, test partition validity and justified tolerances, and do not require identical community label numbers across unrelated versions. Similarity is not evidence entailment; centrality is not proof of social importance.

## Decisions And Assumptions

| Choice | Disposition | Meaning / response if wrong |
|---|---|---|
| Full text-derived represent/retrieve/analyze vision, onto-canon authority, native harness reuse | human_set | Preserve throughout; do not narrow to the latest topic |
| Approximately 1,000 LOC/hour authoring target | human_set | Measure rather than assert; no line-count completion gate |
| Single writer, SQLite first, small JSONL/manifest, shared growing runner | agent_decided_reversible | Reuse existing code and avoid new services; coordinate only when real parallel work appears |
| Existing Foundation projectors substantially reusable | assumption | Batch 0 determines repair scope; no “lossless” certification from names/docstrings |
| A suitable runner and configured embedding route can be obtained | assumption | Recheck access; failed Git DNS here does not prove all runners unavailable; no dependent source-only pileup |
| Real representative export and source companion can be used locally | assumption | Locate through the producer seam; synthetic tests do not replace the real outcome; never publish private data to this public repo |
| Generalized schema/ontology can be fully rendered from the current export | assumption, not a claim | Render observed vocabulary plus authoritative references; mark missing ontology context explicitly |
| Additional spend, public data release, production deployment, upstream authority changes | human_required only if activated | Not authorized by this planning revision |

**Human decisions now:** none required to start the baseline. The outcome and rapid operating model are already specified. Access, real-input availability, actual runtime failures, and verified throughput remain unresolved evidence questions.

## Later Horizon — Preserved, Not On The First Demo's Critical Path

After the integrated workflow is observed, broaden toward the existing roadmap: RDF/SPARQL where useful; specialized lexical/BM25 beyond native search; hierarchy/tree projections; additional graph/SNA and non-graph statistical methods; complete reference-method coverage; generic cross-representation discovery only when real consumers need it; stronger derivation queries/invalidation; Python/CLI/MCP convergence; legacy retirement; then comparative evaluation and scaling.

Raw-document ontology/chunking repairs remain a standalone compatibility workstream. They precede the governed-IR demo only when an observed shared failure actually blocks it. Geospatial remains out of scope. No benchmark novelty, internal planner, multi-agent platform, or polished UI is added to this slice.

## Evidence, Course Correction, And Continuation

The earlier Foundation/identity/SQL/graph tranche is retained in Git history and the baseline above. Its source-present state is not promoted to passed by this planning revision.

Planning-path validation was executed locally against byte-identical copies of Company Planning's validator (`cf839eb33f60e918cfe8199650e7cc6eaef97ddd`) and schema (`d56f8b6d51337f35c0f1e7d370e4948080049912`). The route record SHA-256 is `ed672bc50b4f1e9fce592562b1efd7f00de071d2ec513c7672ebb00094b9cef7`; it classified as durable_solo. Negative controls rejected a wrong selected path and an observed concurrent writer. This validates planning structure only, not DIGIMON code, factual assumptions, or live coordination state.

At each coherent batch record here: base/result revision, focused and integration commands/results, observed artifact, authoring versus repair/access effort, changed LOC, remaining failures, and exact next action. Keep existing CI/canary/failure plans as supporting evidence, not separate competing priorities.

Course-correct if two successive increments produce only supporting machinery, repeated failures add no diagnostic information, or an adapter grows into a second platform. Inspect the first missing boundary and return to an authentic run; do not create a new roadmap or approval checkpoint. Change architecture only when the concrete counterexample requires it.

**Exact next action:** obtain a usable checkout at the current revision, inspect local changes, and run the three existing projection contract files listed in Batch 0. Capture the first failure or scoped pass before extending the demo. This turn changes plans only; no product tests or feature implementation are claimed.
