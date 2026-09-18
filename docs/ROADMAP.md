# DIGIMON Roadmap

**Updated:** 2026-09-17  
**Scope:** realize [VISION.md](VISION.md): governed text-derived IR → **Represent → Retrieve → Analyze/Transform** → evidence/findings, with shared identity and derivation lineage.

## Current Execution Frontier

The active execution sequence is [NORTH_STAR_VERTICAL_SLICE_PLAN.md](planning/NORTH_STAR_VERTICAL_SLICE_PLAN.md), revision 2. It supersedes earlier calendar estimates and duplicated immediate-action lists. This roadmap retains the full capability horizon; the living plan owns the next runnable batch and its evidence.

The contributor selected rapid coherent implementation batches, targeting approximately **1,000 authored code/test lines per active authoring hour**, followed immediately by execution, trace inspection, and repair. That rate is a target to measure, not a delivery promise or quality metric. The first integrated slice is provisionally **3,000–5,000 additional authored implementation/test lines**, not including generated artifacts or planning prose.

The ordered frontier is:

```text
0. Execute and repair the existing Foundation/identity/SQL/graph baseline
1. Save/reopen one project and run SQL → evidence with minimal trace/lineage
2. Connect Foundation graphs to maintained graph retrieval
3. Build/query/reload vectors through the existing embedding/index path
4. Generate the progressive-disclosure semantic + operational catalog
5. Retrieve a subgraph → Leiden/centrality → select entities → recover evidence
6. Converge the same growing journey and observe a real harness using it
```

Graph and vector work can be reordered after the shared seam is stable. Analytics can follow the graph before catalog polish. This is one writer by default, not a claim of active parallel execution.

**Two execution changes matter:** tracing/lineage starts with the first artifact instead of arriving after all features; integration happens every batch instead of being deferred to a final phase. Do not accumulate dependent source-only batches when the current batch cannot be run.

The broad deterministic core suite and MCP reuse/clean-build canaries remain supported-surface regression obligations. They are not substitutes for the actual governed-IR workflow. Missing Actions runs alone establish neither correctness nor failure; use an available suitable runner and record the exact scope.

## Existing Foundation, Not Work To Rebuild

At inspected base `d548f0c1f84e3450c252eedea8f3a02abf0f3520`, `Core/Projection/` already contains a Foundation IR consumer, identity manifest, normalized SQLite projection, and two property-graph projectors. Their tests exist but have not been executed in this planning revision. Runtime adoption, field fidelity, and failure behavior still need evidence.

Keep SQLite as the first relational backend. A DuckDB migration is not a prerequisite. Preserve the role-aware assertion graph separately from the derived entity-network graph; no implicit n-ary clique expansion or silent loss of parallel assertion identity is allowed. See [FOUNDATION_PROPERTY_GRAPH_DESIGN.md](planning/FOUNDATION_PROPERTY_GRAPH_DESIGN.md).

## Full Capability Horizon

The rows below preserve the original roadmap's goals. They describe what must eventually be true, not a mandate to finish one whole plane before demonstrating the next.

| Area | Desired result and retained work | Evidence / promotion trigger |
|---|---|---|
| **Documentation truth** | Separate vision, target design, current implementation, gaps, execution plan, and historical material; use [DOCUMENTATION_COVERAGE.md](DOCUMENTATION_COVERAGE.md) | Canonical set covers all three planes, wiki, identity, harness boundary, surfaces, and provenance without confusing source inspection with runtime proof |
| **Current runtime correctness** | Execute deterministic contracts and relevant reuse/rebuild canaries; repair observed failures rather than continue source-only audits | Exact revision/environment and inspectable results; failures or unrun checks explicitly reported |
| **Governed IR seam** | Consume actual onto-canon export; preserve canonical IDs, n-ary roles, literals, qualifiers, aliases, source scope, evidence, and supported-version rules | Producer-compatible fixture plus real authorized export; no invented semantics or silent field loss |
| **Shared identity** | Preserve entity/assertion/predicate/source/evidence IDs and explicit scope across representations; retain mappings for projection-local IDs | Cross-representation lookup succeeds without fuzzy rediscovery; same name/different identity remains distinct |
| **Relational/tabular** | Reuse SQLite projection; exact filters, joins, grouping, aggregation and relevant analytical SQL; expose actual schema and typed downstream results | Native SQL query and aggregate feed later retrieval/analysis and exact evidence recovery |
| **Graph and vector** | Adopt IR projections through maintained graph/VDB consumers; state graph direction, edge multiplicity, weights, omission policies and index model/metric/dimension | Retrieval, persistence/reload, identity, evidence and compatible resource version are observed; vector-document generation alone does not count as indexing |
| **Wiki/progressive disclosure** | Generate semantic pages plus an environment map of actual representations, schemas/ontology references, join IDs, capabilities and evidence locations | Native file/navigation tools can discover content and move to real SQL/graph/vector artifacts; unavailable resources are not advertised as built |
| **Analytics/transformation** | Promote existing SNA machinery first: centrality, communities/Leiden, paths, PCST/Steiner, diffusion, components, k-core/cohesion/density, brokerage/assortativity as useful. Add table/statistical methods, clustering, anomalies, trends and model fitting from concrete needs | Working sets enter methods and typed outputs enter subsequent operations; methods/parameters/scope retained, derived state distinct from source evidence |
| **Artifact/derivation lineage** | Distinguish evidence provenance, semantic provenance, and artifact/execution lineage. Begin with small persisted records; expand traversal, reproducibility and invalidation as needed | Source/IR → projection → retrieval → analysis → finding is recoverable through actual execution records, with loss and missing external observations explicit |
| **Resource correctness** | Bind graph-derived matrices/indexes to exact graph/snapshot identity; preserve practical manifests and invalidation; retain last good output on failed rebuild; extend to new artifacts only when needed | Same-shaped stale resources cannot silently substitute; failed rebuild and changed-input controls work |
| **Public surfaces** | One maintained core with Python/application, CLI, and MCP access. A small bridge may serve the demo now; full CLI modernization and broad parity follow | Equivalent supported operations behave consistently; the legacy internal planner is not silently invoked as the default brain |
| **Errors and legacy consolidation** | Distinguish missing/incompatible/stale input, empty evidence, invalid wiring, provider failure, unsupported/lossy conversion and internal failure; identify live callers before retiring old AgentBrain/AoT/orchestrator/MCP variants | Machine-actionable recovery facts; no competing architecture or fake successful output on failure |
| **Remaining text-derived representations** | RDF/semantic queries, hierarchy/tree projection and specialized lexical/BM25 beyond native search remain in the vision; add useful native operation families, not arbitrary engines for symmetry | An actual use case/consumer and a bounded projection/retrieval check justify each addition |
| **Reliability, incremental semantics and evaluation** | Extend deterministic projection/identity/analytic/lineage/parity checks; keep live-provider tests separate. Preserve conflicts/validity and define supported updates. Later compare graph, vector, SQL, lexical and wiki approaches, adaptive vs reference methods, cost and incomplete-graph behavior | Integrated architecture works before comparative benchmarking becomes the driver; each claimed update/temporal behavior has relevant evidence |

The current first proof uses SQLite, graph, vectors, Markdown navigation, graph analytics, a SQL aggregation, exact evidence, and minimal lineage. It does not remove RDF, lexical, hierarchy, broader analytics, remaining reference methods, or surface convergence from the longer-term vision.

## Scope And Ownership Rules

**onto-canon6** retains extraction, semantic binding, governance, canonical identity, and governed export. **DIGIMON** owns derived representations, specialized retrieval/analysis, and lineage. **The external harness** owns adaptive selection, sequencing, retries, branching, comparison, and stopping.

AoT/GoT/ReAct remain optional heuristic guidance, not a programmed mandatory planner. Do not implement `wiki.open`, `wiki.follow`, basic grep/search wrappers, an SQL reasoning agent, or a new orchestrator merely because the catalog describes those operations. Reuse native harness tools and existing engines.

The domain graph and provenance graph answer different questions. A centrality, community assignment, forecast or model output is derived state, not a newly governed source fact. Findings should state analytical scope and uncertainty; a numerical score does not establish causal influence.

Raw-document chunking and custom-ontology extraction remain supported standalone/compatibility work. They enter the governed-IR critical path only when a demonstrated shared failure blocks it. Geospatial is outside the text-focused scope.

No generalized telemetry/resource framework, multi-agent platform, new dashboard, production scaling, or benchmark-specific architecture is a prerequisite for the first vertical. Broader functionality is added through actual consumers rather than speculative machinery.

## Evidence And Authority

[VISION.md](VISION.md) owns project purpose; [ARCHITECTURE.md](ARCHITECTURE.md) owns target invariants; [CURRENT_STATE.md](CURRENT_STATE.md) and [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md) describe code reality; [GAP_ANALYSIS.md](GAP_ANALYSIS.md) owns the broader distance to target.

The [living execution plan](planning/NORTH_STAR_VERTICAL_SLICE_PLAN.md) owns batch readiness, focused checks, the observation/repair loop, unresolved assumptions and exact next action. [PLANNING_SUMMARY.md](PLANNING_SUMMARY.md) is a short navigation view, not another schedule. Existing canary/CI/failure plans remain supporting workstreams under that frontier.

Full evaluation questions remain in [FUTURE_EVALUATION_QUESTIONS.md](FUTURE_EVALUATION_QUESTIONS.md). Technical tests, stakeholder-reviewable artifacts, and a stakeholder-observed useful result remain separate kinds of evidence.
