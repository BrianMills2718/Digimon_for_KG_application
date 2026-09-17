# DIGIMON Gap Analysis

**Updated:** 2026-09-17  
**Purpose:** identify the concrete distance between [CURRENT_STATE.md](CURRENT_STATE.md) and the full target in [VISION.md](VISION.md) / [ARCHITECTURE.md](ARCHITECTURE.md).

The current priority is **make the maintained core demonstrably correct, then close the highest-value gaps in the full Represent → Retrieve → Analyze vision**. Benchmark optimization and novelty claims remain later work.

## Priority model

- **A — foundational:** required for the system to be trustworthy and for later representation/analytics work to compose cleanly.
- **B — vision completion:** closes major missing planes or representation families in the north star.
- **C — consolidation/hardening:** converges surfaces, errors, tests and legacy code after the core path is trustworthy.
- **Later:** research validation, benchmarking and optimization.

## Gap matrix

| Priority | Area | Target | Current state | Main gap | Next action |
|---|---|---|---|---|---|
| **A** | Current-head runtime proof | Deterministic maintained core is freshly executable | Many source-level contract tests exist; current GitHub writes have produced no new Actions runs | We cannot honestly call current head green | Run minimal install + `tests/core` + reuse/rebuild MCP canaries on a real runner; fix first red before broader refactors |
| **A** | Governed IR seam | onto-canon/Foundation-style governed IR is the canonical ecosystem input | Compatibility DIGIMON JSONL/import paths exist; raw corpus path is much more exercised inside current DIGIMON | The canonical upstream handoff is not yet the dominant tested build path | Verify current Foundation IR fields and implement/test deterministic projections without inventing semantics |
| **A** | Cross-representation identity | Canonical entity/assertion/source/evidence IDs survive every projection | Graph/VDB/evidence paths preserve many IDs | No explicit invariant/test matrix spans relational, graph, vector, wiki/catalog and later semantic projections | Define cross-representation identity rules and make projection tests enforce them |
| **A** | Derivation lineage | Every important projection/retrieval/analytic artifact records inputs + execution + parameters + output identity | Evidence IDs, producer metadata, manifests and invalidation are real foundations | No first-class artifact/execution lineage graph across the full analytical workflow | Define the smallest derivation record/edge model and add it first to projection/build + analytic outputs |
| **A** | Grounding/evidence | Source-backed outputs fail closed and remain reopenable | Strongly improved for maintained answer/retrieval paths | Not every future analytic/finding type has evidence/lineage semantics yet | Treat source evidence vs derived state as a universal contract when expanding types |
| **A** | Sparse resource identity | Derived matrices cannot silently bind to the wrong graph | Shape checks and ER invalidation catch many stale cases | Same-shaped cross-graph reuse remains possible | Bind sparse artifacts to graph identity/version with the smallest explicit check |
| **A** | Custom ontology path | Selected ontology actually reaches maintained extraction | Config/ontology material exists | Override/load path is not cleanly verified | Complete path override → load → extraction and add a contract test |
| **B** | Representation plane | One governed semantic core can be projected into complementary useful forms | Graph/vector/tree/community are strongest; table/vector conversion experiments exist | Relational, wiki/catalog, semantic graph and specialized lexical representations are not first-class canonical projections | Build representations incrementally from real use cases, beginning with relational/tabular and wiki/catalog |
| **B** | Relational/tabular projection | Stable schema suitable for exact SQL/analytics with canonical IDs | DataFrame/table conversion code exists | No canonical Foundation IR → database schema + maintained query surface | Define normalized relational schema for entities/assertions/roles/evidence/aliases and prove round-trip identity |
| **B** | Agent wiki/catalog | Progressive-disclosure map of semantic content, representations, schemas, IDs, capabilities and evidence | Vision documented only | No generated artifact exists | Generate a minimal deterministic wiki/catalog from one governed fixture; rely on harness-native file/link search rather than building wrapper APIs |
| **B** | Analytics capability plane | Retrieved working sets can feed reusable analytical transformations with typed outputs | Graph analysis is substantive but scattered/retrieval-centric | No comprehensive typed analytics taxonomy or derived-artifact model | Inventory existing graph analytics first; promote real methods such as centrality/community/structural measures into typed descriptors before adding new methods |
| **B** | Derived analytic artifacts | Communities, score vectors, models/findings are explicitly derived state with method/parameter lineage | Community/score outputs exist; provenance metadata is uneven | Derived state is not modeled uniformly across methods | Add reusable derived-artifact metadata/type semantics when analytic catalog is promoted |
| **B** | Semantic/RDF projection | Ontology-rich semantic querying can be added when useful | Upstream IR has predicates/roles/types/aliases/hierarchy information | No maintained RDF/SPARQL projection | Prototype only after relational/wiki seams prove the projection contract; do not make it a prerequisite for current core |
| **B** | Specialized lexical retrieval | BM25/fielded search exists only where it adds value beyond harness-native search | No canonical specialized lexical index | Need is use-case dependent | Add only when scale/fielded-ranking need demonstrates native text search is insufficient |
| **B** | Capability discovery | Harness can understand specialized retrieval + analytics without multiple undocumented universes | Typed registry is strong; some build/config/analysis tools sit outside it | Descriptor coverage does not yet span full future Represent/Retrieve/Analyze plane | Extend descriptor concept when concrete non-operator capabilities are promoted; avoid a speculative mega-schema |
| **B** | Public Python runtime | Clean application/developer library over maintained core | Implementation exists across Core but not one obvious stable API | Current agent-facing MCP surface is more coherent than library surface | Introduce narrow `digimon` library entry points for projection/resource access and typed execution |
| **B** | CLI convergence | Human CLI uses the same maintained runtime | Current CLI uses `PlanningAgent`/`AgentOrchestrator` | User surface tells an older architecture story | Rebuild/adapter CLI on the public runtime; keep agent reasoning outside DIGIMON by default |
| **C** | Error semantics | Machine-actionable distinctions for empty evidence, missing resource, bad wiring, provider failure, etc. | Many fail-open cases fixed but conventions still vary | Harness recovery still has tool-specific details | Define a small result/error convention for maintained public surfaces |
| **C** | Legacy isolation | Old cognitive/orchestration generations cannot be mistaken for current architecture | Legacy code still has live/transitional callers | Cognitive ownership remains visually ambiguous in tree | Classify callers; isolate/deprecate unused paths after CLI/runtime convergence |
| **C** | Test taxonomy | Tests map visibly to Represent/Retrieve/Analyze + identity/lineage boundaries | Large and growing deterministic core suite | No simple coverage ledger for the reconciled vision | Add a test/coverage matrix keyed to VISION/DOCUMENTATION_COVERAGE |
| **C** | CI trust | Supported deterministic contracts block regressions | Workflow exists but current execution signal is absent | Documentation cannot point to fresh certification | Restore an executable runner/Actions path and keep provider-expensive suites separate |
| **Later** | Evaluation | Measure when each representation/method helps and what composition costs | Evaluation infrastructure exists | Architecture and representation planes still evolving | Benchmark after canonical projections/analytics are stable |

## The gaps that matter most now

### 1. Runtime proof is the immediate bottleneck

Many previously documented correctness gaps are no longer accurate: validation is stricter, control-flow typing was repaired, reference methods were rewired, grounding fails closed, and resource invalidation/manifests now exist. The highest-value next step is therefore not another abstraction—it is a real execution signal.

### 2. The canonical onto-canon → DIGIMON seam must become real, not merely documented

The north star assumes governed semantic IR is the ecosystem input. DIGIMON currently has stronger standalone raw-document machinery than canonical cross-repo projection coverage. We need to prove that the IR carries what each projection needs and preserve canonical IDs/provenance through every projection.

### 3. Representation breadth is behind retrieval depth

DIGIMON is currently strongest in property-graph/vector/tree/community representations. The full vision also requires at least a canonical relational projection and an agent-readable progressive-disclosure wiki/catalog. Those should be treated as representation artifacts, not as excuses to duplicate native harness navigation/search tools.

### 4. Analytics are real but not yet a first-class organized plane

Graph analysis is part of DIGIMON's lineage, not an optional afterthought. Existing PPR, community, subgraph, PCST, Steiner and score operations prove the pattern, but the documentation/API do not yet present a coherent analytical suite. The next move is to inventory and type the real existing methods before adding a broad new toolbox.

### 5. Provenance must grow from citations into derivation lineage

Final-answer evidence provenance is much healthier now. The remaining target is broader:

```text
source/artifact
→ governed IR
→ projection execution
→ representation version
→ retrieval execution
→ bounded working set
→ analytic execution + parameters
→ derived artifact
→ finding
```

This lineage should support reproducibility and stale-artifact detection without confusing the derivation graph with the domain/property graph.

### 6. Shared identity is the glue for agentic composition

The harness should be able to discover Alice in a wiki, use the same canonical ID in SQL, then traverse the corresponding graph node or query vector metadata without fuzzy rediscovery. That requires explicit projection invariants more than it requires a smarter router.

## Architectural decisions that should not be accidentally reopened

- onto-canon owns governed semantic authority; DIGIMON owns downstream retrieval/analytic projections;
- raw-document ingestion is useful standalone/compatibility functionality, not the conceptual ecosystem center;
- DIGIMON is **Represent → Retrieve → Analyze**, not just GraphRAG and not just an MCP server;
- property graphs are important because of graph retrieval/analytics, but the system is representation-general;
- the wiki is a progressive-disclosure semantic/environment map, not merely a document store;
- do not wrap native harness file/search/link-navigation capabilities merely for symmetry;
- the harness owns adaptive planning, sequencing, retries, branching and stopping;
- DIGIMON owns specialized capabilities, typed contracts, resources, identity, evidence and derivation facts;
- AoT/GoT/ReAct remain optional heuristics, not the core cognitive runtime;
- CLI, Python and MCP should converge on the same maintained core;
- analytic outputs are derived state and must not masquerade as original evidence;
- the domain graph and derivation/provenance graph are distinct;
- do not build a generalized resource/governance framework when a concrete identity/invalidation rule solves the observed failure;
- do not let benchmarks define the architecture prematurely.

## What “ready for serious evaluation” means

Before broad benchmarking becomes a primary activity, the project should be able to answer:

1. Can a governed IR fixture deterministically produce each supported canonical representation?
2. Which canonical IDs connect those representations?
3. Can the harness discover what representations/schemas/capabilities exist without hidden implementation knowledge?
4. Can it retrieve a bounded working set from one representation and feed it into a compatible analytic operation?
5. Can derived analytic outputs be reused as inputs to later operations?
6. Can every material result distinguish source evidence from derived state?
7. Can the system recursively explain which source/prior artifacts and transformation executions produced an important derived artifact?
8. Does a changed source/IR invalidate the right projections without destroying unrelated resources?
9. Can CLI, Python and MCP reach the same maintained capability core?
10. Does the current deterministic suite and clean-rebuild canary actually pass on the current head?

The roadmap should be read as the ordered closure plan for these questions.
