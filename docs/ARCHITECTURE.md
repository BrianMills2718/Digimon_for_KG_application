# DIGIMON Architecture

**Status:** canonical target architecture  
**Updated:** 2026-09-17

## Architectural thesis

DIGIMON is a **general text-derived representation, retrieval, and analytics runtime**.

Its target architecture starts from governed semantic IR and does three things well:

1. **Represent** the same semantic core in complementary structures;
2. **Retrieve** bounded working sets/evidence using the strengths of those structures;
3. **Analyze / transform** those working sets into typed derived artifacts that can feed later operations.

Cross-representation identity and derivation lineage bind the whole system together.

The harness-first principle remains important, but it is a control boundary underneath this thesis:

> **Program the representations, capabilities, contracts, analytics, resources, and evidence/derivation boundaries. Let the external harness own adaptive strategy.**

See [VISION.md](VISION.md) for the durable product/research north star.

## Ecosystem boundary

```text
source text / source artifacts
        ↓
onto-canon6
semantic extraction + ontology binding + governance
identity/canonicalization + governed assertions + provenance
        ↓
Foundation-style governed semantic IR
        ↓
┌────────────────────────────────────────────────────────────┐
│ DIGIMON                                                    │
│                                                            │
│  REPRESENT           RETRIEVE           ANALYZE/TRANSFORM  │
│  projections         working sets       derived artifacts  │
└────────────────────────────────────────────────────────────┘
        ↓
grounded evidence / findings / reusable analytical state
        ↓
external harness / application / human workflow
```

### onto-canon6 owns

- source-backed semantic extraction/binding;
- ontology/profile semantics;
- review/governance and promoted assertions;
- canonical semantic identity and aliases;
- source/evidence provenance;
- governed semantic IR/export.

### DIGIMON owns

- derived representation/projection producers;
- specialized retrieval capabilities;
- analytic/transformation capabilities;
- cross-representation identity mappings;
- representation/schema/catalog metadata;
- evidence recovery;
- artifact/derivation lineage for downstream projections and analysis.

Raw-document ingestion/chunking can remain a standalone/benchmark/compatibility capability. It is not the conceptual ecosystem center.

## Representation architecture

Different data representations foreground different analytical/retrieval strengths. DIGIMON should project from the governed semantic core rather than force one universal physical model.

### Relational / tabular

Purpose:

- exact filtering;
- joins;
- grouping/aggregation;
- window/analytical queries;
- deterministic structured lookup.

A canonical relational projection should preserve semantic identifiers such as `entity_id`, `assertion_id`, predicate/role identity and source/evidence references.

### Property graph

Purpose:

- neighborhoods and paths;
- k-hop/subgraph extraction;
- PPR/diffusion;
- communities;
- centrality and structural analysis;
- subgraph optimization/transformation.

The property graph is a derived projection, not an independent semantic source of truth.

### Vector indexes

Purpose:

- semantic similarity;
- approximate nearest-neighbor retrieval;
- embedding-based entity/assertion/evidence search.

Vector representations deliberately trade explicit structure for similarity geometry. Canonical semantic IDs should survive in vector metadata so hits can be joined back to other projections and source evidence.

### Semantic graph / RDF

Where useful, the governed IR can project into a typed semantic/RDF representation for ontology-aware querying and SPARQL-style operations. RDF is conceptually distinct from the property graph: it foregrounds typed semantic statements and ontology relations rather than graph-algorithm convenience.

### Full-text / lexical indexes

Specialized lexical indexes are useful for BM25, exact terms, phrases, identifiers and fielded/boolean search when they add capability beyond the harness's native file/text search.

DIGIMON should not wrap ordinary grep/file search merely for symmetry.

### Hierarchy / tree

Purpose:

- abstraction levels;
- recursive summaries;
- taxonomy navigation;
- hierarchical retrieval.

Some hierarchies may be projected directly from ontology/type relations; other trees may be derived analytically through clustering/summarization and therefore require explicit derivation lineage.

### Wiki / progressive-disclosure representation

The wiki is a first-class **agent-navigation representation**, not merely another store.

It should materialize a progressive-disclosure view of both:

- the semantic content; and
- the retrieval/analysis environment itself.

It can expose:

- entity/concept/topic/source organization;
- available representations;
- schemas/ontologies for those representations;
- canonical IDs that join them;
- specialized capabilities available over each;
- links to deeper evidence/source material.

The wiki is descriptive, not prescriptive. It should not encode a mandatory workflow such as “SQL first, then graph.” The external harness decides the plan.

The harness can use its native file/search/link-navigation abilities over the generated wiki; DIGIMON need not provide trivial wrapper tools unless specialized state/capability is required.

### Source / evidence representation

Exact source artifacts, claims, chunks/spans and evidence identifiers form the grounding layer underneath every derived representation.

## Cross-representation identity invariant

A representation is easier to compose when semantic identity survives projection.

Where possible, preserve:

- `entity_id`;
- `assertion_id`;
- `predicate_id`;
- source identity/reference;
- evidence/span identity.

A canonical entity should therefore be directly addressable across SQL, property graph, vector metadata, semantic graph and wiki/catalog metadata.

Projection-specific fields—vector IDs, implementation row IDs, wiki paths, communities, centrality scores—remain downstream unless they carry genuine semantic authority.

## Retrieval architecture

Retrieval asks: **which bounded working set or evidence do I need from a representation?**

Capabilities include or may include:

- exact structured queries and SQL;
- vector similarity search;
- lexical/BM25 search;
- exact entity/assertion lookup;
- graph neighbor/path/k-hop/subgraph retrieval;
- PPR/diffusion;
- community retrieval;
- tree/hierarchy navigation;
- exact source/evidence recovery.

The project remains conditional rather than graph-first. A simple fact should not require graph traversal merely because a graph exists.

## Analytics / transformation architecture

Analytics are a first-class DIGIMON capability family.

The canonical pattern is:

```text
retrieval artifact / working set
        ↓
analytic transformation
        ↓
derived typed artifact
        ↓
further retrieval / analysis / interpretation
```

### Graph analytical families

Examples include:

- degree, betweenness, eigenvector and PageRank centrality;
- Leiden/Louvain and other community detection;
- connected components;
- k-core/cohesion/density/assortativity;
- brokerage/bridge/structural-hole measures;
- shortest paths, motifs and diffusion;
- PCST, Steiner and related subgraph transformations.

### Non-graph analytical families

Where supported by represented data, examples include:

- SQL aggregation and descriptive statistics;
- distributions and group comparisons;
- correlations;
- clustering and dimensionality reduction;
- regression/classification;
- anomaly detection;
- temporal aggregation/trend analysis.

Analytic results are **derived state**, not source assertions. They should retain method, parameters/configuration, input artifacts, representation/version and evidence/uncertainty where applicable.

## Typed composability

The current operator core uses seven `SlotKind` values:

- query text;
- entity set;
- relationship set;
- chunk set;
- subgraph;
- community set;
- score vector.

These are reusable data semantics, not cognitive states.

The type system should grow only when real reusable capabilities require it. Likely future concepts include `TABLE`, `VECTOR_SET`, `MODEL` or `FINDING_SET` if and when concrete operators need those distinctions.

Illustrative compositions:

```text
ENTITY_SET → k-hop → SUBGRAPH → Leiden → COMMUNITY_SET
SUBGRAPH → centrality → SCORE_VECTOR → top-k → ENTITY_SET → evidence
TABLE → aggregate → TABLE
QUERY_TEXT → vector search → ENTITY_SET
```

Composition exists to connect capabilities safely; it is not a hidden global planner.

## Capability descriptors

A canonical capability should eventually describe at least:

- stable capability ID/version;
- human-readable purpose;
- typed inputs/outputs;
- representation/resource prerequisites;
- produced/modified resources/artifacts;
- deterministic vs model-assisted behavior;
- analytical method/parameter semantics where applicable;
- cost/side-effect/lossiness characteristics;
- failure classes;
- implementation binding;
- evidence/provenance/derivation behavior.

`Core/Schema/OperatorDescriptor.py` and `Core/Operators/registry.py` are current foundations. Build/config/analysis/cross-modal surfaces should map into the same conceptual model rather than forming undocumented capability universes.

## Harness boundary

### External harness owns

- interpretation of the user's goal;
- representation/tool selection;
- decomposition when useful;
- sequencing, branching, parallelization and retries;
- comparing observations across representations;
- strategy revision;
- stopping policy;
- communicating uncertainty/gaps.

### DIGIMON owns

- projections/representations;
- specialized retrieval/analytic methods;
- typed capability contracts;
- compatibility/prerequisite/resource facts;
- cross-representation identity facts;
- source/evidence identifiers;
- artifact/derivation lineage;
- bounded local model-assisted transformations when a capability intrinsically requires semantic judgment;
- reference plans as inspectable conveniences.

DIGIMON should not recreate native harness abilities such as opening files, following Markdown links or ordinary text search merely for API symmetry.

## CLI, Python and MCP

The target is one maintained runtime with several access surfaces:

- **CLI** — human-facing shell;
- **Python runtime** — developer/application library;
- **MCP/tool protocol** — agent-facing specialized capability interface.

MCP is not the product. `digimon_mcp_stdio_server.py` is currently the strongest modern external-harness surface, while the current CLI still instantiates older planner/orchestrator components and is transitional.

All surfaces should converge on the same representation/retrieval/analytics core.

## Reference methods

The ten named methods remain useful as:

- known compositions;
- regressions/baselines;
- simple-client shortcuts;
- optional routing targets;
- research comparison units.

They are not DIGIMON's identity and should not limit custom mixed-method composition.

## AoT / GoT / ReAct policy

AoT, GoT, ReAct and decomposition are optional reasoning heuristics, not mandatory runtime state machines.

The harness may follow, merge, reorder, branch, parallelize, revise or ignore decomposition suggestions.

A formal dependency graph should be introduced only when it enables a concrete system capability such as scheduling, resumability, caching, auditing or derivation tracking—not merely because reasoning can be drawn as a DAG.

## Evidence and derived-state model

DIGIMON should distinguish:

1. **source-backed semantic state** — governed assertions and exact source evidence;
2. **retrieval artifacts** — entity sets, tables, subgraphs, chunks/evidence sets;
3. **derived analytical artifacts** — communities, score vectors, centralities, fitted outputs, forecasts, classifications;
4. **findings/interpretations** — downstream judgments grounded in evidence plus derived analytical state.

A synthesizer/final-answer capability should consume structured evidence and should not reconstruct provenance from prose where avoidable.

## Provenance / derivation architecture

Three related provenance layers remain distinct:

### Evidence provenance

What exact source text supports an assertion or answer?

### Semantic provenance

How did governed semantic state arise from candidates/source evidence?

### Artifact / derivation lineage

What exact prior artifacts and transformation executions produced a projection, retrieval artifact, analytic result or finding?

Target lineage:

```text
source resource
  ↓ acquisition/capture
immutable source artifact/version
  ↓ semantic extraction/governance
governed semantic IR
  ↓ projection execution
SQL | graph | vectors | RDF | tree | wiki | lexical index
  ↓ retrieval execution
bounded working set / subgraph / table / entity set
  ↓ analytic transformation
community | score vector | model/result | derived artifact
  ↓ interpretation
finding
```

Every important derived artifact should eventually expose:

- exact input artifact identities/versions/hashes;
- transformation/method identity/version;
- parameters/configuration;
- output artifact identity/version;
- declared scope/lossiness where relevant;
- source/evidence lineage.

Recursive lineage should terminate at reopenable immutable source evidence, explicit human input or declared root observations.

The **derivation graph is not the domain graph**. The domain graph models subject matter; the derivation graph models artifacts, executions, projections, methods, parameters and findings.

This lineage supports reproducibility, source recovery, stale-artifact detection, invalidation and auditing.

## Resource / artifact lifecycle

The architecture should remain failure-driven rather than inventing an enterprise resource-governance framework prematurely.

Current practical lifecycle mechanisms—stable IDs/namespaces, graph/chunk manifests, derived-artifact invalidation, active dataset/graph selection and fail-closed stale-community behavior—are useful foundations.

Where further lifecycle abstraction is justified, a resource/artifact description should expose:

- stable ID and kind;
- representation/schema;
- dataset/source identity;
- build/transformation fingerprint;
- dependencies/input artifacts;
- state/currentness;
- producer/method/version;
- derivation lineage;
- reuse/rebuild/invalidation semantics.

Add generalized machinery only when several concrete failures require the same missing boundary.

## Error model

Failures must be machine-actionable. At minimum distinguish:

- resource/artifact not found;
- prerequisite absent;
- stale/incompatible representation;
- invalid plan/wiring/type;
- provider/model failure;
- empty retrieval result;
- likely extraction incompleteness/unsupported evidence;
- unsupported/lossy conversion;
- analytic-method failure;
- timeout/internal failure.

A returned empty set and an execution failure are not equivalent.

## Evidence → Action orientation

The architecture should support reusable analytical work:

```text
evidence → representation → retrieval → analysis/transformation → review/interpretation → finding/action
```

The durable residue should include:

- evidence and provenance;
- structured knowledge;
- methods and parameters;
- findings and uncertainties;
- open questions.

> **Every analysis should make the next one easier.**

## Legacy and compatibility policy

The repository contains valuable older implementations. They should be classified rather than silently mixed into the target architecture.

Legacy/transitional examples include:

- `Core/AOT/` programmed atomic-state/transition logic;
- `Core/AgentBrain/` broad internal planning logic;
- multiple `Core/AgentOrchestrator/` implementations;
- older `Core/MCP/` server/client/coordination experiments;
- CLI paths that still instantiate internal planner/orchestrator code.

New architecture work should not deepen dependencies on those layers unless a deliberate supported path requires them.

## Design invariants

1. **Governed IR upstream:** semantic authority remains upstream of DIGIMON projections.
2. **Represent → Retrieve → Analyze:** all three capability planes are first-class.
3. **Mixed representations:** no single physical representation defines the system.
4. **Shared identity:** canonical semantic IDs survive projections wherever possible.
5. **Harness-first control:** adaptive strategy belongs to the capable caller by default.
6. **Do not rebuild native harness tools:** expose specialized capability/state, not wrappers for symmetry.
7. **Typed composability:** retrieval and analytic outputs can safely become later inputs.
8. **Derived state is explicit:** analytic outputs are distinguishable from source evidence.
9. **Derivation lineage:** projections/retrievals/transformations/findings are recursively traceable.
10. **Conditional graph use:** graph structure is used when it adds value, not by default.
11. **Reference methods are optional:** named pipelines are conveniences/baselines, not system identity.
12. **Multiple public surfaces, one core:** CLI/Python/MCP should converge on the same maintained runtime.
13. **Graceful incompleteness:** missing structure/evidence produces explicit gaps/fallback opportunities, not false conclusions.
14. **Failure-driven abstraction:** generalize only when concrete repeated failures justify it.
15. **Documentation must preserve the whole thesis:** use [DOCUMENTATION_COVERAGE.md](DOCUMENTATION_COVERAGE.md) during reconciliation.

## Non-goals for the current architecture phase

The current phase is not primarily about:

- maximizing benchmark scores;
- claiming research novelty;
- building a new general-purpose cognitive/agent brain;
- creating a multi-agent society/coordination framework;
- forcing every question through graph reasoning;
- wrapping native harness file/search/navigation operations;
- adding geospatial representation;
- building enterprise lifecycle/governance machinery before concrete failures justify it.

## Relationship to canonical docs

- [VISION.md](VISION.md) — durable north star.
- [CURRENT_STATE.md](CURRENT_STATE.md) — how much of this architecture exists now.
- [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md) — exact module classification and implementation caveats.
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — distance from current state to this target.
- [ROADMAP.md](ROADMAP.md) — closure order and exit criteria.
- [DOCUMENTATION_COVERAGE.md](DOCUMENTATION_COVERAGE.md) — whole-thesis reconciliation checklist.
