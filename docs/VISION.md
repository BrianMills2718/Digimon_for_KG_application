# DIGIMON Vision

**Status:** canonical product/research north star  
**Updated:** 2026-09-17

## One-line thesis

**DIGIMON is a general text-derived representation, retrieval, and analytics runtime: it consumes governed semantic IR, projects that semantic core into complementary data representations, exposes specialized retrieval and analytic capabilities over those representations, and preserves cross-representation identity and derivation lineage so an intelligent external harness can compose its own evidence-to-action workflow.**

The recent harness-first refactor is an architectural strategy underneath this vision. It is not the project thesis by itself.

## Ecosystem boundary

DIGIMON is downstream of semantic governance.

```text
textual/source evidence
        ↓
onto-canon6
semantic extraction + ontology binding + governance
identity/canonicalization + governed assertions + provenance
        ↓
Foundation-style governed semantic IR
        ↓
DIGIMON
represent → retrieve → analyze/transform
        ↓
grounded evidence, derived analytical artifacts, findings
        ↓
external intelligent harness / application / human workflow
```

### onto-canon6 owns

- source-backed semantic extraction and binding;
- ontology/profile semantics;
- review/governance and promoted assertions;
- canonical semantic identity, aliases and recanonicalization;
- source/evidence provenance;
- the governed semantic IR/export consumed downstream.

### DIGIMON owns

- derived retrieval/analytic representations of that governed semantic core;
- specialized retrieval capabilities over those representations;
- composable analytic/transformation capabilities;
- shared identity mappings across representations;
- representation/catalog metadata that makes the environment legible to an agent;
- evidence recovery and derivation lineage through projections, retrievals, transformations and analytical outputs.

Raw-document ingestion/chunking may remain useful for standalone operation, experiments, benchmarks and compatibility, but it is not the conceptual center of the ecosystem architecture.

## Three capability planes

### 1. Represent

Project the same governed semantic core into structures that foreground different kinds of reasoning.

Primary representation families are:

- **Relational/tabular** — exact structured filtering, joins, grouping, aggregation and analytical SQL;
- **Vector** — semantic similarity and nearest-neighbor retrieval;
- **Property graph** — paths, neighborhoods, diffusion, communities and structural algorithms;
- **Semantic graph / RDF** — ontology-aware and typed semantic querying where useful;
- **Full-text / lexical index** — BM25, exact terms, phrases, identifiers and fielded/boolean retrieval when this adds capability beyond the harness's native file search;
- **Hierarchy/tree** — abstraction levels, recursive summaries, taxonomies and hierarchical retrieval;
- **Wiki / progressive-disclosure knowledge surface** — agent-oriented semantic/navigation structure over the knowledge environment;
- **Source/evidence representation** — reopenable exact source material, claims and evidence spans.

Geospatial representation is outside the current text-focused vision.

The IR does not need to physically resemble every representation. It needs enough semantic identity, roles, predicates, evidence and provenance to project into them without inventing meaning.

## Cross-representation identity

Derived representations should preserve canonical identity wherever possible.

Examples include:

- `entity_id`;
- `assertion_id`;
- `predicate_id`;
- `source_ref`;
- `evidence_id` / evidence-span identity.

A canonical entity such as `entity:alice-smith` should be cheaply addressable across SQL rows, graph nodes, vector metadata, semantic-graph identifiers and wiki metadata. The purpose is not uniform storage; it is reliable composition across heterogeneous projections.

Projection-specific state belongs downstream in DIGIMON rather than in the canonical semantic IR unless it carries genuine semantic authority. Examples include vector IDs, community assignments, PageRank scores, wiki paths and SQL implementation row IDs.

## 2. Retrieve

Retrieval asks: **which bounded working set or evidence do I need from a representation?**

Examples include:

- SQL filters, joins and structured queries;
- vector similarity search;
- lexical/BM25 search;
- exact entity/assertion lookup;
- graph k-hop expansion;
- path/subgraph retrieval;
- PageRank/PPR-style diffusion;
- community retrieval;
- tree/hierarchy expansion;
- exact source/evidence recovery.

Different representations foreground different retrieval strengths. DIGIMON should make them composable rather than forcing every question through a graph pipeline.

## 3. Analyze / transform

DIGIMON is also an analytical-method suite, not only a retrieval system.

The canonical pattern inherited from computational social-network analysis is:

```text
retrieve a bounded working set
        ↓
apply an analytic/transformation method
        ↓
produce a typed derived artifact
        ↓
use that result as input to further retrieval or analysis
```

Graph-oriented examples include:

- degree, betweenness, eigenvector and PageRank centrality;
- Leiden/Louvain and other community detection;
- connected components;
- k-core / cohesion / density / assortativity;
- brokerage and structural-hole measures;
- shortest paths, motifs and diffusion analysis;
- PCST, Steiner and other subgraph transformations.

Non-graph examples can include:

- SQL aggregation and descriptive statistics;
- distributions, correlations and group comparisons;
- clustering and dimensionality reduction;
- regression/classification where appropriate;
- anomaly detection;
- temporal aggregation and trend calculations over text-derived structured data.

The system should treat analytic outputs as **derived state**, not original evidence.

## Typed composability

The typed runtime exists so retrieval and analytical outputs can become reliable inputs to later operations.

Representative semantic types include the current:

- `QUERY_TEXT`;
- `ENTITY_SET`;
- `RELATIONSHIP_SET`;
- `CHUNK_SET`;
- `SUBGRAPH`;
- `COMMUNITY_SET`;
- `SCORE_VECTOR`.

The long-term algebra may also need reusable concepts such as `TABLE`, `VECTOR_SET`, `MODEL` or `FINDING_SET` when concrete capabilities require them.

Examples:

```text
ENTITY_SET → k-hop → SUBGRAPH → Leiden → COMMUNITY_SET
SUBGRAPH → centrality → SCORE_VECTOR → top-k → ENTITY_SET → evidence
TABLE → aggregate → TABLE
QUERY_TEXT → vector search → ENTITY_SET
```

New types should encode reusable data semantics, not a programmed thought process.

## Wiki as progressive-disclosure knowledge and environment map

The wiki concept is not merely "documents plus links" and does not imply special `wiki.open` / `wiki.follow` APIs.

DIGIMON should be able to materialize an **agent-readable progressive-disclosure surface** that answers questions such as:

- What semantic content exists?
- How is it organized by entities, concepts, topics, assertions and sources?
- What derived representations exist?
- What ontology/schema does each representation use?
- Which canonical IDs connect those representations?
- What specialized retrieval/analytic capabilities apply to each representation?
- Where is deeper evidence or source material located?

A concept/entity page can therefore contain both semantic content and operational catalog facts. For example, an entity page can identify the canonical ID and state that it appears in a relational table, a property graph and a vector collection, along with the schemas and IDs needed to move between them.

The wiki is descriptive, not a workflow policy engine. It should say what exists and what operations are available; the harness decides whether to query SQL, traverse two graph hops, use vector similarity, inspect source evidence, or combine them.

## Do not rebuild native harness capabilities

DIGIMON should expose a specialized tool only when DIGIMON provides capability or state the harness does not already possess.

If the harness already has good native abilities to:

- open/read files;
- search/grep text;
- follow Markdown links;
- navigate directories/wiki pages;

DIGIMON should generally materialize good artifacts and let the harness use those native capabilities rather than wrapping them for symmetry.

Specialized operations belong in DIGIMON when they require DIGIMON-owned state or engines, such as graph algorithms, vector indexes, community resources, specialized lexical indexes, structured queries over generated databases, or analytical transformations.

## Harness boundary

The external harness owns adaptive reasoning policy:

- interpreting the goal;
- selecting representations and tools;
- sequencing, branching, parallelizing and retrying;
- comparing results across representations;
- revising strategy after observations;
- deciding when evidence is sufficient and when to stop.

DIGIMON owns:

- representations;
- specialized retrieval operations;
- analytic/transformation operations;
- typed contracts and compatibility facts;
- resource and representation metadata;
- evidence/provenance/derivation facts;
- bounded model-assisted transformations local to an individual capability.

AoT, GoT, ReAct and decomposition remain optional heuristics for the harness, not a mandatory internal cognitive runtime.

## Public access surfaces

The intended access surfaces are different interfaces over the same maintained core:

- **CLI** — human-facing shell;
- **Python runtime** — application/developer-facing library surface;
- **MCP/tool protocol** — agent-facing specialized capability surface.

MCP is not the product. The legacy CLI currently uses older planner/orchestrator code and should eventually converge on the same maintained capability/runtime core rather than defining a second architecture.

## Evidence, derived state and findings

DIGIMON should distinguish:

1. **source-backed semantic state** — governed assertions and exact evidence;
2. **retrieval artifacts** — bounded working sets such as entity sets, tables and subgraphs;
3. **derived analytic artifacts** — communities, score vectors, centralities, model outputs, forecasts, bridge-account classifications, etc.;
4. **findings / interpretations** — claims or judgments made from evidence plus analytical results.

A derived value should retain enough information to know what method produced it, with what parameters, from which inputs and representation version, and with what evidence/uncertainty where applicable.

## Provenance / derivation graph

DIGIMON's provenance concept is broader than final-answer citations.

Three related forms must remain distinct:

- **evidence provenance** — what source text supports an assertion or answer;
- **semantic provenance** — how governed semantic assertions relate to source candidates/evidence;
- **artifact/derivation lineage** — what source or prior artifacts and transformation executions produced each projection, retrieval artifact, analytical artifact and finding.

Target lineage:

```text
source resource
  ↓ acquisition/capture
immutable source artifact/version
  ↓ semantic extraction/governance
Foundation/governed semantic IR
  ↓ projection execution
SQL | property graph | vectors | RDF | tree | wiki | lexical index
  ↓ retrieval execution
bounded working set / subgraph / table / entity set
  ↓ analytic transformation
community | score vector | model/result | derived artifact
  ↓ interpretation
finding
```

Every important derived artifact should be able to identify its exact inputs, transformation/method identity, parameters/configuration, output identity/version, and source lineage. Recursive lineage should end at immutable/reopenable source evidence, explicit human inputs, or declared root observations.

The provenance/derivation graph is **not the domain graph**. A domain graph represents people, organizations, assertions, events and relationships. The derivation graph represents artifacts, projections, executions, methods, parameters and findings.

This lineage supports reproducibility, stale-artifact detection, invalidation, auditability and source recovery.

## Evidence → Action

The broader ambition is not just retrieval. It is to support reusable analytical work from evidence to action:

```text
evidence
  ↓
representation
  ↓
retrieval
  ↓
analysis / transformation
  ↓
review / interpretation
  ↓
finding / decision support
```

The durable residue of analysis should include:

- evidence and provenance;
- structured knowledge;
- methods and parameters;
- findings and uncertainties;
- open questions.

The operating principle is:

> **Every analysis should make the next one easier.**

That implies reusable representations, reusable retrieval outputs, reusable analytic outputs, preserved lineage, composable methods and derived results that can safely become inputs to later work.

## Current engineering focus versus the north star

Recent work has concentrated on making the existing retrieval/runtime core truthful: graph builds, VDB semantics, PPR, reference methods, grounding, resource selection, composition validation, invalidation and evidence propagation.

That work is necessary but narrower than the project vision. Documentation and implementation planning must not mistake the current workstream for the full project identity.

The north star remains:

**Governed semantic IR → Represent → Retrieve → Analyze/Transform → Grounded evidence/findings → Action, with shared identity and derivation lineage across the entire path.**
