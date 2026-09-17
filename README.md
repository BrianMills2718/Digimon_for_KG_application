# DIGIMON

DIGIMON is a **general text-derived representation, retrieval, and analytics runtime**.

Its ecosystem role is to consume governed semantic IR—primarily from `onto-canon6`—and project that semantic core into complementary retrieval/analysis representations such as relational tables, property graphs, vector indexes, hierarchies, communities and agent-navigable knowledge surfaces. It then exposes specialized retrieval and analytic operations over those representations so a capable external agent or application can compose its own evidence-to-action workflow.

> **Governed semantic IR → Represent → Retrieve → Analyze/Transform → Grounded evidence/findings → Action**

The recent harness-first refactor is an architectural strategy underneath this thesis. It is not the whole project identity.

## The north star

DIGIMON is intended to be a one-stop environment for two closely related jobs:

1. **Structure governed text-derived knowledge into useful analytical/retrieval representations.**
2. **Expose the native retrieval and analytical strengths of those representations as composable capabilities.**

The system should preserve canonical identity and provenance across representations so an agent can move naturally between them. For example, an agent may discover `entity:alice-smith` through a progressive-disclosure wiki/catalog, query her exact attributes in a relational database, traverse two graph hops around the same canonical entity, run Leiden clustering or centrality over the retrieved subgraph, and then recover the exact source evidence behind the relevant relationships.

DIGIMON does **not** need to prescribe that sequence. The external harness decides the strategy.

See **[docs/VISION.md](docs/VISION.md)** for the complete project thesis and **[docs/DOCUMENTATION_COVERAGE.md](docs/DOCUMENTATION_COVERAGE.md)** for the checklist that prevents the canonical docs from collapsing onto only the latest implementation topic.

## Ecosystem boundary

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
external harness / application / human workflow
```

`onto-canon6` owns semantic governance and canonical semantic truth. DIGIMON owns downstream retrieval/analysis projections and their specialized methods.

Raw-document ingestion/chunking remains useful for standalone operation, experiments, benchmarks and compatibility, but it is not the conceptual center of the ecosystem architecture.

## Three capability planes

### 1. Represent

The same semantic core can be projected into different structures because different structures make different questions easy.

Current and target families include:

- **Relational/tabular** — exact structured filtering, joins, grouping and aggregation;
- **Vector** — semantic similarity and nearest-neighbor retrieval;
- **Property graph** — paths, neighborhoods, diffusion, communities and structural algorithms;
- **Semantic graph / RDF** — ontology-aware typed semantic querying where useful;
- **Full-text / lexical indexes** — BM25, exact terms, phrases and fielded retrieval when this adds capability beyond native harness file search;
- **Hierarchy/tree** — abstraction levels, recursive summaries and taxonomy navigation;
- **Wiki / progressive-disclosure knowledge surface** — an agent-readable map of semantic content, representations, schemas, IDs, capabilities and sources;
- **Source/evidence representation** — exact reopenable evidence and source material.

Geospatial representation is outside the current text-focused vision.

### 2. Retrieve

Retrieval obtains a bounded working set from a representation. Examples include:

- SQL queries and exact lookups;
- vector similarity search;
- lexical/BM25 search;
- graph k-hop/path/subgraph retrieval;
- PPR/diffusion;
- community retrieval;
- hierarchy expansion;
- source/evidence recovery.

The design principle is conditional rather than graph-first: **use graph structure when it adds evidence value; use a simpler representation when it does not.**

### 3. Analyze / transform

DIGIMON is also an analytic-method suite, reflecting its social-network-analysis lineage.

A canonical pattern is:

```text
retrieve a working set
        ↓
apply an analytic method
        ↓
produce a typed derived artifact
        ↓
use that result for further retrieval or analysis
```

Graph-oriented examples include centrality, Leiden/community detection, connected components, k-core/cohesion, shortest paths, brokerage, diffusion, PCST and Steiner-style subgraph transformations.

Non-graph examples can include SQL aggregation, descriptive statistics, clustering, dimensionality reduction, regression/classification, anomaly detection and temporal aggregation when supported by the represented data.

Derived analytical values are not original evidence; they should retain method, parameters, inputs, representation/version and lineage.

## Cross-representation identity

Composability depends on preserving canonical identity across projections.

Where possible, the same identifiers should survive into every derived form:

- `entity_id`;
- `assertion_id`;
- `predicate_id`;
- `source_ref`;
- evidence/span identity.

A single canonical entity should therefore be addressable through SQL rows, graph nodes, vector metadata, RDF identifiers and wiki/catalog metadata without fuzzy rediscovery.

Projection-specific state—vector IDs, community assignments, PageRank scores, wiki paths, implementation row IDs—belongs downstream in DIGIMON rather than in semantic authority unless it has genuine semantic meaning.

## Wiki / progressive-disclosure navigation

The wiki concept is not merely another document store and does not require DIGIMON to implement trivial `wiki.open` or `wiki.follow` wrappers.

Its role is to materialize an **agent-readable semantic and operational map** that can answer:

- what knowledge exists;
- how it is organized;
- which representations exist;
- what schemas/ontologies those representations use;
- which canonical IDs connect them;
- what specialized retrieval/analytic capabilities apply;
- where deeper evidence and source material live.

A capable harness can then use its own native file/search/link-navigation abilities to traverse that representation.

DIGIMON should expose a specialized tool only when DIGIMON provides capability or state the harness does not already possess.

## Harness boundary

The external harness owns adaptive control policy:

- interpreting the goal;
- choosing representations and tools;
- sequencing, branching, retrying and parallelizing;
- comparing observations;
- revising strategy;
- deciding when to stop.

DIGIMON owns:

- representations;
- specialized retrieval operations;
- analytic/transformation operations;
- typed contracts and compatibility facts;
- resource/representation metadata;
- evidence and derivation lineage;
- bounded model-assisted transformations local to individual capabilities.

> **Program the capabilities, contracts, resources, analytics and evidence boundaries. Prompt useful reasoning heuristics. Let the harness remain intelligent.**

AoT/GoT/ReAct/decomposition remain optional heuristics, not a mandatory internal cognitive runtime.

## Public access surfaces

The target is one maintained core with several interfaces:

- **CLI** — human-facing shell;
- **Python runtime** — application/developer-facing library surface;
- **MCP/tool protocol** — agent-facing specialized capability surface.

MCP is not the product. The current `digimon_cli.py` still uses the older `PlanningAgent` / `AgentOrchestrator` path and is transitional until it converges on the maintained runtime.

## Typed composition

The current operator core uses typed query/entity/relationship/chunk/subgraph/community/score-vector values. The purpose of those types is to make outputs reusable as later inputs—not to encode thought states.

Examples:

```text
ENTITY_SET → k-hop → SUBGRAPH → Leiden → COMMUNITY_SET
SUBGRAPH → centrality → SCORE_VECTOR → top-k → ENTITY_SET → evidence
QUERY_TEXT → vector search → ENTITY_SET
```

Future reusable types such as `TABLE`, `VECTOR_SET`, `MODEL` or `FINDING_SET` should be introduced only when concrete capabilities require them.

The ten named retrieval methods remain useful reference compositions, regressions and shortcuts. They are not DIGIMON's identity.

## Provenance / derivation graph

DIGIMON's provenance model is broader than answer citations.

It should distinguish:

1. **evidence provenance** — which source text supports an assertion or answer;
2. **semantic provenance** — how governed assertions relate to candidates/evidence;
3. **artifact/derivation lineage** — which prior artifacts and transformation executions produced projections, retrieval artifacts, analytical outputs and findings.

Target lineage:

```text
source resource
  ↓ acquisition/capture
immutable source artifact/version
  ↓ semantic extraction/governance
governed semantic IR
  ↓ projection execution
SQL | graph | vector | RDF | tree | wiki | lexical index
  ↓ retrieval execution
bounded working set / subgraph / table / entity set
  ↓ analytic transformation
community | score vector | model/result | derived artifact
  ↓ interpretation
finding
```

The derivation/provenance graph is not the domain/property graph. The domain graph represents the subject matter; the derivation graph represents artifacts, projections, executions, methods, parameters and findings.

This lineage supports reproducibility, source recovery, stale-artifact detection, invalidation and auditability.

## Evidence → Action

The broader ambition is reusable analytical knowledge work:

```text
evidence → representation → retrieval → analysis/transformation → review/interpretation → finding/action
```

The durable residue of analysis should include:

- evidence and provenance;
- structured knowledge;
- methods and parameters;
- findings and uncertainties;
- open questions.

> **Every analysis should make the next one easier.**

## Current status

As of **2026-09-17**, the repository remains hybrid/transitional, but a substantial retrieval/runtime core exists and has been heavily corrected at source level.

Implemented or substantially present areas include:

- typed operator records/dataflow and composition;
- multiple graph types and vector indexes;
- entity, relationship, chunk, subgraph and community operations;
- ten named reference retrieval/reasoning methods;
- graph/table/vector conversion code;
- MCP build/retrieval/resource surfaces;
- direct and typed analytical graph operations;
- evidence-grounded answer generation and provenance metadata;
- resource invalidation and graph/chunk manifest logic;
- deterministic contract tests covering many repaired semantics.

Important caveat: the current head has **not received a fresh end-to-end runtime certification** through the available environment. Historical GitHub Actions failed before tests on dependency installation, and current connector-created commits have not triggered new runs. Source-level regression contracts therefore should not be described as a green current-head test suite until executed by a real runner.

For code reality and verification boundaries, see **[docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)**.

## Canonical documentation

1. **[docs/VISION.md](docs/VISION.md)** — durable product/research north star.
2. **[docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)** — what is materially implemented/verified now.
3. **[docs/IMPLEMENTATION_MAP.md](docs/IMPLEMENTATION_MAP.md)** — module-level classification and implementation caveats.
4. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — target system design realizing the vision.
5. **[docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md)** — current → target gaps.
6. **[docs/ROADMAP.md](docs/ROADMAP.md)** — ordered closure plan.
7. **[docs/DOCUMENTATION_COVERAGE.md](docs/DOCUMENTATION_COVERAGE.md)** — reconciliation checklist preventing documentation myopia.
8. **[docs/README.md](docs/README.md)** — canonical documentation index and maintenance rules.

## Repository map

```text
Core/                       operators, graph/index/provider implementations, composition and legacy layers
Config/                     configuration models and ontology material
Option/                     runtime/method configuration
prompts/                    reasoning/routing/synthesis heuristics
Data/                       example/evaluation datasets
eval/                       evaluation infrastructure
tests/ + test_*.py          deterministic/integration/E2E/experimental tests
docs/                       canonical architecture plus supporting/historical material
examples/                   example workflows
api.py                      secondary HTTP/API surface
digimon_cli.py              human-facing CLI, currently transitional internally
digimon_mcp_stdio_server.py agent-facing MCP surface
```

The repository intentionally retains historical/experimental code. **File existence does not imply canonical architecture.** See `docs/IMPLEMENTATION_MAP.md`.

## Development principle

Prefer concrete failures over speculative frameworks:

> **Working but messy → reproducibly working → broaden coverage → clean abstractions where actual pain appears.**

Do not build a new agent brain, resource-governance platform, or wrapper around capabilities the harness already possesses merely for architectural symmetry.

## Lineage

DIGIMON began from the unified GraphRAG framework lineage including [JayLZhou/GraphRAG](https://github.com/JayLZhou/GraphRAG) and *In-depth Analysis of Graph-based RAG in a Unified Framework* (Zhou et al., arXiv:2503.04338, 2025), and has since expanded toward a general composable representation/retrieval/analytics environment.
