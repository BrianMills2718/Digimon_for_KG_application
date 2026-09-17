# DIGIMON Planning Summary

**Updated:** 2026-09-17  
**Status:** current planning summary

## Current direction

DIGIMON's north star is broader than the recent harness-first retrieval refactor:

> **Governed semantic IR → Represent → Retrieve → Analyze/Transform → grounded evidence/findings → action, with shared canonical identity and derivation lineage across the whole path.**

The canonical upstream semantic authority is onto-canon6. DIGIMON should consume governed Foundation-style IR, project it into complementary retrieval/analytic representations, expose specialized retrieval and analytical methods over those representations, and preserve evidence/derivation lineage.

The external harness still owns adaptive reasoning policy: selecting representations/tools, sequencing, branching, retrying and stopping. Harness-first is therefore an architectural boundary underneath the product thesis, not the project identity.

## Three capability planes

### Represent

Target representation families include:

- relational/tabular;
- vector;
- property graph;
- semantic/RDF graph where useful;
- specialized lexical/full-text indexes where native harness search is insufficient;
- hierarchy/tree;
- source/evidence representations;
- a progressive-disclosure agent wiki/catalog describing semantic content, representation schemas, canonical IDs and available specialized capabilities.

Raw-document ingestion/chunking remains valid standalone/compatibility behavior, but governed semantic IR is the canonical ecosystem seam.

### Retrieve

The agent should be able to obtain bounded working sets using each representation's native strengths: SQL, vector search, graph traversal/PPR/community/subgraph methods, hierarchy expansion, specialized lexical retrieval and exact evidence recovery.

### Analyze / transform

DIGIMON is also an analytical suite. Its SNA/graph lineage matters: a common workflow is retrieve a graph/subgraph, run an analytic transformation such as Leiden or centrality, then reuse the derived result in subsequent retrieval/analysis.

Graph analytics already exist substantially but need to be organized as a coherent first-class typed capability plane. Non-graph analytics should be added from concrete reusable needs rather than as an indiscriminate toolbox.

## Cross-representation identity

Canonical identity is the glue for composition. IDs such as `entity_id`, `assertion_id`, `predicate_id`, `source_ref` and evidence identity should survive derived projections wherever possible.

This should let a harness:

```text
find an entity in the wiki/catalog
→ use its canonical ID in SQL
→ traverse the corresponding graph node
→ query vector metadata
→ recover exact source evidence
```

DIGIMON makes those moves possible; the harness decides whether and when to make them.

## Wiki/catalog boundary

The wiki is not primarily a new API family. It is an agent-readable progressive-disclosure artifact and environment map.

It should describe:

- semantic organization of the knowledge;
- available representations;
- schemas/ontologies;
- canonical IDs joining them;
- specialized retrieval/analytic capabilities;
- deeper evidence/source paths.

Do not build `wiki.open`, `wiki.follow` or basic text-search wrappers when the external harness already performs those tasks well.

## Provenance / derivation

Keep three concepts distinct:

1. evidence provenance;
2. semantic provenance;
3. artifact/derivation lineage.

The target derivation chain is:

```text
source artifact
→ governed semantic IR
→ representation projection
→ retrieval artifact / bounded working set
→ analytic transformation + parameters
→ derived artifact
→ finding
```

The derivation graph is different from the domain/property graph. It exists for reproducibility, invalidation, auditability and source recovery.

## Current implementation reality

The current maintained implementation is strongest in graph/vector retrieval and evidence grounding. Recent work materially repaired:

- strict composition wiring/execution;
- loop/conditional control flow;
- raw standalone chunking;
- graph build truthfulness and source manifests;
- VDB score/identity semantics;
- PPR modes;
- reference-method wiring and evidence accumulation;
- structural/community materialization;
- grounding/citation validation;
- active graph/canonical VDB selection;
- pragmatic VDB/community/matrix invalidation;
- multilingual graph identity/semantic text handling.

These fixes are necessary foundations but do not constitute the whole north star.

## Largest current gaps

- no fresh current-head deterministic runtime certification;
- canonical onto-canon/Foundation IR → DIGIMON projection path needs to become the dominant tested seam;
- cross-representation identity needs explicit projection invariants;
- no canonical relational database projection yet;
- no generated progressive-disclosure agent wiki/catalog yet;
- graph analytics are not yet organized as a comprehensive typed analytic plane;
- no first-class artifact/execution derivation graph across projections, retrievals and analytics;
- sparse matrices still have a same-shaped cross-graph identity edge case;
- Python/CLI/MCP have not fully converged on one maintained runtime;
- custom ontology selection/load on the maintained build path still needs closure.

## Active plan

The authoritative sequence is in [ROADMAP.md](ROADMAP.md):

1. get a real current-head core/canary execution signal;
2. finish first concrete runtime reds and custom-ontology wiring;
3. verify the current onto-canon/Foundation IR handoff;
4. establish cross-representation canonical identity;
5. implement the first canonical relational/tabular projection;
6. generate the first progressive-disclosure wiki/catalog over the same fixture;
7. inventory/promote existing graph analytics into typed analytic capabilities;
8. add minimal derivation records across projection → retrieval → analysis;
9. bind remaining graph-derived resources to exact graph identity;
10. converge Python/CLI/MCP on one maintained core;
11. broaden deterministic architecture tests;
12. make benchmarking/research validation primary only after those seams are real.

## What is deliberately not the plan

Do not respond to this vision by building:

- another internal general-purpose agent brain;
- another orchestrator generation;
- a mandatory AoT/GoT/ReAct executor;
- wrapper tools around native harness file/search/wiki navigation merely for symmetry;
- a generalized enterprise resource catalog before concrete projections require it;
- geospatial support in the current text-derived scope;
- every possible database/analytics engine at once;
- benchmark-specific core architecture.

## Current source-of-truth set

- `docs/VISION.md`
- `docs/CURRENT_STATE.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/ARCHITECTURE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`
- `docs/DOCUMENTATION_COVERAGE.md`

`docs/adr/002-harness-first-capability-architecture.md` remains an accepted decision about **orchestration ownership**, not the complete DIGIMON north star.
