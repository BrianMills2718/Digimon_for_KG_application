# DIGIMON Roadmap

**Updated:** 2026-09-17  
**Scope:** move from the current graph/vector-heavy retrieval core toward the full [VISION.md](VISION.md): **Represent → Retrieve → Analyze**, downstream of governed semantic IR, with shared identity and derivation lineage across the workflow.

This roadmap is ordered by dependency, not calendar estimates.

## Guiding principle

> **Make the current core demonstrably correct first. Then broaden DIGIMON from a strong graph/vector retrieval system into the intended general text-derived representation, retrieval, and analytics runtime—without rebuilding the external harness's reasoning abilities.**

## Stage 0 — Keep the documentation hierarchy truthful

**Goal:** prevent the current workstream from being mistaken for the project north star.

### Canonical roles

- `VISION.md` — durable product/research north star;
- `CURRENT_STATE.md` — current code reality;
- `IMPLEMENTATION_MAP.md` — module-level implementation map;
- `ARCHITECTURE.md` — target technical/system boundaries;
- `GAP_ANALYSIS.md` — current→target gaps;
- `ROADMAP.md` — ordered work;
- `DOCUMENTATION_COVERAGE.md` — checklist preventing myopic documentation updates.

### Exit criteria

- README and docs index lead with the full Represent/Retrieve/Analyze thesis;
- harness-first is documented as a control boundary, not the project identity;
- onto-canon6 is clearly upstream semantic authority;
- provenance/derivation is important but not allowed to displace representation, retrieval, analytics, wiki/catalog, identity or surface concerns;
- current-state docs do not claim fresh runtime certification without a real run.

**Current status:** substantially complete; maintain continuously.

## Stage 1 — Obtain a real current-head runtime signal

**Goal:** stop relying on source inspection as the primary confidence mechanism.

### Work

Run the maintained deterministic path on a real runner:

```bash
pip install -r requirements-minimal.txt
pytest tests/core -q
python tests/e2e/test_mcp_smoke.py
DIGIMON_CANARY_REBUILD=1 python tests/e2e/test_mcp_smoke.py
```

Then use the existing failure-driven rule:

```text
first concrete failure
→ smallest fix
→ regression test
→ rerun
```

Do not use the lack of GitHub Actions runs as evidence that current code is green or broken.

### Exit criteria

- deterministic core suite runs on current head;
- reuse canary runs;
- clean-rebuild canary runs;
- first failures have been fixed rather than worked around in documentation;
- runtime status is recorded explicitly.

## Stage 2 — Finish the canonical governed-IR input seam

**Goal:** make onto-canon/Foundation-style governed IR the canonical ecosystem input to DIGIMON projections.

### Work

- inspect the actual current Foundation IR/export contract from onto-canon6;
- map stable IDs, predicates, n-ary roles, entity types, aliases, literals, qualifiers, source refs and evidence spans into DIGIMON projection inputs;
- complete and test custom-ontology selection/load behavior where graph extraction still uses standalone/raw mode;
- distinguish **canonical governed-IR projection** from **standalone raw-document ingestion** in code/docs;
- preserve raw mode for experiments, benchmarks and independent use without treating it as the ecosystem authority path;
- add one small governed fixture that can drive multiple DIGIMON projections.

### Exit criteria

- one governed IR fixture imports deterministically;
- entity/assertion/source/evidence identities survive the handoff;
- no projection silently invents semantic meaning absent from the IR;
- raw-document mode is still usable but clearly secondary in the ecosystem architecture.

## Stage 3 — Establish cross-representation identity as an invariant

**Goal:** make heterogeneous representations cheaply composable by the harness.

### Work

Define and test canonical identity mappings for at least:

- `entity_id`;
- `assertion_id`;
- `predicate_id`;
- `source_ref`;
- evidence/span identity.

Ensure projections use those identities wherever their native engines permit it.

Example invariant:

```text
entity:alice-smith
    ├─ relational entities.entity_id
    ├─ property-graph node ID
    ├─ vector metadata.entity_id
    ├─ semantic-graph identifier
    └─ wiki/catalog metadata
```

Projection-local IDs may exist, but they must not become the only bridge between representations.

### Exit criteria

- a harness can move from one representation to another using explicit canonical IDs rather than fuzzy rediscovery;
- projection tests verify identity preservation;
- evidence/source IDs remain reopenable after cross-representation moves.

## Stage 4 — Canonical relational/tabular projection

**Goal:** add the first major non-graph canonical representation family.

### Work

Define a compact relational projection of governed IR, likely including:

- entities;
- aliases / identity memberships;
- assertions;
- assertion roles;
- literal values;
- source/evidence references;
- qualifiers/provenance.

Prefer an engine such as DuckDB/SQLite where it keeps local operation simple. The point is not the database brand; the point is native relational capability:

- exact filtering;
- joins;
- aggregation;
- grouping;
- analytical SQL;
- window/recursive operations where useful.

Do not embed user-level reasoning policy in SQL helpers.

### Exit criteria

- governed fixture projects to a relational artifact;
- schema is documented in machine- and agent-readable form;
- canonical IDs match other projections;
- a few representative exact/aggregate SQL tasks work;
- relational results can feed typed downstream retrieval/analytic operations.

## Stage 5 — Materialize the agent wiki / progressive-disclosure catalog

**Goal:** give the harness a navigable map of both knowledge and the retrieval/analytic environment.

### Work

Generate a deterministic artifact surface such as:

```text
index.md
knowledge/
  people/
  organizations/
  concepts/
  topics/
  sources/
representations/
  property-graph.md
  relational.md
  vectors.md
  ...
schemas/
  ontology.md
  graph-schema.md
  relational-schema.md
  vector-collections.md
```

Pages should expose:

- semantic summaries and organization;
- canonical IDs;
- source/evidence links;
- available representations for an entity/assertion/concept;
- representation schemas;
- specialized capabilities applicable to those representations.

Do **not** build `wiki.open`, `wiki.follow`, basic grep/search wrappers merely for symmetry when the harness already has those native capabilities.

### Exit criteria

- the governed fixture generates a useful progressive-disclosure knowledge/catalog artifact;
- a harness using ordinary file/link/search abilities can discover an entity and learn how to address it in SQL/graph/vector representations;
- the wiki describes available operations without prescribing a workflow.

## Stage 6 — Make analytics a first-class capability plane

**Goal:** expose DIGIMON's analytical lineage explicitly instead of treating analytics as scattered retrieval helpers.

### Work

First inventory existing analysis/transformation code before adding anything new.

Prioritize graph/SNA capabilities already close to the codebase and the project's lineage, such as:

- degree/weighted degree;
- PageRank/eigenvector-style scores where supported;
- betweenness and related centrality measures;
- Leiden/community detection;
- connected components;
- k-core/cohesion/density/assortativity where useful;
- brokerage/bridging measures;
- shortest paths and path statistics;
- diffusion/propagation;
- PCST/Steiner/subgraph transformations.

Then add non-graph methods only where they have clear reusable value:

- relational aggregation/statistics;
- distributions/group comparisons;
- clustering/dimensionality reduction;
- anomaly detection;
- temporal aggregation/trends;
- model fitting where a concrete use case requires it.

### Type model

Extend the current typed algebra only when concrete capabilities require it. Candidate reusable types include:

- `TABLE`;
- `VECTOR_SET`;
- richer score/metric records;
- `MODEL`;
- `FINDING_SET` / derived artifact records.

Do not encode a thought process as types.

### Exit criteria

- analytic capabilities are discoverable alongside retrieval capabilities;
- retrieved working sets can feed analytics directly;
- analytic outputs can feed later retrieval/analytics;
- graph analytics are no longer hidden as incidental implementation details;
- source evidence and derived state are clearly distinguished.

## Stage 7 — Build first-class artifact / derivation lineage

**Goal:** extend provenance from source citations into the entire analytical chain.

### Distinguish three things

- evidence provenance;
- semantic provenance;
- artifact/derivation lineage.

### Work

Define the smallest useful derivation contract recording:

- input artifact/resource IDs + versions/hashes;
- transformation/capability identity;
- parameters/configuration/provider where material;
- output artifact ID + version/hash;
- projection/analytic kind;
- scope/lossiness where applicable;
- links back to canonical semantic/source identity.

Target lineage:

```text
source artifact
→ governed semantic IR
→ representation projection
→ retrieval artifact / bounded working set
→ analytic transformation
→ derived artifact
→ finding
```

The derivation graph must remain distinct from the domain/property graph.

### First practical uses

- reproduce a centrality/community result;
- explain which subgraph produced it;
- trace it to the graph/IR/source versions;
- mark dependent artifacts stale when their inputs change.

### Exit criteria

- at least projection, retrieval and analytic outputs have explicit derivation records;
- recursive lineage reaches source evidence or declared root/human inputs;
- stale-artifact detection uses recorded dependency identity rather than only naming/path conventions;
- findings can cite both evidence and analytic derivation.

## Stage 8 — Close remaining resource correctness seams

**Goal:** keep resource logic concrete and trustworthy without inventing enterprise infrastructure.

### Work

- bind sparse matrices to exact graph identity/version to eliminate same-shaped cross-graph ambiguity;
- preserve current source-chunk manifest behavior;
- retain practical VDB/community invalidation rules;
- extend identity/invalidation only as new relational/wiki/vector/analytic artifacts are added;
- introduce more general resource descriptors only where multiple real producers/consumers need them.

### Exit criteria

- no maintained projection/analytic artifact can be silently paired with a different source graph/IR version;
- resource rebuild/reuse behavior is deterministic and explainable;
- derivation records provide enough dependency information for invalidation.

## Stage 9 — Converge public surfaces on one maintained core

**Goal:** CLI, Python and MCP become different interfaces over the same runtime.

### Python runtime

Create a narrow supported application/developer API for:

- loading/creating projections;
- inspecting representations/resources;
- executing typed retrieval/analytic capabilities;
- obtaining evidence/derivation results.

### CLI

Replace or adapt the current `PlanningAgent`/`AgentOrchestrator` dependence so the CLI calls the same maintained runtime. The CLI remains human-facing; it does not need to become another internal agent brain.

### MCP

Retain MCP as the agent-facing protocol surface for specialized DIGIMON capabilities. Do not make MCP itself the product identity.

### Exit criteria

- equivalent supported operations behave consistently across Python/CLI/MCP;
- user-level planning is not duplicated inside the CLI/runtime by default;
- legacy planner/orchestrator paths are clearly compatibility-only or retired.

## Stage 10 — Standardize errors and retire architectural ambiguity

**Goal:** make the stabilized public surfaces easy for humans, applications and harnesses to recover from.

### Work

Use a small actionable error/result convention distinguishing at least:

- missing resource/prerequisite;
- incompatible/stale resource;
- invalid wiring/type;
- empty evidence/result;
- provider/model failure;
- unsupported/lossy transformation;
- internal failure.

Then finish live-caller classification for old AgentBrain/AOT/orchestrator/MCP generations and remove or isolate dead code.

### Exit criteria

- maintained callers do not need tool-specific prose parsing for ordinary recovery;
- old cognitive/runtime generations cannot be mistaken for the preferred architecture;
- repository navigation reflects the current system.

## Stage 11 — Reliability and CI as an architectural gate

**Goal:** make the full thesis testable, not merely documented.

### Test matrix

Cover at least:

- governed IR → representation projections;
- cross-representation identity;
- graph/vector/relational retrieval contracts;
- wiki/catalog generation;
- typed retrieval→analytic chains;
- evidence vs derived-state semantics;
- derivation lineage;
- invalidation on upstream changes;
- public Python/CLI/MCP parity for supported operations;
- clean rebuild + reuse canaries.

Keep live-provider/expensive suites separately classified from deterministic core tests.

### Exit criteria

- architectural regressions fail CI;
- current-state docs can cite fresh deterministic certification;
- representation/retrieval/analytic composition is covered by at least one small canonical governed fixture.

## Stage 12 — Later evaluation and research validation

Only after the earlier stages are substantially coherent should evaluation become a primary workstream.

Questions then include:

- when relational vs vector vs graph vs wiki navigation is most useful;
- when graph structure adds evidence value over simpler methods;
- fixed retrieval methods vs adaptive harness composition;
- retrieval→analytics workflows versus retrieval-only workflows;
- quality/cost effects of different projections;
- robustness to incomplete/noisy semantic extraction;
- value of derivation lineage for debugging/reproducibility;
- when specialized lexical indexing adds value beyond native harness search;
- how well agents use the progressive-disclosure catalog to move across representations.

See [FUTURE_EVALUATION_QUESTIONS.md](FUTURE_EVALUATION_QUESTIONS.md).

## What not to build merely for symmetry

Avoid adding:

- another internal general-purpose planner;
- another orchestrator generation;
- mandatory AoT/GoT/ReAct state machines;
- `wiki.open` / `wiki.follow` / basic text-search wrappers when the harness already has those abilities;
- a generalized enterprise resource catalog before real projection/analytic artifacts demand it;
- geospatial representation in the current text-derived scope;
- every conceivable database/analytic engine before a concrete capability needs it;
- benchmark-specific core architecture.

## Immediate implementation sequence

If code work resumes directly from this roadmap:

1. get a real current-head test/canary run;
2. finish custom ontology wiring and any first runtime reds;
3. verify current onto-canon/Foundation IR and its DIGIMON handoff;
4. define/test cross-representation canonical identity;
5. implement the first canonical relational projection;
6. generate the first progressive-disclosure wiki/catalog over the same fixture;
7. inventory and promote existing graph analytics into a coherent typed analytic catalog;
8. add minimal derivation records across projection → retrieval → analytic outputs;
9. bind remaining graph-derived resources such as sparse matrices to exact graph identity;
10. converge Python/CLI/MCP on the same core;
11. expand deterministic architecture tests;
12. evaluate only after these seams are real.
