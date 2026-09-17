# DIGIMON Documentation Coverage Checklist

**Purpose:** prevent canonical documentation from collapsing onto the most recent implementation topic.  
**Updated:** 2026-09-17

Every canonical documentation reconciliation should preserve visibility of the full project thesis below. A document does not need to repeat every detail, but the canonical set as a whole must cover each item explicitly and consistently.

## 1. Product / research north star

- [ ] DIGIMON is described as a **general text-derived representation, retrieval, and analytics runtime**.
- [ ] The project is not reduced to GraphRAG, MCP, an agent harness substrate, or the current refactor.
- [ ] The core flow is visible: **governed semantic IR → represent → retrieve → analyze/transform → grounded evidence/findings → action**.
- [ ] Conditional/mixed-method retrieval is explicit: graph methods are used when graph structure adds value, not by default.

## 2. onto-canon6 boundary

- [ ] onto-canon6 owns semantic extraction/binding, ontology/profile semantics, governance/review, canonical semantic identity and governed semantic export.
- [ ] Foundation-style governed semantic IR is the canonical ecosystem input to DIGIMON.
- [ ] DIGIMON does not duplicate semantic authority/governance.
- [ ] Raw-document ingestion/chunking is described as standalone/compatibility/experiment capability rather than the conceptual ecosystem center.

## 3. Representation plane

- [ ] Relational/tabular projection is represented.
- [ ] Vector projection is represented.
- [ ] Property-graph projection is represented.
- [ ] Semantic graph/RDF is represented as a potential/target semantic projection where useful.
- [ ] Full-text/lexical indexing is represented where it adds capability beyond native harness search.
- [ ] Hierarchy/tree projection is represented.
- [ ] Wiki/progressive-disclosure representation is represented.
- [ ] Source/evidence representation is represented.
- [ ] Geospatial is not accidentally introduced as a current core requirement.

## 4. Cross-representation identity

- [ ] Canonical IDs are described as shared join keys across projections wherever possible.
- [ ] `entity_id`, `assertion_id`, `predicate_id`, source/evidence identity remain distinguishable.
- [ ] Projection-specific state such as vector IDs, wiki paths, community IDs and analytical scores remains downstream rather than polluting semantic authority.

## 5. Wiki / agent navigation

- [ ] Wiki is described as a **progressive-disclosure semantic and operational map**, not merely pages plus hyperlinks.
- [ ] Wiki can describe semantic organization, representations, schemas/ontologies, canonical IDs, capabilities and source/evidence locations.
- [ ] Wiki/catalog surfaces are descriptive rather than prescriptive workflow policy.
- [ ] Documentation does not imply DIGIMON must wrap native harness file open/search/link-follow capabilities.

## 6. Retrieval plane

- [ ] SQL/structured retrieval is represented.
- [ ] Vector retrieval is represented.
- [ ] Lexical retrieval is represented where specialized indexing is useful.
- [ ] Entity/assertion exact lookup is represented.
- [ ] Graph neighborhood/path/subgraph retrieval is represented.
- [ ] PPR/diffusion and community retrieval are represented.
- [ ] Tree/hierarchy retrieval is represented.
- [ ] Exact source/evidence recovery is represented.

## 7. Analytics / transformation plane

- [ ] DIGIMON is explicitly an analytical method suite, not only retrieval.
- [ ] Graph analytics include centrality, community detection and structural/subgraph methods.
- [ ] Non-graph analytics include structured aggregation/statistics and other reusable methods where supported.
- [ ] Retrieval outputs and analytic outputs can become typed inputs to later operations.
- [ ] Derived analytical state is distinguishable from source evidence.

## 8. Typed composability

- [ ] Current slot/record types are described accurately.
- [ ] Types are treated as reusable data semantics, not cognitive states.
- [ ] Target discussion allows concrete future types such as tables/models/findings only when capabilities require them.
- [ ] Composition examples include retrieval → transformation → further retrieval/analysis, not only query → answer pipelines.

## 9. Harness boundary

- [ ] Harness owns adaptive planning, sequencing, branching, retries, comparison and stopping.
- [ ] DIGIMON owns specialized capabilities, state, contracts and lineage.
- [ ] AoT/GoT/ReAct/decomposition are optional heuristics, not mandatory internal cognitive architecture.
- [ ] DIGIMON does not rebuild native harness capabilities merely for symmetry.

## 10. Public surfaces

- [ ] CLI is identified as human-facing.
- [ ] Python runtime is identified as application/developer-facing.
- [ ] MCP/tool protocol is identified as agent-facing.
- [ ] These are target interfaces over one maintained core.
- [ ] MCP is not described as the product itself.
- [ ] Legacy CLI/internal planner coupling is described as transitional where still true.

## 11. Evidence and derived state

- [ ] Source-backed semantic state, retrieval artifacts, derived analytic artifacts and findings are distinguished.
- [ ] Analytic outputs can retain method, parameters, inputs, representation version and uncertainty/evidence where applicable.
- [ ] Grounded final-answer citation is only one part of the broader provenance model.

## 12. Provenance / derivation graph

- [ ] Evidence provenance, semantic provenance and artifact/derivation lineage are distinguished.
- [ ] Source → semantic IR → representation → retrieval → analytic transformation → finding lineage is described.
- [ ] Domain/property graph and derivation/provenance graph are not conflated.
- [ ] Lineage supports reproducibility, invalidation/staleness, auditing and source recovery.
- [ ] Transformation identity/config/parameters and exact input/output artifact identities are part of the target lineage model.

## 13. Evidence → Action framing

- [ ] Documentation recognizes the larger workflow: gather evidence, represent it, retrieve a working set, apply methods, review/interpret and decide what to do next.
- [ ] Durable analytical residue includes evidence/provenance, structured knowledge, methods/parameters, findings/uncertainties and open questions.
- [ ] The principle **“Every analysis should make the next one easier”** is preserved as an orientation, not treated as an implementation claim.

## 14. Current state versus target state

- [ ] Recent retrieval/harness-first work is described as the current engineering focus, not the full north star.
- [ ] Current implemented capabilities are not confused with target representation families.
- [ ] Current source inspection is not presented as fresh runtime certification.
- [ ] Known runtime/CI verification limitations remain visible.

## Canonical document responsibilities

- `VISION.md` — durable project thesis, ecosystem boundary and complete north star.
- `ARCHITECTURE.md` — target system design that realizes the vision.
- `CURRENT_STATE.md` — what is actually implemented/verified now.
- `IMPLEMENTATION_MAP.md` — module-level code reality and classification.
- `GAP_ANALYSIS.md` — distance from current state to the full architecture/vision.
- `ROADMAP.md` — ordered closure plan.
- `README.md` — concise orientation and links into the canonical set.

When updating documentation, check this file before considering the reconciliation complete.
