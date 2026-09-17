# DIGIMON Documentation Index

**Canonical snapshot date:** 2026-09-17

This directory contains current architecture/status documentation plus older research/planning material. The canonical set below is the source of truth for the repository. Historical reports, handoffs and exploratory plans remain useful lineage but must not override it.

## Canonical documentation

1. **[VISION.md](VISION.md)** — durable north star: governed semantic IR → Represent → Retrieve → Analyze/Transform, with shared identity and derivation lineage.
2. **[CURRENT_STATE.md](CURRENT_STATE.md)** — what the code materially contains now and what remains unverified at runtime.
3. **[IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)** — module-by-module implementation/classification detail.
4. **[ARCHITECTURE.md](ARCHITECTURE.md)** — target system boundaries, representation/retrieval/analytics architecture, harness boundary and provenance model.
5. **[GAP_ANALYSIS.md](GAP_ANALYSIS.md)** — concrete current→target gaps.
6. **[ROADMAP.md](ROADMAP.md)** — dependency-ordered closure sequence.
7. **[DOCUMENTATION_COVERAGE.md](DOCUMENTATION_COVERAGE.md)** — checklist preventing future docs from collapsing onto the latest implementation topic.
8. **[PLANNING_SUMMARY.md](PLANNING_SUMMARY.md)** — concise current execution plan derived from the roadmap.
9. **[../FUNCTIONALITY.md](../FUNCTIONALITY.md)** — concise implemented-capability view.
10. **[QUICK_START.md](QUICK_START.md)** — current setup/entry points.
11. **[AGENT_INTELLIGENCE_ENHANCEMENTS.md](AGENT_INTELLIGENCE_ENHANCEMENTS.md)** — reasoning-policy detail; harness-first is a control boundary, not the project thesis.
12. **[FUTURE_EVALUATION_QUESTIONS.md](FUTURE_EVALUATION_QUESTIONS.md)** — deliberately deferred benchmarking/research questions.

## Read the docs in this order

```text
README.md
   ↓
docs/VISION.md
   ↓
docs/CURRENT_STATE.md
   ↓
docs/ARCHITECTURE.md
   ↓
docs/IMPLEMENTATION_MAP.md
   ↓
docs/GAP_ANALYSIS.md
   ↓
docs/ROADMAP.md
```

Use `DOCUMENTATION_COVERAGE.md` when changing any high-authority documentation.

## The hierarchy to preserve

Future summaries should keep these levels separate:

### Project thesis

DIGIMON is a **general text-derived representation, retrieval and analytics runtime**. Its canonical ecosystem input is governed semantic IR from onto-canon6; it projects that semantic core into complementary representations, supports composable retrieval and analysis, and preserves identity/evidence/derivation lineage.

### Architectural/control decision

The external intelligent harness owns adaptive reasoning policy. DIGIMON should not rebuild the harness's planning, sequencing, branching, retries or stopping logic. AoT/GoT/ReAct are optional heuristics rather than a mandatory internal cognitive runtime.

### Current engineering workstream

Recent implementation work has concentrated on making the graph/vector retrieval core truthful: composition semantics, graph builds, chunking, VDBs, PPR, reference methods, grounding, resource selection, invalidation and evidence propagation.

The current workstream is narrower than the project thesis.

## Ecosystem ownership boundary

### onto-canon6

Owns governed semantic authority:

- source-backed extraction/binding;
- ontology/profile semantics;
- review/governance;
- promoted assertions;
- identity/aliases/canonicalization;
- source/evidence provenance;
- governed Foundation-style IR/export.

### DIGIMON

Owns downstream retrieval/analysis projections:

- property graphs and graph-derived structures;
- vector indexes;
- relational/tabular projections;
- semantic/RDF projection where useful;
- specialized lexical indexes where native harness search is insufficient;
- hierarchy/tree structures;
- agent wiki/progressive-disclosure catalog;
- specialized retrieval capabilities;
- analytic/transformation capabilities;
- cross-representation identity;
- evidence recovery and artifact/derivation lineage.

Raw-document ingestion/chunking remains valid standalone/benchmark/compatibility functionality but is not the conceptual center of the ecosystem architecture.

## Three capability planes

Canonical docs must preserve all three:

1. **Represent** — derive useful structures from governed semantic IR.
2. **Retrieve** — obtain bounded working sets/evidence using each representation's strengths.
3. **Analyze / transform** — run analytical methods over those working sets and produce reusable derived artifacts.

Graph analytics are particularly important to DIGIMON's SNA lineage, but the target system is representation-general.

## Wiki/catalog meaning

The target wiki is a progressive-disclosure **knowledge and environment map for agents**, not merely a document store and not necessarily a special tool API.

It should describe:

- semantic organization of entities/concepts/topics/sources;
- available representations;
- ontology/schemas;
- canonical IDs joining representations;
- specialized retrieval/analytic capabilities;
- source/evidence paths.

If the harness already opens files, searches text and follows links well, DIGIMON should materialize good wiki artifacts and let the harness use native capabilities rather than wrapping them for symmetry.

## Cross-representation identity

Canonical IDs such as `entity_id`, `assertion_id`, `predicate_id`, `source_ref` and evidence identity should survive projections where possible. This enables agent workflows such as:

```text
find Alice in the wiki
→ use canonical entity ID in SQL
→ use the same ID for graph traversal
→ use vector metadata for semantic expansion
→ recover exact evidence
```

The harness decides that workflow; DIGIMON provides the interoperable representations/capabilities.

## Analytics and derived state

Retrieval outputs should be valid inputs to analytical transformations. Examples include:

```text
ENTITY_SET → k-hop → SUBGRAPH → Leiden → COMMUNITY_SET
SUBGRAPH → centrality → SCORE_VECTOR → top-k → ENTITY_SET → evidence
TABLE → aggregate → TABLE
```

Derived outputs such as communities, centrality, bridge classifications, diffusion paths, forecasts or model results must remain distinguishable from original evidence and should carry method/parameter/input lineage.

## Provenance terminology

Keep these separate:

- **evidence provenance** — which source supports a claim/answer;
- **semantic provenance** — how governed assertions relate to source evidence/candidates;
- **artifact/derivation lineage** — which prior artifacts and transformation executions produced each projection, retrieval artifact, analytic artifact or finding.

The derivation graph is not the domain/property graph.

## Access surfaces

The intended surfaces are:

- **CLI** — human-facing;
- **Python runtime** — application/developer-facing;
- **MCP/tool protocol** — agent-facing specialized capabilities.

They should converge on one maintained core. MCP is not the product identity. The current CLI remains transitional because it uses older internal planner/orchestrator code.

## Architecture decisions

- **[adr/002-harness-first-capability-architecture.md](adr/002-harness-first-capability-architecture.md)** — accepted control-policy decision: external harness owns adaptive orchestration; DIGIMON owns capabilities/contracts/resources/evidence boundaries.
- `adr/001-agent-orchestration-architecture.md` — superseded earlier dual-brain design.

ADR-002 should not be paraphrased as the overall DIGIMON north star; it answers a narrower orchestration-ownership question.

## Status vocabulary

- **Implemented** — meaningful code exists and is wired into a maintained/current surface by source inspection.
- **Partial** — meaningful code exists but target coverage/integration/runtime proof is incomplete.
- **Legacy / Transitional** — retained for history/compatibility or older live callers, not preferred target architecture.
- **Planned** — target capability is not materially complete.

Current head has not yet received a fresh complete runtime certification in the available environment. Do not translate source-level contract coverage into a claim of green CI/runtime.

## Current truths worth keeping visible

- composition validation/execution has been materially hardened and invalid plans fail closed by default;
- generic loop/conditional execution no longer double-runs control-owned steps and carried values preserve their slot kinds;
- configured raw-document chunking now actually occurs on the maintained standalone path;
- graph source manifests detect changed/added chunk inputs and trigger rebuilds;
- known VDB/community/matrix artifacts are invalidated pragmatically after successful rebuilds;
- active graph identity and canonical VDB selection are explicit;
- grounding fails closed with no evidence and validates answer citation IDs;
- Unicode entity identity/free-text handling is safer on maintained graph paths;
- reference methods have received substantial wiring/evidence repairs;
- analytics exist substantially in graph-focused form but are not yet organized as a complete first-class capability plane;
- relational, wiki/catalog, RDF and derivation-graph target surfaces are not yet complete;
- current deterministic tests are source-reviewed but need a real current-head run.

See `CURRENT_STATE.md` and `IMPLEMENTATION_MAP.md` for detail.

## Historical and supporting material

Older documents that prescribe a hand-built general-purpose cognitive state machine, mandatory AoT/Markov preprocessing, old WebSocket MCP checkpoint sequences, multi-agent coordination as the immediate priority, or graph-only/product framing should be treated as historical unless restated in the canonical documentation.

## Maintenance rule

When architecture or implementation changes:

1. check `DOCUMENTATION_COVERAGE.md` first;
2. update `VISION.md` only if the durable project thesis/boundary changes;
3. update `CURRENT_STATE.md` for code reality;
4. update `IMPLEMENTATION_MAP.md` for module/capability classification;
5. update `ARCHITECTURE.md` for target technical design changes;
6. reconcile `GAP_ANALYSIS.md`;
7. update `ROADMAP.md` priorities/exit criteria;
8. refresh `PLANNING_SUMMARY.md`;
9. reconcile root `README.md`, `FUNCTIONALITY.md`, `AGENTS.md`, `CLAUDE.md` and `QUICK_START.md` when their guidance is affected.

Do not create another competing current-state or roadmap document.
