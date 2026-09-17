# CLAUDE.md — DIGIMON Implementation Guide

**Updated:** 2026-09-17

This repository contains multiple generations of DIGIMON architecture. Use the canonical documentation rather than inferring current intent from older planners, checkpoint files or historical reports.

## Canonical documentation

Read in this order:

1. `docs/VISION.md` — durable project north star.
2. `docs/CURRENT_STATE.md` — current implementation truth and verification boundary.
3. `docs/IMPLEMENTATION_MAP.md` — exact module classification/code caveats.
4. `docs/ARCHITECTURE.md` — target technical design.
5. `docs/GAP_ANALYSIS.md` — current→target gaps.
6. `docs/ROADMAP.md` — ordered implementation sequence.
7. `docs/DOCUMENTATION_COVERAGE.md` — coverage checklist for high-authority docs.
8. `docs/adr/002-harness-first-capability-architecture.md` — accepted orchestration-ownership decision.
9. `docs/README.md` — documentation hierarchy/maintenance rules.

## Do not confuse thesis with control policy

### Project thesis

DIGIMON is a **general text-derived representation, retrieval, and analytics runtime** downstream of governed semantic IR from onto-canon6.

Target flow:

```text
governed semantic IR
→ REPRESENT
→ RETRIEVE
→ ANALYZE / TRANSFORM
→ grounded evidence / derived findings
```

DIGIMON should project one semantic core into complementary representations, preserve shared canonical identity across them, expose specialized retrieval/analytic capabilities, and preserve artifact/derivation lineage so outputs can be reused safely.

### Control-policy decision

DIGIMON is **harness-first for adaptive reasoning**.

The external harness owns:

- goal interpretation;
- decomposition decisions;
- representation/tool selection;
- sequencing/branching/parallelism;
- retries/fallbacks;
- adaptation after observations;
- stopping.

DIGIMON owns:

- derived representations and schemas;
- specialized retrieval capabilities;
- analytic/transformation capabilities;
- typed compatibility facts;
- identity/evidence/derivation metadata;
- bounded local model-assisted operations.

Do not build another general-purpose internal agent brain unless a new ADR explicitly changes this decision.

## Ecosystem boundary

onto-canon6 owns semantic extraction/binding, ontology/profile semantics, governance/review, promoted assertions, canonical identity/aliases, source/evidence provenance and governed Foundation-style IR/export.

DIGIMON owns downstream representation/retrieval/analysis projections.

Raw-document ingestion/chunking remains useful standalone/compatibility functionality, not the conceptual center of the ecosystem architecture.

## Representation families

Current/target families include:

- relational/tabular;
- vector;
- property graph;
- semantic/RDF graph where useful;
- hierarchy/tree;
- specialized lexical/full-text indexes where they add capability beyond harness-native search;
- source/evidence artifacts;
- progressive-disclosure wiki/catalog artifacts.

Geospatial is outside current text-derived scope.

Do not add a new representation merely for symmetry. Add it when it unlocks a useful native operation family or materially improves agent navigation/analysis.

## Wiki/catalog rule

The target wiki is an agent-readable progressive-disclosure map of both:

- semantic content; and
- the retrieval/analytic environment.

It should expose knowledge organization, representations, schemas/ontologies, canonical IDs joining them, specialized capabilities, and evidence/source paths.

Do not create `wiki.open`, `wiki.follow` or basic file-search wrappers when the external harness already has good native file/link/search abilities.

## Cross-representation identity

Preserve canonical identity wherever possible:

- `entity_id`;
- `assertion_id`;
- `predicate_id`;
- `source_ref`;
- evidence/span identity.

The harness should be able to find an entity in the wiki/catalog and then address the corresponding SQL row, graph node, vector metadata and evidence directly rather than rediscovering identity fuzzily.

## Analytics are first-class

DIGIMON is not retrieval-only.

Canonical pattern:

```text
retrieve bounded working set
→ apply analysis/transformation
→ produce typed derived artifact
→ feed result into later retrieval/analysis
```

Graph/SNA capabilities are especially important to project lineage: centrality, PageRank/diffusion, Leiden/community detection, connected components, k-core/cohesion/density/assortativity, brokerage/bridging, shortest paths, PCST/Steiner and subgraph transformations.

Promote existing real analytic code before inventing broad new method families.

Add non-graph analytics from concrete reusable needs such as SQL aggregation, descriptive statistics, clustering, anomaly detection and temporal trends.

## Source evidence versus derived state

Keep distinct:

1. source-backed semantic state;
2. retrieval artifacts / bounded working sets;
3. derived analytic artifacts;
4. findings / interpretations.

A community assignment, centrality score, bridge classification, diffusion path, forecast or model result is derived state. It should carry method, parameters, inputs/representation version and relevant evidence/uncertainty.

## Provenance terminology

Do not conflate:

- **evidence provenance** — what source supports a claim/answer;
- **semantic provenance** — how governed assertions relate to source candidates/evidence;
- **artifact/derivation lineage** — which prior artifacts and transformation executions produced each projection, retrieval artifact, analytical artifact or finding.

The derivation/provenance graph is distinct from the domain/property graph.

## Current code center

The strongest maintained implementation currently includes:

- `Core/Schema/SlotTypes.py` — typed dataflow records;
- `Core/Schema/OperatorDescriptor.py` — operator metadata;
- `Core/Operators/` + registry — retrieval/meta/utility implementations;
- `Core/Composition/` — validation/execution/composition;
- `Core/Methods/` — ten maintained reference plans;
- graph/index/VDB/community/build implementations;
- `digimon_mcp_stdio_server.py` — strongest current agent-facing protocol facade.

The operator catalog can be dynamically extended. Do **not** hard-code a permanent operator count in new documentation/architecture.

## Current implementation facts

Before modifying the core, do not rely on stale 2026-09-16 caveats:

- required inputs now need explicit named wiring;
- invalid plans fail closed by default; best-effort is explicit;
- generic loop/conditional body steps are not also executed top-level;
- loop carry-forward preserves actual slot kinds;
- configured raw-document chunking now occurs on the maintained standalone path;
- graph source manifests detect changed/added/missing chunk inputs;
- successful rebuilds pragmatically invalidate known VDB/community/matrix artifacts;
- active graph identity and canonical VDB selection are explicit;
- answer generation fails closed on no evidence and validates citation IDs;
- Unicode graph identity/free text is preserved on maintained extraction/link/community paths;
- major reference-method wiring/evidence defects were repaired;
- same-shaped cross-graph sparse-matrix identity remains a real edge case;
- the current head still lacks a fresh runtime certification in the available environment.

Read `docs/CURRENT_STATE.md` before assuming a previously documented gap remains.

## MCP / CLI / Python surfaces

- **MCP** — current strongest agent-facing specialized capability surface.
- **CLI** — human-facing but still uses older `PlanningAgent` / `AgentOrchestrator` internals.
- **Python runtime** — target developer/application-facing surface; not yet fully consolidated.

The target is CLI + Python + MCP over one maintained core. MCP is not the project identity.

## AoT / GoT / ReAct

Treat these as optional reasoning heuristics. The harness may merge, skip, reorder, branch, parallelize or revise suggested subgoals.

Do not create a mandatory reasoning DAG/state machine just because dependency-aware prompting is useful. Add formal graph structure only when it enables a concrete capability such as scheduling, resumability, caching, provenance or auditing.

## Legacy/transitional architecture

Present but not the target center:

- `Core/AOT/`;
- `Core/AgentBrain/`;
- older `Core/AgentOrchestrator/` variants;
- `Core/Memory/`;
- older `Core/MCP/` experiments;
- `digimon_cli.py` internals using PlanningAgent/AgentOrchestrator;
- historical WebSocket/MCP checkpoint/UKRF/multi-agent plans.

Before deleting legacy code, identify live callers/tests. Before extending it, ask whether the maintained representation/retrieval/analytics core should own the requirement instead.

## Current implementation priority

Follow `docs/ROADMAP.md`. The current sequence is:

1. obtain a real current-head deterministic test/canary signal;
2. finish custom ontology wiring and first runtime reds;
3. verify the canonical onto-canon/Foundation IR handoff;
4. establish cross-representation identity;
5. implement the first canonical relational projection;
6. generate the first progressive-disclosure wiki/catalog;
7. inventory/promote existing graph analytics into a coherent typed analytic plane;
8. add minimal derivation records across projection → retrieval → analysis;
9. bind remaining graph-derived resources to exact graph identity;
10. converge Python/CLI/MCP on the same maintained core;
11. broaden deterministic architecture tests;
12. make broad evaluation/benchmarking primary only after those seams are real.

Do not replace this with another speculative framework unless a concrete failure requires it.

## Capability implementation rules

For new canonical specialized capabilities, prefer:

- explicit typed inputs/outputs;
- stable capability identity;
- explicit representation/resource prerequisites;
- canonical cross-representation IDs;
- deterministic behavior where possible;
- bounded/documented LLM use only where semantically intrinsic;
- explicit source-evidence versus derived-state semantics;
- derivation metadata for material transformations;
- machine-actionable failure semantics;
- deterministic contract tests.

Avoid hidden build side effects, fabricated evidence and tool-specific identity bridges.

## Testing

Prioritize deterministic tests for:

- governed IR → projection behavior;
- cross-representation identity;
- typed retrieval→analytic composition;
- graph/source freshness and invalidation;
- evidence versus derived-state semantics;
- derivation lineage;
- public Python/CLI/MCP parity for supported operations;
- clean reuse/rebuild canaries.

Keep live-provider/LLM suites separate. Historical test/benchmark numbers are not current guarantees unless rerun.

## Documentation maintenance

Before editing high-authority docs, read `docs/DOCUMENTATION_COVERAGE.md`.

When implementation changes status:

1. update `docs/CURRENT_STATE.md`;
2. update `docs/IMPLEMENTATION_MAP.md` for module/contract changes;
3. reconcile `docs/GAP_ANALYSIS.md`;
4. update `docs/ROADMAP.md` when priorities/exit criteria change;
5. change `docs/ARCHITECTURE.md` only when target technical design changes;
6. change `docs/VISION.md` only when the durable project thesis/boundary changes;
7. record meaningful design decisions under `docs/adr/`;
8. align `README.md`, `FUNCTIONALITY.md`, `AGENTS.md`, this file and `QUICK_START.md` when guidance changes.

Do not create another competing current-state or roadmap document.

## Default implementation judgment

When the choice is between adding more internal reasoning machinery and making DIGIMON's representations, specialized capabilities, cross-representation identity, evidence or derivation contracts clearer, prefer the latter unless a concrete supported task requires otherwise.
