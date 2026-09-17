# AGENTS.md — DIGIMON Contributor and Coding-Agent Guide

**Updated:** 2026-09-17

This file gives implementation guidance to coding agents. It should point to the canonical documentation rather than becoming another architecture/status document.

## Read these first

1. `docs/VISION.md` — durable north star.
2. `docs/CURRENT_STATE.md` — what is actually implemented now.
3. `docs/IMPLEMENTATION_MAP.md` — module classification and code caveats.
4. `docs/ARCHITECTURE.md` — target technical design.
5. `docs/GAP_ANALYSIS.md` — concrete current→target gaps.
6. `docs/ROADMAP.md` — dependency-ordered implementation sequence.
7. `docs/DOCUMENTATION_COVERAGE.md` — checklist for avoiding myopic documentation changes.
8. `docs/adr/002-harness-first-capability-architecture.md` — accepted decision about orchestration ownership.

`README.md` and `FUNCTIONALITY.md` are concise public views. Older MCP checkpoints, UKRF plans, internal-agent proposals and historical implementation reports are not current authority unless the canonical docs restate them.

## Project thesis versus control policy

Do not collapse these into one statement.

### Project thesis

DIGIMON is a **general text-derived representation, retrieval, and analytics runtime**. The canonical ecosystem input is governed semantic IR from onto-canon6. DIGIMON projects that semantic core into complementary representations, exposes specialized retrieval and analytical capabilities, preserves shared identity/provenance across them, and supports reusable evidence-to-action workflows.

The three capability planes are:

```text
REPRESENT → RETRIEVE → ANALYZE / TRANSFORM
```

### Control-policy decision

DIGIMON is **harness-first for adaptive reasoning**.

The external intelligent harness owns:

- interpreting the goal;
- deciding whether/how to decompose;
- selecting representations/tools;
- sequencing, branching and parallelization;
- retries/fallbacks;
- adapting after observations;
- stopping criteria.

DIGIMON owns:

- derived representations and their schemas/identity;
- specialized retrieval capabilities;
- analytical/transformation capabilities;
- typed contracts and compatibility facts;
- evidence/provenance/derivation facts;
- bounded local model-assisted operations where intrinsically required.

Do **not** add another general-purpose planner/orchestrator or mandatory cognitive state machine unless a new ADR explicitly changes this decision.

## Ecosystem boundary

### onto-canon6 owns

- source-backed semantic extraction/binding;
- ontology/profile semantics;
- governance/review;
- promoted assertions;
- canonical semantic identity/aliases;
- source/evidence provenance;
- governed Foundation-style IR/export.

### DIGIMON owns downstream projection/retrieval/analytics

Raw-document ingestion/chunking remains useful standalone/compatibility functionality but is not the conceptual ecosystem authority path.

## Representation families

The target representation plane includes:

- relational/tabular;
- vector;
- property graph;
- semantic/RDF graph where useful;
- hierarchy/tree;
- specialized lexical/full-text indexes where they add capability beyond harness-native search;
- source/evidence artifacts;
- progressive-disclosure wiki/catalog artifacts.

Geospatial is out of current text-focused scope.

Do not add a representation just for symmetry. Add it when it exposes a useful native operation family or materially improves agent navigation/analysis.

## Wiki/catalog rule

The target wiki is a **progressive-disclosure semantic and operational map** of the knowledge environment.

It should help an agent discover:

- what knowledge exists;
- how it is semantically organized;
- which representations exist;
- their schemas/ontologies;
- canonical IDs linking them;
- which specialized retrieval/analytic capabilities apply;
- where source/evidence lives.

Do not implement `wiki.open`, `wiki.follow`, basic grep/search wrappers merely for symmetry if the harness already provides those abilities well. Prefer generating high-quality artifacts that native harness tooling can navigate.

## Cross-representation identity is a core invariant

Preserve canonical IDs wherever possible:

- `entity_id`;
- `assertion_id`;
- `predicate_id`;
- `source_ref`;
- evidence/span identity.

A harness should be able to find an entity in the wiki, query it in SQL, traverse it in the graph, inspect vector metadata and recover evidence using explicit identity rather than fuzzy rediscovery.

Projection-local IDs may exist but should not become the only bridge across representations.

## Analytics are first-class

Do not treat DIGIMON as retrieval-only.

The canonical analytical pattern is:

```text
retrieve bounded working set
→ apply analytic/transformation method
→ produce typed derived artifact
→ reuse that artifact in later retrieval/analysis
```

Graph/SNA examples include centrality, PageRank/diffusion, Leiden/community detection, connected components, k-core/cohesion/density/assortativity, brokerage/bridging, shortest paths, PCST/Steiner and subgraph transforms.

Non-graph analytics should be added from concrete reusable needs such as SQL aggregation, descriptive statistics, clustering, anomaly detection or trend calculations.

Before adding new analytic methods, inventory existing code first and promote real current capabilities into coherent descriptors/types.

## Derived state is not source evidence

Keep these distinct:

1. source-backed semantic state;
2. retrieval artifacts/working sets;
3. derived analytical artifacts;
4. findings/interpretations.

A centrality score, community assignment, bridge classification, forecast or model output is derived state. It should retain method, parameters, input artifacts/representation version and evidence/uncertainty where relevant.

## Provenance terminology

Do not conflate:

- **evidence provenance** — which source supports a claim/answer;
- **semantic provenance** — how governed semantic assertions derive from source evidence/candidates;
- **artifact/derivation lineage** — what prior artifacts and transformation executions produced each projection, retrieval artifact, analytic artifact or finding.

The derivation/provenance graph is not the domain/property graph.

## Current code center

The strongest maintained implementation currently includes:

- `Core/Schema/SlotTypes.py` — typed values/records;
- `Core/Schema/OperatorDescriptor.py` — operator metadata;
- `Core/Operators/` and registry — retrieval/meta/utility capabilities;
- `Core/Composition/` — validation/execution/composition;
- `Core/Methods/` — ten maintained reference plans;
- graph/VDB/community/index/build implementations;
- `digimon_mcp_stdio_server.py` — strongest current agent-facing protocol facade.

The operator catalog is extensible/dynamic. **Do not hard-code a permanent operator count in documentation or new architecture.**

## Important current implementation facts

Do not reintroduce already-fixed defects or stale docs:

- required inputs must be explicitly wired by name;
- invalid plans fail closed by default; best-effort is explicit;
- control-flow body steps are not also executed top-level;
- loop carry-forward preserves real slot kinds;
- configured raw-document chunking occurs on the maintained standalone path;
- graph source manifests detect missing/changed/added chunks;
- successful rebuilds pragmatically invalidate known VDB/community/matrix artifacts;
- active graph identity and canonical VDB selection are explicit;
- answer generation fails closed on no evidence and validates citations;
- Unicode graph identity/text handling is preserved in maintained extraction/link/community paths;
- reference methods have received substantial wiring/evidence repairs;
- sparse matrices still have a same-shaped cross-graph identity edge case;
- current head still lacks a fresh runtime certification in the available environment.

See `docs/CURRENT_STATE.md` before assuming a gap still exists.

## Current implementation priorities

Follow `docs/ROADMAP.md`. The current sequence is:

1. get a real current-head deterministic test/canary run;
2. finish custom ontology wiring and first runtime reds;
3. verify the canonical onto-canon/Foundation IR handoff;
4. enforce cross-representation identity;
5. implement the first canonical relational projection;
6. generate the progressive-disclosure wiki/catalog;
7. inventory/promote existing graph analytics into a first-class typed analytic catalog;
8. add minimal derivation lineage across projection → retrieval → analysis;
9. bind remaining graph-derived resources such as sparse matrices to exact graph identity;
10. converge Python/CLI/MCP on the same maintained core;
11. expand deterministic architecture tests;
12. evaluate broadly only after those seams are real.

Do not replace this sequence with a new speculative framework unless a concrete failure requires it.

## Capability design guidance

For a new canonical specialized capability, prefer:

1. explicit typed inputs/outputs;
2. stable capability identity;
3. clear representation/resource prerequisites;
4. canonical cross-representation IDs where possible;
5. deterministic behavior where possible;
6. bounded/documented LLM use only when semantic judgment is intrinsic;
7. explicit source/derived-state behavior;
8. derivation metadata for material transformations;
9. machine-actionable failure semantics;
10. deterministic contract tests.

Do not hide resource-building side effects or silently fabricate evidence/identity.

## AoT / GoT / ReAct

These are reasoning heuristics, not mandatory DIGIMON runtimes. The harness may merge, skip, reorder, branch, parallelize or revise suggested subgoals.

Do not formalize a reasoning DAG merely because a prompt can express dependencies. Add such structure only when it enables a concrete function such as scheduling, resumability, caching, provenance or auditing.

## Legacy/transitional areas

Treat carefully:

- `Core/AOT/` — legacy programmed atomic-state/transition approach;
- `Core/AgentBrain/` — older broad internal planning layer;
- `Core/AgentOrchestrator/` — older/transitional orchestrators;
- `Core/Memory/` — earlier strategy/memory architecture;
- much of `Core/MCP/` — older MCP/coordination lineage;
- `digimon_cli.py` — human-facing but still coupled to old planner/orchestrator internals;
- historical MCP/checkpoint/UKRF/multi-agent documents.

Before deleting legacy code, identify live callers/tests. Before extending it, verify that the maintained core cannot serve the need more cleanly.

## Testing guidance

Prefer deterministic contract tests for:

- governed IR → projection behavior;
- cross-representation identity;
- typed retrieval/analysis composition;
- evidence versus derived-state semantics;
- graph/source freshness and invalidation;
- derivation lineage;
- supported Python/CLI/MCP parity;
- machine-actionable errors;
- clean build/reuse canaries.

Keep live provider/LLM suites separately classified.

Never cite an old test result as current runtime truth unless it has been rerun or explicitly labeled historical.

## Documentation maintenance

Before changing high-authority docs, read `docs/DOCUMENTATION_COVERAGE.md`.

When implementation changes status:

1. update `docs/CURRENT_STATE.md`;
2. update `docs/IMPLEMENTATION_MAP.md` when modules/contracts change;
3. reconcile `docs/GAP_ANALYSIS.md`;
4. update `docs/ROADMAP.md` if priority/exit criteria change;
5. update `docs/ARCHITECTURE.md` only for target-design changes;
6. update `docs/VISION.md` only for durable project-thesis/boundary changes;
7. create/update an ADR for a real architectural decision;
8. reconcile `README.md`, `FUNCTIONALITY.md`, `AGENTS.md`, `CLAUDE.md` and `QUICK_START.md` when guidance changes.

Do not create another competing current-status or roadmap document.

## Default decision rule

When choosing between making DIGIMON's internal agent brain more elaborate and making its representations, specialized capabilities, identity, evidence or derivation contracts clearer, prefer the latter unless a concrete supported requirement says otherwise.
