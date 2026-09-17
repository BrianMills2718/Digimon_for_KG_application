# DIGIMON Implementation Map

**Reconciled:** 2026-09-17  
**Purpose:** map the current repository to the canonical DIGIMON vision so contributors can distinguish maintained implementation, transitional surfaces, target-state gaps and historical lineage.

This document complements:

- `VISION.md` — project north star;
- `CURRENT_STATE.md` — current implementation truth;
- `ARCHITECTURE.md` — target technical design;
- `GAP_ANALYSIS.md` — current→target gaps;
- `ROADMAP.md` — ordered implementation sequence.

It is based on source inspection. It is **not** a claim that current head has passed a fresh full runtime certification.

## Current implementation center of gravity

The strongest maintained path today is:

```text
typed values / records
      ↓
capability implementations + descriptors
      ↓
strict composition / reference plans
      ↓
current graph / VDB / community / sparse resources
      ↓
exact source/evidence materialization
      ↓
grounded answer or explicit evidence gap
```

That is the current implementation center, not the full product thesis. The target remains broader:

```text
governed semantic IR
→ REPRESENT
→ RETRIEVE
→ ANALYZE / TRANSFORM
→ evidence / derived findings
```

## Module classification

| Path / surface | Classification | Current role | Guidance |
|---|---|---|---|
| `Core/Schema/SlotTypes.py` | **Canonical / Implemented** | Current typed retrieval/composition vocabulary | Extend only for reusable data semantics such as tables/derived artifacts when concrete capabilities require them |
| `Core/Schema/OperatorDescriptor.py` | **Canonical / Implemented, evolving** | Machine-readable operator metadata | Use as descriptor foundation; broaden carefully for real representation/analytics needs rather than speculative universal schema |
| `Core/Operators/registry.py` | **Canonical / Implemented, dynamic** | Registers typed operator families; utility operators may register alongside the original base set | Do not hard-code a permanent operator count in docs/architecture |
| `Core/Operators/` | **Canonical / Implemented** | Entity, relationship, chunk, subgraph, community, meta and utility operations | Preferred home for reusable typed retrieval/transform operations |
| `Core/Composition/ChainValidator.py` | **Canonical / materially hardened** | Static named-slot/type validation | Required inputs must be explicitly wired by input name; do not rely on old permissive same-kind auto-wiring assumptions |
| `Core/Composition/PipelineExecutor.py` | **Canonical / materially hardened** | Executes plans, loops and conditionals | Control-owned body steps are not re-run top-level; carried loop outputs preserve actual slot kind; keep runtime checks fail-closed |
| `Core/Composition/OperatorComposer.py` | **Canonical / Implemented** | Profiles/builds/executes maintained reference plans | Invalid plans are rejected by default; best-effort must be explicit; composer does not own global user-level routing policy |
| `Core/Methods/` | **Canonical reference layer / Implemented** | Ten maintained named compositions | Keep as useful shortcuts/regression baselines, not system identity |
| `Core/AgentSchema/context.py` | **Canonical foundation / pragmatic resource state** | Tracks target dataset, providers, graph/VDB instances and active graph state | Current pragmatic selection/invalidation is real; do not assume a generalized resource catalog is already necessary |
| `Core/AgentTools/graph_construction_tools.py` | **Canonical build implementation / hardened** | Shared lifecycle for ER/RK/tree/balanced/passage builds | Uses source-chunk manifests, truthful success, non-empty graph checks and known-artifact invalidation |
| `Core/AgentTools/derived_resource_cleanup.py` | **Canonical pragmatic lifecycle helper** | Deletes known stale derived artifacts after successful rebuild | Keep concrete; generalize only when new representation families require shared lifecycle semantics |
| `Core/AgentTools/graph_chunk_manifest.py` | **Canonical pragmatic freshness helper** | Detects changed/added/missing chunk sources before graph reuse | Important current source-freshness mechanism |
| `Core/AgentTools/cross_modal_tools.py` | **Implemented / Experimental integration** | Graph↔table↔vector transformations | Useful precursor to broader representation plane; not yet canonical governed-IR projection architecture |
| `Core/Common/EntityNormalization.py` | **Canonical helper / Implemented** | Unicode-safe graph identity and semantic-text normalization | Use identity normalization for graph keys/endpoints, semantic-text normalization for descriptions/keywords |
| `Core/Graph/`, `Core/Index/`, `Core/Chunk/`, `Core/Community/`, `Core/Provider/` | **Foundational implementation** | Graph/index/chunk/community/provider machinery | Keep behind stable typed/public boundaries where possible |
| `digimon_mcp_stdio_server.py` | **Implemented agent-facing protocol surface** | Current strongest modern MCP facade | MCP is an interface, not the product thesis; future public surfaces should converge on the same maintained core |
| `digimon_cli.py` | **Implemented / Transitional human surface** | CLI still uses `PlanningAgent`/`AgentOrchestrator`/optional ReAct | Modernize toward maintained runtime; do not deepen old internal-brain dependency |
| prospective `digimon/` public runtime | **Partial / target surface** | Intended developer/application library seam | Should expose maintained projection/resource/retrieval/analysis behavior cleanly |
| `Core/AgentBrain/` | **Legacy / Transitional** | Older broad internal planning logic used by some older entry points | Do not extend as default architecture |
| `Core/AgentOrchestrator/` | **Legacy / Transitional** | Multiple older orchestration generations | Identify callers and isolate/remove over time |
| `Core/AOT/` | **Legacy** | Programmed atomic states/transitions | Not target AoT/GoT policy; current decomposition is advisory |
| `Core/Memory/` | **Experimental / Legacy lineage** | Earlier strategy/memory architecture | Not current priority without concrete consumer need |
| older `Core/MCP/` | **Mixed legacy/experimental** | Older MCP clients/servers/coordination experiments | Do not infer current architecture from presence |
| `eval/` | **Implemented infrastructure / Deferred primary priority** | Benchmark/quality/cost evaluation | Preserve; do not let benchmarks define architecture before target seams are real |
| `tests/`, `testing/`, root `test_*.py` | **Mixed active/experimental estate** | Deterministic contracts, E2E, historical experiments | Current core tests need fresh execution before claiming green runtime |
| `api.py`, dashboards, Streamlit, React UI | **Secondary / mixed generation** | Alternate surfaces | Do not add architecture policy here; converge later or label clearly |

## Typed operator/composition core

### Current slot kinds

`Core/Schema/SlotTypes.py` currently defines:

1. `QUERY_TEXT`
2. `ENTITY_SET`
3. `RELATIONSHIP_SET`
4. `CHUNK_SET`
5. `SUBGRAPH`
6. `COMMUNITY_SET`
7. `SCORE_VECTOR`

These are current useful semantics, not a claim that the final representation/analytics algebra is complete. Likely future additions should come from real capabilities—for example `TABLE` or a derived-artifact/finding type—not from encoding thought-process states.

### Operator catalog

The original static registry included the main entity/relationship/chunk/subgraph/community/meta families. Maintained reference-plan construction now also registers utility operators such as materialization/merge helpers dynamically.

**Rule:** treat the operator catalog as discoverable runtime metadata, not a permanent hard-coded integer.

### Reference methods

Maintained plans:

- `basic_local`
- `basic_global`
- `lightrag`
- `fastgraphrag`
- `hipporag`
- `tog`
- `gr`
- `dalk`
- `kgp`
- `med`

Recent source-level repairs include:

- all methods normally terminate in grounded answer generation;
- Basic Global actually materializes selected community evidence;
- ToG/KGP use explicit hop unrolling rather than broken generic loop state;
- KGP accumulates earlier-hop evidence;
- GR/DALK/Med structural selections control evidence materialization;
- FastGraphRAG/HippoRAG explicitly request their intended PPR modes;
- structural paths/Steiner/PCST behavior is more truthful and fail-closed.

Fresh runtime execution is still required before calling these current-head certified.

## Composition behavior now

### Implemented/hardened

- explicit named-slot wiring checks;
- slot-kind/type validation;
- unknown named output rejection;
- invalid plans rejected by default;
- explicit best-effort only when intentionally requested;
- pre-dispatch runtime checks;
- loops/conditionals without duplicate top-level body execution;
- loop accumulation preserving actual slot kind;
- reference-plan profiling/execution.

### Remaining limitations

- compatibility/chain-discovery helpers remain heuristic and should not be confused with proof of semantic/resource executability;
- descriptor coverage is still centered on the retrieval/meta core rather than the full future Represent/Retrieve/Analyze plane;
- some generic meta/task concepts still use transitional types;
- error/result conventions remain uneven across all maintained/non-maintained surfaces.

## Representation map

### Property graphs — **strong/current**

- ER graph;
- RK graph;
- tree/balanced tree;
- passage graph;
- graph communities;
- sparse graph propagation structures.

### Vector — **strong/current**

- entity VDB;
- relationship VDB;
- FAISS-backed similarity retrieval;
- graph-to-vector related conversion code.

### Source/evidence — **strong/current foundations**

- exact chunk IDs/text;
- graph-source relationships;
- evidence materialization;
- grounded answer citation validation.

### Relational/tabular — **partial / target gap**

DataFrame/table conversions exist, but there is not yet a canonical governed-IR→relational database projection with stable cross-representation IDs and documented schema.

### Wiki/progressive-disclosure catalog — **planned**

No canonical generator exists yet for the agent-readable semantic/environment map described in `VISION.md`.

### Semantic/RDF graph — **planned / not canonical**

Not yet a maintained projection/runtime surface.

### Specialized lexical index — **planned selectively**

Only justified where BM25/fielded ranking adds value beyond native harness search.

## Graph/vector implementation facts

### Graph construction

All five maintained graph wrappers share a truthful lifecycle:

- choose graph-specific namespace;
- load current chunks;
- compare source manifest;
- force rebuild when source chunk set changes or caller requests it;
- require successful non-empty build;
- persist current manifest;
- invalidate known dependent artifacts only after a successful usable rebuild.

### Raw chunking

`ChunkFactory` now applies configured chunking for ordinary document records. Already-pre-chunked records remain compatible. Chunk IDs include document identity so identical boilerplate in different documents does not collapse, and global chunk indices preserve sparse-column alignment.

### Entity/relationship VDBs

Recent hardening includes:

- truthful build success;
- actual embedding dimension inference;
- correct FAISS score direction;
- typed entity seed extraction;
- exact graph match before approximate link;
- full entity identity content in entity embeddings;
- source/target identity included in relationship embedding text;
- canonical VDB prioritization.

### PPR / propagation

- reset/teleport semantics corrected;
- FastGraphRAG and HippoRAG pin different intended PPR modes;
- sparse propagation validates dimensions;
- unresolved evidence is skipped rather than fabricated;
- same-shaped cross-graph sparse matrix reuse remains a known identity gap.

## Analytics map

DIGIMON's current code already contains meaningful graph-oriented analysis/transformation capability, though it is not yet organized as a complete first-class analytics catalog.

Current examples include:

- PPR/diffusion scoring;
- community detection/materialization;
- k-hop/path/subgraph transforms;
- PCST optimization;
- Steiner approximation;
- score propagation/aggregation;
- tree/community-derived transforms;
- cross-modal graph/table/vector transformations.

The target analytical plane should inventory existing centrality/SNA/statistical capabilities before adding new ones, then promote reusable methods into typed descriptors/outputs.

## Evidence and derivation map

### Evidence provenance — current strong path

Current foundations include:

- `EntityRecord.source_id`;
- `RelationshipRecord.source_id`;
- exact `ChunkRecord.chunk_id`;
- graph→chunk/source materialization;
- `SlotValue.producer`/metadata;
- evidence IDs/provenance preserved into grounded answer results;
- answer citations validated against retrieved evidence.

### Semantic provenance — upstream authority

Onto-canon6 owns governed semantic/source provenance. DIGIMON should preserve those identities/provenance through projections rather than redefine semantic authority.

### Artifact/derivation lineage — target gap

Current manifests, producer metadata and invalidation rules are foundations, but there is not yet one first-class derivation graph recording:

```text
input artifact/version
→ projection/retrieval/analytic execution + parameters
→ output artifact/version
→ finding
```

This graph is distinct from the domain/property graph.

## Resource/freshness behavior

The old description “no canonical invalidation semantics” is now too broad.

Current pragmatic behavior includes:

- active dataset/graph selection;
- canonical VDB preference;
- in-memory same-dataset VDB eviction when graph is replaced;
- source-chunk manifests for graph freshness;
- removal of canonical entity/relation VDB artifacts after successful rebuild;
- community report/map invalidation;
- ER sparse-matrix invalidation;
- fail-closed use of stale in-memory community reports.

This is not a generalized enterprise resource catalog, and that is intentional. Generalize only when concrete new projections/analytics require shared machinery.

## Unicode identity/text

Maintained graph extraction/link/community paths now distinguish:

- Unicode-safe graph identity normalization; and
- Unicode-safe semantic text cleanup.

Non-ASCII names/descriptions should not be erased merely because older `clean_str()` behavior was ASCII-oriented.

## Prompt/reasoning surfaces

Current policy remains:

- AoT/GoT/ReAct/decomposition are advisory heuristics;
- harness may ignore/reorder/branch/parallelize/revise;
- malformed decomposition falls back conservatively rather than turning arbitrary prose into planner state;
- reason-step refinement is evidence-gated;
- synthesis is evidence-gated.

Prompt duplication between YAML and operator-local text may still merit cleanup, but it is not the project north star.

## Public/control surfaces

### MCP

Current strongest modern agent-facing protocol facade. Useful for specialized DIGIMON state/capabilities, but not the product identity.

### CLI

Human-facing but internally transitional because it still uses older PlanningAgent/AgentOrchestrator logic.

### Python runtime

Target developer/application surface; needs consolidation over the same maintained core.

Target: **CLI + Python + MCP over one core**.

## Harness-native capability boundary

Do not build DIGIMON wrappers merely because a conceptual capability can be named.

If the external harness already handles file reading, grep/text search, link following, directory/wiki navigation, planning, sequencing or retries well, DIGIMON should generally generate good artifacts and expose only the specialized state/engines it owns.

## Testing / runtime verification

Many deterministic contract tests have been added around the repaired semantics above. However:

- current head has not been run in this environment;
- current connector-created commits/PRs have not produced fresh GitHub Actions runs;
- historical Actions failures occurred during dependency installation before meaningful testing.

Do not claim current-head green status until a real runner executes:

```bash
pip install -r requirements-minimal.txt
pytest tests/core -q
python tests/e2e/test_mcp_smoke.py
DIGIMON_CANARY_REBUILD=1 python tests/e2e/test_mcp_smoke.py
```

## Canonical documentation

- `README.md`
- `FUNCTIONALITY.md`
- `docs/README.md`
- `docs/VISION.md`
- `docs/CURRENT_STATE.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/ARCHITECTURE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`
- `docs/DOCUMENTATION_COVERAGE.md`
- `docs/PLANNING_SUMMARY.md`
- `docs/AGENT_INTELLIGENCE_ENHANCEMENTS.md`
- `docs/FUTURE_EVALUATION_QUESTIONS.md`
- `AGENTS.md`
- `CLAUDE.md`

`docs/adr/002-harness-first-capability-architecture.md` remains the accepted **orchestration-ownership** decision, not the complete project vision.

## Contributor decision rule

Before adding an abstraction, ask:

1. Does it support Represent, Retrieve or Analyze/Transform?
2. Is it a specialized capability/data/identity/derivation fact DIGIMON should own, or orchestration policy the harness already owns?
3. Does it preserve canonical cross-representation identity?
4. Does it preserve or explicitly characterize evidence/derivation lineage?
5. Is there an existing implementation that should be promoted instead of rebuilt?
6. Does a concrete failure/use case justify generalization?

Prefer concrete improvements to representations, specialized capabilities, identity, evidence and derivation over another internal reasoning framework.

## Where to work next

Follow `ROADMAP.md`:

1. fresh current-head deterministic execution;
2. custom ontology + first real runtime failures;
3. canonical onto-canon/Foundation IR handoff;
4. cross-representation identity;
5. canonical relational projection;
6. progressive-disclosure wiki/catalog;
7. first-class analytic capability inventory/promotion;
8. derivation records across projection→retrieval→analysis;
9. exact graph identity for remaining derived resources;
10. Python/CLI/MCP convergence;
11. deterministic architecture tests;
12. broader evaluation later.
