# DIGIMON Current State

**Snapshot:** 2026-09-17  
**Repository:** `BrianMills2718/Digimon_for_KG_application`  
**Purpose:** describe what is materially present in the code now, without confusing the full north star with current implementation or historical plans.

## Executive summary

DIGIMON remains a **hybrid/transitional codebase**, but the maintained core is substantially more coherent than the 2026-09-16 documentation described.

The strongest current implementation is a typed representation/retrieval/analysis core with:

- typed operator slots and records;
- strict-by-default composition validation/execution;
- ten maintained reference methods;
- ER/RK/tree/balanced-tree/passage graph build surfaces;
- vector indexes and graph-derived resources;
- entity/relationship/chunk/subgraph/community operations;
- graph structural/analytic operators such as PPR, PCST, Steiner and community-related transforms;
- evidence-grounded synthesis with exact evidence IDs and citation validation;
- source/chunk manifests and practical invalidation rules for derived resources;
- active dataset/graph and canonical VDB selection;
- a stdio MCP surface plus transitional CLI/API surfaces;
- deterministic core contract tests covering many repaired semantics.

The target project vision is broader: DIGIMON should eventually project governed semantic IR into complementary representations, support composable retrieval and analytics over them, materialize an agent-readable progressive-disclosure catalog/wiki, preserve shared identity across projections, and maintain artifact/derivation lineage. See [VISION.md](VISION.md).

This file is a **source-level current-state reconciliation**, not a fresh full runtime certification. The current head has not been executed end-to-end in the available environment. Historical GitHub Actions failed during dependency installation before tests, while commits made through the connected GitHub path have not generated new workflow runs. “Implemented” below means substantive code is present and wired, not that every current path is freshly certified green.

## Status vocabulary

- **Implemented** — meaningful code exists and is wired into a maintained/current surface.
- **Partial** — meaningful code exists, but coverage/integration/reliability or the target architecture remains incomplete.
- **Legacy / Transitional** — retained for compatibility/history or still used by an older entry point but not the preferred target architecture.
- **Planned** — part of the target vision but not materially complete in current code.

## Current capability map

| Area | Status | Current reality |
|---|---|---|
| Typed operator dataflow | **Implemented** | Current core has typed query/entity/relationship/chunk/subgraph/community/score-vector values plus producer/metadata lineage. |
| Operator registry | **Implemented / evolving** | Stable typed operators exist across entity, relationship, chunk, subgraph, community and meta families. Utility adapters such as materialization/merge operators can register alongside the original base set, so documentation should not depend on a permanent hard-coded operator count. |
| Composition validation | **Implemented / materially hardened** | Explicit wiring/type validation is stricter; omitted required named inputs no longer pass merely because a same-kind value exists elsewhere. Unknown named outputs are errors. |
| Composition execution | **Implemented / materially hardened** | `OperatorComposer` rejects static-invalid plans by default; best-effort is explicit. Generic loop/conditional ownership was repaired so control-owned steps are not executed again at top level, and loop carry-forward preserves actual slot kinds. |
| Reference methods | **Implemented / repaired** | Ten maintained plans: `basic_local`, `basic_global`, `lightrag`, `fastgraphrag`, `hipporag`, `tog`, `gr`, `dalk`, `kgp`, `med`. They normally terminate in grounded answer generation; context-only mode strips only the terminal answer step. |
| Canonical raw corpus path | **Implemented / standalone mode** | Raw corpus preparation exists and the maintained `ChunkFactory` now actually applies configured chunking instead of treating each document as one chunk. Pre-chunked records remain supported. This is useful standalone/compatibility behavior, not the canonical ecosystem authority boundary. |
| Graph build lifecycle | **Implemented / hardened** | All five graph wrappers return truthful success/failure, require usable non-empty graphs, share one lifecycle path, use source-chunk manifests to detect corpus changes, and invalidate known derived artifacts after successful rebuilds. |
| ER/RK extraction | **Implemented / hardened** | Rich metadata defaults are retained; RK enables keyword extraction; zero-node extraction fails closed; entity identity/text normalization now preserves Unicode rather than deleting non-ASCII semantics. |
| Vector indexes / FAISS | **Implemented / hardened** | Dimensions derive from actual embeddings, L2 distances are normalized in the correct direction, persisted/load state is fail-closed, batch retrieval exists, and single-item upsert no longer replaces the entire collection. |
| Entity VDB | **Implemented / hardened** | Build refuses false success; stale registered indexes reload/rebuild; graph-prepared identity content is embedded; graph-native metadata survives indexing; stable graph IDs are preferred; exact graph entity linking precedes approximate VDB linking. |
| Relationship VDB | **Implemented / hardened** | Exact requested IDs are preserved, build success is checked, retrieval uses normalized scores, endpoints/source metadata survive mapping, and embedding text includes source/target identity plus relation semantics. |
| Entity retrieval / linking | **Implemented** | Vector retrieval, one-hop, PPR, linking, TF-IDF and model-assisted extraction paths exist. Exact canonical graph matches bypass VDB approximation. |
| PPR / diffusion | **Implemented / corrected** | Teleport/reset vs damping semantics were corrected; maintained methods explicitly select intended similarity-vs-specificity behavior; seed fallback handles valid entities when matrix lookup misses. |
| Relationship retrieval | **Implemented** | One-hop, VDB, score aggregation and model-assisted relation selection exist. ToG relation selection now explores the whole beam and preserves exact source chunk IDs. |
| Chunk/evidence retrieval | **Implemented / evidence-safe** | Occurrence, relation→chunk and score→chunk paths resolve exact stored source IDs, normalize real `TextChunk` objects, and fail closed instead of fabricating placeholder/fuzzy evidence. |
| Subgraph/path retrieval | **Implemented / repaired** | K-hop/path normalization no longer invents adjacency across concatenated paths; Steiner is a real NetworkX approximation and degrades to the best connected terminal group on disconnected inputs; PCST prizes/costs reflect retrieved relevance. |
| Structural materialization | **Implemented** | Subgraph selections are materialized back to exact graph entities/source chunks, allowing GR/DALK/Med structural choices to control evidence rather than remain decorative. |
| Community operations | **Implemented / hardened** | Community materialization exists; singleton graphs bypass unnecessary Leiden calls; persisted metadata identity/level/occurrence comes from authoritative schema; set-valued schema fields serialize safely; stale community artifacts are invalidated/rejected after graph rebuild. |
| Basic Global | **Implemented / repaired** | Community selection now reaches exact report/evidence materialization and grounded answer generation. |
| ToG | **Implemented / repaired** | Explicit depth unrolling replaces the broken pseudo-loop; each hop consumes prior entities; relationship→entity bridge works; beam entities are all explored; evidence is accumulated across selected hops. |
| KGP | **Implemented / repaired** | Explicit hop unrolling replaces stale-state loop behavior; refinement is evidence-gated; TF-IDF uses direct sklearn with stop-word fallback; evidence can accumulate across hops. |
| FastGraphRAG | **Implemented / corrected** | Typed entity seeds are mapped correctly into score-matrix lookup; similarity-seeded PPR is requested explicitly; sparse propagation fails on shape mismatch rather than producing plausible wrong rankings. |
| HippoRAG | **Implemented / corrected** | Extract/link/PPR propagation is wired, and the method explicitly requests its intended specificity/IDF-aware PPR mode instead of silently sharing FastGraphRAG defaults. |
| GR / DALK / Med | **Implemented / repaired** | Retrieved/filtered structural results now feed materialization/evidence paths; disconnected Med terminals degrade explicitly instead of collapsing the method. |
| Grounded answer generation | **Implemented / hardened** | No evidence => no LLM call and explicit insufficient-evidence status. Evidence enters prompts with exact IDs; citations are validated; evidence IDs/provenance survive output metadata and composer serialization. |
| Meta decomposition | **Implemented / advisory** | Dependency-aware decomposition is explicitly optional/harness-controlled; malformed model output falls back conservatively to the original question rather than inventing subquestions from prose. |
| Meta synthesis | **Implemented / fail-closed** | Empty evidence/subanswer synthesis does not invite unsupported LLM bridging. |
| Resource selection | **Implemented / pragmatic** | `GraphRAGContext` tracks active dataset/graph, prioritizes exact canonical VDB IDs, restores canonical graph namespaces at registration and evicts same-dataset in-memory VDBs when a graph is replaced. |
| Derived-resource invalidation | **Implemented / pragmatic, not generalized** | Successful graph rebuilds remove canonical entity/relation VDB artifacts and community files; ER rebuilds also invalidate sparse matrices. This is intentionally smaller than a generalized resource-governance system. |
| Graph/source freshness | **Implemented / pragmatic** | Per-graph source-chunk manifests detect missing/changed/added chunks and force rebuild before reuse. |
| Sparse propagation | **Implemented / guarded** | Entity→relationship and relationship→chunk propagation validate matrix dimensions; unresolved chunk IDs/unknown objects no longer become pseudo-evidence. Same-shaped cross-graph matrix reuse remains a known edge case. |
| Cross-modal graph/table/vector code | **Implemented / experimental integration** | Conversion code exists, but it is not yet the complete projection system described in the vision and does not yet include all target representation families/catalog metadata. |
| Analytical tool suite | **Partial** | Substantial graph analytics/transformations exist (PPR, community, PCST, Steiner, paths and related score/subgraph operations). The broader general analytical suite—centrality families, statistical/table analytics, typed finding/model artifacts—is not yet organized as a complete first-class capability plane. |
| Relational projection/runtime | **Partial / target gap** | Table/DataFrame conversion code exists, but there is not yet a canonical Foundation-IR→relational database projection with an agent-readable schema/catalog and stable cross-representation IDs. |
| Wiki/progressive-disclosure projection | **Planned** | The target agent-readable semantic/representation catalog is not yet a canonical implemented projection. |
| RDF/semantic-graph projection | **Planned / not canonical** | Not yet a maintained projection surface in current DIGIMON. |
| Specialized lexical/BM25 index | **Planned / selective** | Ordinary harness-native text/file search should not be rewrapped. A specialized lexical index remains a target only where it adds real retrieval capability. |
| Artifact/derivation graph | **Partial foundations / target gap** | Evidence IDs, producer metadata, manifests and invalidation exist, but there is not yet a first-class cross-representation graph of projection/retrieval/transformation executions and analytical findings. |
| MCP agent surface | **Implemented in code** | Stdio MCP exposes build, retrieval, inspection, composition and resource tools; it is the strongest current modern agent-facing surface. |
| Python public runtime | **Partial / target gap** | The target is a clean developer/application library over the maintained core; current code is less consolidated than the vision requires. |
| CLI | **Implemented / Transitional** | `digimon_cli.py` remains human-facing but currently instantiates legacy `PlanningAgent`/`AgentOrchestrator` and optional ReAct behavior rather than the maintained typed runtime directly. |
| Internal agent brain / old orchestrators / programmed AoT | **Legacy / Transitional** | Significant older planning/orchestration code remains and is still used by some entry points, but it is not the target control architecture. |
| Error/recovery semantics | **Partial** | Many fail-open cases were repaired, but one small uniform machine-actionable error taxonomy/envelope is not yet universal across all layers. |
| Evaluation framework | **Implemented / deferred priority** | Benchmark/evaluation code exists, but evaluation is not the current architecture driver. |
| Current-head runtime certification | **Not available yet** | Deterministic contracts have been added source-side, but the available environment cannot execute the repository and current connector-created commits/PRs have not triggered GitHub Actions. |

## Current architectural center

The strongest maintained path is now:

```text
typed capability records
        ↓
strict composition / reference plans
        ↓
current graph / VDB / community / matrix resources
        ↓
exact evidence recovery
        ↓
grounded answer or explicit evidence gap
```

That is an important current implementation center, but it is narrower than the full `Represent → Retrieve → Analyze` north star.

## Current public/control surfaces

### MCP

The stdio MCP server is currently the strongest modern agent-facing integration surface. It supports individual capabilities and maintained reference methods, plus older/direct helpers.

### CLI

The CLI is a real human-facing surface but still belongs to the older planner/orchestrator generation internally.

### Python/application surface

A clean consolidated public library over the same maintained runtime remains incomplete.

The target is **CLI + Python + MCP over one core**, not three independent architectures.

## Current representation reality

### Strong/current

- property graphs (ER/RK plus tree/passage variants);
- vector indexes;
- communities/sparse structures;
- graph/table/vector conversion experiments;
- exact source/chunk evidence representation.

### Partial or target-state only

- canonical governed-IR→relational database projection;
- first-class semantic/RDF projection;
- generated wiki/progressive-disclosure catalog describing semantic organization plus available representations/schemas/capabilities;
- specialized lexical/BM25 representation where native harness search is insufficient;
- unified analytical artifact/finding model across graph and non-graph methods.

## Current analytics reality

DIGIMON already has significant graph-focused analytical machinery reflecting its GraphRAG/SNA lineage. However, analytics are not yet documented/organized as fully as retrieval.

Current substantive examples include:

- PPR/diffusion scores;
- community detection/materialization;
- PCST optimization;
- Steiner-tree approximation;
- k-hop/path/subgraph operations;
- score aggregation/propagation;
- tree/community-derived structures.

The broader north-star suite—centrality families, cohesion/brokerage measures, general table/statistical methods, explicit model/finding outputs—remains a gap rather than a current guarantee.

## Current provenance reality

### Implemented foundations

- entity/relationship source IDs;
- exact chunk/evidence IDs;
- evidence provenance carried into answers;
- composer step metadata;
- graph/source manifests;
- practical dependent-artifact invalidation;
- source-aware structural materialization;
- community source lineage.

### Missing target layer

The system does not yet have one explicit derivation graph recording, for every important artifact:

```text
input artifact/version
→ projection/retrieval/analytic execution + parameters
→ output artifact/version
→ finding
```

That target is broader than answer citations and should not be confused with the domain/property graph.

## Important remaining implementation gaps

1. **Fresh runtime execution** — run the deterministic core suite and E2E canaries on the current head; fix real failures before more speculative refactors.
2. **Governed IR projection seam** — make the onto-canon/Foundation IR path the clear canonical ecosystem input and verify what semantic fields each target projection requires.
3. **Representation plane expansion** — relational database projection, wiki/catalog projection, and other high-value text-derived representations are not yet first-class maintained outputs.
4. **Analytics as a first-class plane** — expose/describe a coherent analytic method taxonomy and typed outputs, not only retrieval/reference methods.
5. **Artifact/derivation lineage** — record projection/retrieval/transformation executions and outputs, extending the current evidence/manifests/invalidation foundations.
6. **Cross-graph sparse resources** — sparse matrices are still effectively dataset/ER scoped and can theoretically be same-shaped but semantically stale for another active graph.
7. **Public surfaces convergence** — modernize CLI/Python surfaces onto the same maintained core used by agent-facing capabilities.
8. **Error/result uniformity** — finish a small machine-actionable convention across maintained layers.
9. **Legacy isolation** — continue classifying/deprecating old internal brain/orchestrator/AoT/MCP paths without deleting useful lineage prematurely.

## Verification boundary

No current documentation should claim that the present head has passed the full deterministic suite or clean-rebuild canary until a real runner executes it.

Historical CI evidence showed dependency installation failing before meaningful tests due the obsolete `umap==0.1.1` dependency. That dependency has been removed, but current commits have not produced a fresh GitHub Actions signal through the available integration.

Continue with:

- [VISION.md](VISION.md) — full north star;
- [ARCHITECTURE.md](ARCHITECTURE.md) — target system design;
- [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md) — module-level reality;
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — current → target gaps;
- [ROADMAP.md](ROADMAP.md) — closure order;
- [DOCUMENTATION_COVERAGE.md](DOCUMENTATION_COVERAGE.md) — reconciliation checklist.
