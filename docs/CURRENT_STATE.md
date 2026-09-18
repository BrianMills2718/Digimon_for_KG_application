# DIGIMON Current State

**Latest execution checkpoint:** [Foundation graph runtime, Batch 02](reports/PROJECTION_BATCH_02.md).  
**Repository:** `BrianMills2718/Digimon_for_KG_application`.  
**Full vision:** [VISION.md](VISION.md), not only the latest implementation batch.

## Executive summary

DIGIMON is a text-derived **representation, retrieval, and analytics** runtime downstream of onto-canon6's governed semantic IR. The recent increment makes a concrete part of that vision executable: saved IR → SQLite/graphs → maintained graph retrieval → exact source evidence, with shared identity and local execution lineage.

**69 selected tests passed, zero skipped**, in the final recorded Batch 2 run. This is scoped execution on verified source bytes, **not** the full repository suite, clean dependency installation, live-provider path, MCP canaries, or autonomous-harness observation. Earlier blanket statements that no new projection tests had run are superseded by the two receipts below.

The codebase remains transitional. The broader graph/vector/reference-method inventory below carries forward the previous source review; it was not all re-executed in this batch. The detailed historical source inventory is retained in Git history at the parent revision `2b2a35322187951008759cb33ec127c055076225`.

## Evidence vocabulary

Keep **source present**, **focused checks passed**, **intended consumer observed**, and **full outcome observed** separate. No single “implemented” label implies all four. Synthetic tests do not establish production-corpus quality, semantic retrieval, or agent usefulness.

## Newly executed governed-input path

| Capability | Current evidence | Remaining boundary |
|---|---|---|
| Foundation IR 1.3 + passage 1.0 consumer | Focused import/identity/field/source-scope tests pass; actual file hashes checked | Real producer/corpus export integration remains unobserved |
| Shared identity | Canonical entity/assertion/passage keys survive tested SQL/graph moves; same labels do not merge IDs | Vector/catalog identity still needs integration |
| SQLite projection | Normalized roles/qualifiers/provenance and original payload; exact entity-to-passage joins, safe overwrite, reopen tested | Broader relational/analytical workflows and other engines not certified |
| Assertion graph | Tested supported-field reconstruction after GraphML save/reload, including n-ary roles | Not universal fidelity certification for every possible input/edit |
| Binary MultiGraph | Parallel assertion IDs retained; n-ary assertions explicitly skipped | It is a lossy declared projection, not a universal semantic graph |
| Saved project | Input/artifact hashes, safe generation publication, stale detection, fresh-process reopening tested | No production lifecycle service or scaling claim |
| Maintained graph adoption | `GraphRuntime.py` binds actual `NetworkXStorage` and `OperatorContext`; existing khop/materialize functions execute | Global GraphRAGContext/MCP registration and all reference methods not certified on this input |
| Bounded graph working set | One/two/three hops, filters before aggregation, isolates/self-loops, typed SUBGRAPH and attributed `nx_graph` tested | General analytics plane not yet integrated |
| Graph evidence | Exact passage text/source scope; selected-edge evidence only; partial/missing evidence explicit | Grounded answer generation and actual agent findings not exercised here |
| Execution/derivation foundations | Runtime view → subgraph → evidence each has exact artifact/producer lineage; injected failure/recovery tested | Full derivation query/invalidation graph and findings/model lineage still incomplete |
| Local Python/CLI bridge | `FoundationProject.graph_neighborhood`, typed helper, and `--graph-hops` demo build/reuse executed | Legacy main CLI modernization and parity across Python/CLI/MCP remain |

The retrieval adapter uses **undirected binary assertion associations**, unit weight per entity pair, retained self-loops, explicit predicate selection, and all parallel assertion records. It does not reinterpret polarity as affirmative truth, assign confidence as tie strength, or infer causal/social influence. SQL still exposes claims intentionally absent from the binary network.

## Existing wider core — source inventory, not newly certified

| Area | Carried-forward reality |
|---|---|
| Typed operators/composition | QUERY_TEXT, ENTITY_SET, RELATIONSHIP_SET, CHUNK_SET, SUBGRAPH, COMMUNITY_SET, SCORE_VECTOR; strict named wiring/execution and reference-plan machinery |
| Reference methods | basic_local, basic_global, lightrag, fastgraphrag, hipporag, tog, gr, dalk, kgp, med; prior source-level semantic repairs remain, not all freshly executed |
| Standalone ingestion/build | Raw corpus preparation/chunking; ER/RK/tree/balanced-tree/passage builders, source manifests and practical invalidation |
| Vector/index machinery | FAISS/entity/relationship indexes and prior scoring/identity/upsert repairs; **not yet integrated with the new Foundation project** |
| Retrieval and structural transforms | Entity/relation/chunk, PPR/diffusion, path/subgraph, PCST/Steiner, community operations and source materialization |
| Grounded synthesis | Evidence IDs/citation validation and empty-evidence safeguards exist; no fresh live-provider certification in this tranche |
| Resource context | Active dataset/graph, canonical VDB selection, pragmatic invalidation; same-shaped cross-graph sparse-resource identity remains a gap |
| Current interfaces | MCP exists; legacy CLI/API still couple to older orchestration; the new local projection bridge is not their complete replacement |
| Legacy/research | Older AgentBrain/AOT/orchestrator/memory/MCP and evaluation code remain; retain useful capabilities, isolate ambiguity, do not build another brain |

Public operator exports now load lazily so direct graph access does not eagerly initialize unrelated LLM/vector dependencies. Existing names are retained; complete import/registry compatibility across all optional stacks remains a broader regression obligation.

## Full north star and remaining gaps

The target still includes relational, vector, property graph, semantic/RDF where useful, hierarchy/tree, specialized lexical retrieval beyond native search, source evidence, and an agent-readable wiki/catalog. Geospatial is outside this text-focused scope.

The wiki is a **progressive-disclosure map of content and the operational environment**: entities/topics/sources, actual representations, schemas/ontology references, canonical join keys, capability availability and evidence locations. It is not merely pages about entities and not a reason to duplicate native file/search/wiki tools.

Analytics remain first-class: retrieve a working set → transform/analyze → reuse typed outputs → recover evidence and form a finding. The new attributed SUBGRAPH is an input to that work, not completion of Leiden/centrality or the broader statistical suite.

The external harness still owns strategy, planning, composition, adaptation and stopping. DIGIMON owns representations, specialized retrieval/analytics, contracts, and observed lineage. CLI/Python/MCP should converge over that capability core without imposing another planner.

Evidence provenance, semantic provenance and artifact/execution derivation are distinct. Source-backed assertions, retrieved sets, computed scores/communities and interpretations must not be conflated. The derivation graph is not the domain graph. Local traces/artifact references are an implemented foundation, not completion of the whole provenance architecture.

## Ordered frontier

Follow [the living plan](planning/NORTH_STAR_VERTICAL_SLICE_PLAN.md), revision 3. Batches 0–2 have bounded execution evidence. **Next is real vectors from the same IR through the existing provider/index seam**, including query and persisted reload. Then progressive-disclosure catalog, first-class analytic access and source recovery, integrated harness finding, and broader hardening/representation coverage.

FAISS, llama-index and provider packages were absent from the scoped Batch 2 runner. Recheck authorized execution/configuration; do not promote a document list or hash embedding as semantic vector retrieval. Do not accumulate dependent unexecuted code.

## Receipts and limitations

- [Batch 01](reports/PROJECTION_BATCH_01.md): baseline repairs and saved/reopened project, 45 selected tests.
- [Batch 02](reports/PROJECTION_BATCH_02.md): maintained graph storage/operator adoption and exact evidence, 69 selected cumulative tests.

Both used a byte-verified scoped snapshot because direct Git/DNS access was unavailable. Normal repository pytest configuration/root fixtures were retained; no wholesale test bypass or fake provider execution. These do not certify all tests/core, dependency installation, other runtimes, CI, MCP, real production corpora, vectors, analytics or external-harness composition. Historical CI observations are not current green signals.

[IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md), [GAP_ANALYSIS.md](GAP_ANALYSIS.md), [ROADMAP.md](ROADMAP.md), and [DOCUMENTATION_COVERAGE.md](DOCUMENTATION_COVERAGE.md) retain the broader context. Where an older source-only status conflicts with the bounded receipts above, use the newer receipt for that exact tested scope, not as evidence for unrelated capabilities.
