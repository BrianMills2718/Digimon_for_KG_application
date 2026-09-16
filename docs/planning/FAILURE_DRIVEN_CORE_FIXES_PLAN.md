# Failure-Driven Core Fixes Plan

**Status:** Active — major source-level defects fixed; first fresh run pending  
**Priority:** P0/P1  
**Planning level:** Execution stub  
**Updated:** 2026-09-16  
**Owner:** DIGIMON maintainers

## Goal

Fix fundamental defects exposed by the canary, CI history, and direct source tracing before adding architecture ceremony. Let observed failures determine the next code change.

## Operating rule

Record concrete failure → smallest fix → regression check → rerun. Generalize only when the same missing boundary breaks multiple real paths.

## Implemented fixes

| Failure | Smallest fix | Regression signal |
|---|---|---|
| Full requirements cannot resolve `umap==0.1.1` | Remove invalid pin; keep `umap-learn` | dependency install reaches tests |
| CI/style/research bootstrap hides product signal | Minimal-core blocking job; style advisory | CI core job |
| Fresh checkout lacks private `Config2.yaml` | Fallback to checked-in/default config | MCP bootstrap test/canary |
| Example API-key placeholders override env credentials | Normalize placeholders; omit empty embedding key | credential/factory tests |
| Minimal MCP path imports undeclared dependencies | Declare MCP/core direct dependencies | minimal install + MCP import |
| Optional graph/embedding backends import eagerly | Lazy-load selected implementations | ER/OpenAI startup path |
| Fixture-specific query expansion contaminates generic retrieval | Replace Fictional-Test synonyms with generic expansion | query-expansion tests |
| Graph builds can report success after internal failure | Honor `build_graph()` result for all graph types; return graph instance on success | graph-build contract tests |
| Passage graph skips fresh/small datasets and loads global checkpoints | Remove cold-start/checkpoint shortcuts; bounded dataset-local async build | graph-build contract + canary pending |
| VDB build/persistence can look successful while unusable | Boolean index build contract; refuse registration on failure | index-build tests |
| FAISS assumes 1024/provider-declared dimensions | Infer from actual vectors | FAISS dimension tests |
| FAISS L2 distance treated as higher-is-better similarity | Normalize L2 distance to monotonic higher-is-better score | FAISS score-direction tests |
| Entity/relationship VDB wrappers drift from backend API | Use real retrieval API, exact IDs, truthful build result | relationship VDB/operator tests |
| Multi-dataset method context can choose another dataset's VDB | Dataset-aware resource priority in `GraphRAGContext` | context-resource selection tests |
| Base graph assumes embedding model is a tokenizer | Use tiktoken helpers for graph summarization | graph-tokenization tests |
| PPR reset/damping semantics inverted | 0.15 teleport → 0.85 damping; align typed/MCP paths and config default | PPR tests |
| PPR/relationship/chunk propagation returns wrong order or loses provenance | Sort descending, finite normalization, preserve real chunk IDs | score-propagation tests |
| ToG loop rereads stale seed entities and runs `depth + 1` times | Explicit dependency-aware hop unrolling; typed relation→entity adapter | ToG tests + method-plan validation |
| KGP loop rereads stale state; TF-IDF score is candidate index | Evidence-guided hop unrolling; direct sklearn cosine scores | KGP/TF-IDF tests |
| k-hop path operator treats edge dicts as node IDs | Normalize storage edge records into node paths/edges | subgraph tests |
| `steiner_tree` is an un-awaited induced subgraph, not a Steiner tree | Use NetworkX Steiner approximation | subgraph tests |
| GR/DALK/Med compute structural subgraphs but ignore them for evidence | Add one `subgraph.materialize` bridge to entities/source chunks and rewire methods | subgraph + method-plan tests |
| Basic Global stops at community reports | Add `community.materialize` and answer synthesis | community contract tests |
| Named methods inconsistently stop at chunks | All 10 reference methods terminate in `meta.generate_answer`; context-only strips it | method-plan contract tests |
| Answer generation can hallucinate after empty retrieval | Do not call LLM without evidence; explicit insufficient-evidence result | grounded-answer tests |
| Invalid composed plans execute implicitly | Fail static validation by default; explicit best-effort opt-in only | composition contract tests |
| Canary accepts fabricated/empty success signals | No fallback evidence; require real retrieval and real named-method answer | MCP canary |

## Current execution state

The repository now has:

- cached and clean-rebuild MCP canary modes;
- a minimal-install/import CI gate plus deterministic `tests/core` contract suite;
- contract coverage for composition, credentials, resource scoping, graph builds, FAISS dimensions/scores, PPR, TF-IDF, score propagation, subgraphs, community/subgraph evidence bridges, grounded answers, and all ten reference method plans;
- all maintained reference methods wired so each structural/retrieval step changes downstream evidence rather than being decorative.

Runtime verification is still distinct from implementation status. GitHub reports **zero Actions runs** for the current fork/branch, connector commits have no attached status checks, workflow dispatch is not exposed through the connected GitHub tool, and this execution environment cannot reach GitHub for a fresh clone. This now looks like a one-time GitHub Actions enablement/execution problem, not a reason to add more CI architecture.

## Priority order from here

1. run `pytest tests/core -q` in a clean minimal environment;
2. run the cached MCP canary;
3. run `DIGIMON_CANARY_REBUILD=1` with provider credentials;
4. fix the first observed red failure with the smallest patch;
5. only after those are green, broaden to optional graph/method surfaces.

## Non-goals

- No speculative framework rewrites.
- No universal resource/provenance ontology unless repeated observed failures require it.
- No cleanup of every legacy module before the maintained core works.
- No benchmark or novelty work in this plan.

## Escalation rule

Generalize an abstraction only when multiple observed failures share the same underlying cause and the abstraction removes real duplication or inconsistency. `subgraph.materialize` is the current example: one small bridge fixed the same concrete evidence-boundary failure in GR, DALK, and Med.

## Done when

The canonical canary and maintained-core CI gate are green enough that the next failures come from breadth/features rather than bootstrap or fundamental path correctness.
