# Failure-Driven Core Fixes Plan

**Status:** Active — major source-level defects fixed; first fresh run pending  
**Priority:** P0/P1  
**Planning level:** Execution stub  
**Updated:** 2026-09-16  
**Owner:** DIGIMON maintainers

## Goal

Fix fundamental defects exposed by the canary, prior CI evidence, and direct source tracing before adding architecture ceremony. Let observed failures determine the next code change.

## Operating rule

Concrete failure → smallest fix → regression check → rerun. Generalize only when the same missing boundary breaks multiple real paths.

## Implemented fixes

| Failure | Smallest fix | Regression signal |
|---|---|---|
| Full requirements cannot resolve `umap==0.1.1` | Remove invalid pin; keep `umap-learn` | dependency install reaches tests |
| CI/style/research bootstrap hides product signal | Minimal-core blocking job; style advisory | CI core job |
| Fresh checkout lacks private `Config2.yaml` | Fall back to checked-in/default config | MCP bootstrap/canary |
| Example API-key placeholders override environment credentials | Normalize placeholders; omit empty provider key | credential/factory tests |
| Minimal MCP path imports undeclared dependencies | Declare direct MCP/core dependencies | minimal install + MCP import |
| Optional graph/embedding backends import eagerly | Lazy-load selected implementations | ER/OpenAI startup path |
| Fixture-specific query expansion contaminates generic retrieval | Replace fixture synonyms with generic expansion | query-expansion tests |
| ER extraction can return success with a zero-node graph | Treat non-empty corpus → zero nodes as build failure | ER graph build contract |
| Other graph builders can report success after internal failure | Honor `build_graph()` result and return graph instance | graph-build contracts |
| Passage graph skips fresh/small datasets and loads global checkpoints | Remove hard-coded resume/checkpoint behavior; bounded direct build | graph-build contract + canary pending |
| VDB build/persistence can look successful while unusable | Boolean index-build contract; refuse registration on failure | index-build tests |
| FAISS assumes 1024/provider-declared dimensions | Infer dimension from returned vectors | FAISS dimension tests |
| FAISS L2 distance treated as higher-is-better similarity | Normalize L2 distance to higher-is-better score | FAISS score tests |
| Entity linking discards match strength | Preserve normalized VDB score and support threshold | entity-link tests |
| Relationship VDB embeds stale `type` field instead of DIGIMON edge semantics | Map legacy default to `relation_name`/`keywords`/`description`; preserve endpoint metadata | relationship embedding/VDB tests |
| Entity/relationship VDB wrappers drift from backend API | Use real retrieval API, exact IDs, truthful build result | relationship VDB/operator tests |
| Ambiguous dataset names can bind another dataset's resources | Exact-dataset-first graph ordering + active-dataset VDB priority | resource-selection tests |
| Base graph assumes embedding model is a tokenizer | Use tiktoken helpers | graph-tokenization tests |
| PPR reset/damping semantics inverted | 0.15 teleport → 0.85 damping; align typed/MCP/config paths | PPR tests |
| PPR/relationship/chunk propagation returns wrong order or loses provenance | Descending ranking, finite normalization, preserve real chunk IDs | score-propagation tests |
| ToG rereads stale seed entities and effectively runs `depth + 1` | Explicit dependency-aware hops + relation→entity adapter | ToG + method-plan tests |
| KGP rereads stale state; TF-IDF score is candidate index | Evidence-guided hops + direct cosine TF-IDF | KGP/TF-IDF tests |
| K-hop path parsing can fabricate adjacency between concatenated paths | Normalize edge records and split disconnected path segments | subgraph tests |
| `steiner_tree` is an un-awaited induced subgraph, not a Steiner tree | Use NetworkX Steiner approximation | subgraph tests |
| PCST ignores retrieval relevance and gives unseen endpoints implicit prize | Entity scores → prizes, relationship scores → costs, unseen endpoints → zero prize | PCST tests |
| GR/DALK/Med compute structural subgraphs but ignore them for evidence | Add `subgraph.materialize` and rewire methods | subgraph + method-plan tests |
| Structural materialization can stringify opaque runtime objects as evidence | Emit only resolvable real text/content | fail-closed subgraph grounding test |
| Community clustering calls nonexistent `logger.start()` | Use ordinary lifecycle logging; align abstract loader signature | community lifecycle test |
| Persisted Leiden reports lose ID/level/occurrence on read | Pair report text/rating with authoritative community schema | community metadata tests |
| Basic Global stops at community reports | Add `community.materialize` and answer synthesis | community/method-plan tests |
| Direct chunk tools fabricate placeholder text or fuzzy-match missing source IDs | Exact stored source references only | direct chunk grounding tests |
| Named methods inconsistently stop at chunks | All 10 terminate in `meta.generate_answer`; context-only strips it | method-plan tests |
| Answer generation can hallucinate after empty retrieval | Do not call LLM without evidence | grounded-answer tests |
| Retrieved source IDs disappear before answer synthesis | Label evidence as `[chunk_id]`; preserve IDs in answer metadata | grounded-answer tests |
| Missing/invented answer citations are invisible | Validate cited IDs and expose citation status | citation-validation tests + canary |
| Method execution drops slot provenance metadata | Add `all_step_metadata` / `final_metadata` | composer metadata test |
| `return_context_only` turns dataclass records into repr strings | Recursive JSON-safe structural serialization | context transport test |
| Invalid composed plans execute implicitly | Reject failed static validation by default | composition tests |
| Static validator accepts wrong `plan_inputs.*` types | Validate explicit plan-input kind | plan-input validation test |
| Static validator invents types for nonexistent named outputs | Reject unknown output keys | named-output validation test |
| Canary accepts fluent but ungrounded named-method answers | Require real evidence + valid evidence citations | MCP canary |

## Current execution state

The maintained core now has deterministic contracts covering composition, transport, credentials, resource scoping, graph builds, FAISS dimensions/scores, entity/relationship retrieval, PPR, TF-IDF, score propagation, subgraphs/PCST, communities, exact source grounding, answer provenance/citation validation, and all ten reference plans.

Runtime verification remains distinct from implementation status. GitHub currently reports **zero Actions runs** for this fork/branch and connector-generated commits have no attached checks. The repository is a fork, so a one-time GitHub-side Actions enablement is a plausible cause, but that has not been verified through an Actions-permissions endpoint. The connected tooling also cannot dispatch a workflow or run a fresh checkout here.

## Remaining work before broader architecture work

1. run `pytest tests/core -q` in a clean minimal environment;
2. run the cached MCP canary;
3. run `DIGIMON_CANARY_REBUILD=1` with provider credentials;
4. fix the first observed red failure with the smallest patch;
5. only after those are green, smoke optional breadth paths such as live Leiden community generation and the external WAT-backed PassageGraph.

## Non-goals

- No speculative framework rewrites.
- No universal resource/provenance ontology unless repeated observed failures require it.
- No cleanup of every legacy module before the maintained core works.
- No benchmark or novelty work in this plan.

## Escalation rule

Generalize only after repeated concrete failures reveal the same missing boundary. `subgraph.materialize` is the current example: one small adapter fixed the same evidence-boundary defect in GR, DALK, and Med.

## Done when

The canonical canary and maintained-core gate are green enough that the next failures come from breadth/features rather than bootstrap, false-success states, or core retrieval/grounding correctness.
