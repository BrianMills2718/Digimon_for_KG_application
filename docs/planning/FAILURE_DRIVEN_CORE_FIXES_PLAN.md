# Failure-Driven Core Fixes Plan

**Status:** Active — implementation underway; first fresh run pending  
**Priority:** P0/P1  
**Planning level:** Execution stub  
**Updated:** 2026-09-16  
**Owner:** DIGIMON maintainers

## Goal

Fix fundamental defects exposed by the canary, CI history, and direct source tracing before adding architecture ceremony. Let observed failures determine the next code change.

## Operating rule

For each failure, record only:

| Failure | Evidence | Smallest fix | Regression check | Status |
|---|---|---|---|---|
| Full requirements cannot resolve `umap==0.1.1` | Existing GitHub Actions install log | Remove invalid `umap` pin; keep `umap-learn` | Full dependency install reaches tests | **Implemented; rerun pending** |
| CI fails on style/research bootstrap before product signal | Existing workflow run | Blocking job installs minimal core, imports MCP server, runs core tests; style advisory | CI core job | **Implemented; rerun pending** |
| Clean checkout has no `Option/Config2.yaml`, while MCP loads it directly | Repository contents + startup code | Fall back to checked-in/default config resolution | MCP initialization without local YAML | **Implemented; canary pending** |
| Example config API-key placeholders can override normal env credentials | Example config + validators | Normalize placeholder keys to unset for LLM/embedding providers | `test_config_credentials.py` | **Implemented** |
| Preferred MCP path dependencies missing from minimal install | Direct import trace | Declare MCP SDK 1.x and direct core dependencies | Minimal install + MCP import | **Implemented; install pending** |
| Optional graph/embedding implementations load unrelated dependencies eagerly | `GraphFactory.py`, `EmbeddingFactory.py` | Lazy-load selected graph/embedding backends | ER/OpenAI path does not require tree/Ollama/HF dependencies | **Implemented; import pending** |
| Generic entity search contains Fictional-Test-specific synonyms | `query_expansion.py` | Replace fixture knowledge with corpus-agnostic variants | `test_query_expansion.py` | **Implemented** |
| Entity VDB logs inspect nonexistent `_vdbs` | `GraphRAGContext` API | Use `list_vdbs()` | VDB registration log reports actual ID | **Implemented** |
| Entity VDB build can report success when index setup failed | `BaseIndex` + VDB build source | Make index build return real bool; refuse registration/success on failure | `test_index_build_contract.py` | **Implemented; canary pending** |
| Index persistence can create a directory yet leave unusable index | Base build/storage contract | Require live index after persistence as well as persisted path | `test_index_build_contract.py` | **Implemented** |
| FAISS fresh build assumes 1024/configured dimensions | `FaissIndex.py` | Infer dimension from actual returned embeddings | `test_faiss_dimension_contract.py` | **Implemented; canary pending** |
| Base graph uses embedding provider as tokenizer | `BaseGraph._handle_entity_relation_summary` | Use DIGIMON tiktoken helpers | `test_graph_summary_tokenization.py` | **Implemented** |
| Expanded entity search mutates third-party score objects with `_replace()` | Entity search source | Carry adjusted score as scalar | Query-expansion/core tests | **Implemented; canary pending** |
| Relationship VDB build ignores index failure and mutates requested VDB ID | `relationship_tools.py` | Honor build result and exact requested collection ID | Relationship VDB contract tests | **Implemented** |
| Relationship VDB search calls nonexistent FAISS `search*` methods | `relationship_tools.py` vs `BaseIndex` API | Use `retrieval()`; report embedding-only mode unsupported | `test_relationship_vdb_contract.py` | **Implemented** |
| Relationship operator treats `(edges, scores)` as edge list | `relationship.vdb` + VDB result contract | Unpack tuple, preserve scores; accept old/new endpoint metadata | `test_relationship_operator_contract.py` | **Implemented** |
| Invalid composed plans execute implicitly | `OperatorComposer.execute()` | Reject failed static validation by default; explicit debug opt-in only | `test_composition_contract.py` | **Implemented** |
| Canary could pass with fabricated fallback evidence or empty method envelope | Previous canary logic | Require retrieved chunks; no invented fallback; named method must return evidence | `tests/e2e/test_mcp_smoke.py` | **Implemented; canary pending** |
| MCP method context can select another dataset's first matching VDB in a multi-dataset process | `_build_operator_context_for_dataset` source | Scope VDB selection by requested dataset ID | Add multi-dataset context test | **Observed; deferred until safe patch/run** |

## Priority order

1. clean installation / import failures;
2. canonical MCP initialization failures;
3. corpus/ER graph build or load failures;
4. entity VDB build/search failures;
5. relationship/chunk/source retrieval failures;
6. answer-generation/grounding failures;
7. incorrect failure swallowing or misleading success results;
8. only then broader cleanup required by observed behavior.

## Current execution state

The repository now has:

- a portable cached MCP canary;
- a separate clean-rebuild canary mode;
- deterministic core tests for composition, credentials, query expansion, index build semantics, FAISS dimensions, graph tokenization, and relationship VDB/operator contracts;
- a blocking CI job that installs `requirements-minimal.txt`, imports the MCP server, compiles the maintained core, and runs `pytest tests/core -q`;
- advisory style checking rather than style blocking product signal.

GitHub Actions has not created runs for connector-generated commits, the GitHub connector does not expose workflow dispatch, and the connected development machine is temporarily unavailable through its execution quota. The local sandbox also cannot materialize the GitHub archive through the allowed project connector path. Therefore implementation status and runtime verification are intentionally distinct: **the patches exist; the first fresh run is still pending**.

## Non-goals

- No speculative framework rewrites.
- No universal resource/provenance ontology unless a concrete failure demands it.
- No cleanup of every legacy module before the core works.
- No benchmark or novelty work in this plan.

## Checkpoint loop

For every newly observed red failure:

1. reproduce it;
2. identify the narrowest cause;
3. patch that cause;
4. add or tighten one regression check;
5. rerun;
6. continue only if still red.

## Escalation rule

Generalize an abstraction only when multiple observed failures share the same underlying cause and the abstraction removes real duplication or inconsistency.

## Done when

The canonical canary and maintained-core CI gate are green enough that the next failures come from breadth/features rather than bootstrap or fundamental path correctness.
