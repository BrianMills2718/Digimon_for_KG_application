# Failure-Driven Core Fixes Plan

**Status:** Active — implementation underway; first fresh run pending  
**Priority:** P0/P1  
**Planning level:** Execution stub  
**Updated:** 2026-09-16  
**Owner:** DIGIMON maintainers

## Goal

Fix fundamental defects exposed by the canary and CI before adding architecture ceremony. Let observed failures determine the next code change.

## Operating rule

For each failure, record only:

| Failure | Evidence | Smallest fix | Regression check | Status |
|---|---|---|---|---|
| Full requirements cannot resolve `umap==0.1.1` | Existing GitHub Actions install log | Remove invalid `umap` pin; keep `umap-learn` | Full dependency install reaches tests | **Implemented; rerun pending** |
| CI fails on style/research bootstrap before product signal | Existing workflow run | Replace first gate with deterministic maintained-core contracts; make style advisory | `pytest tests/core -q` in CI | **Implemented; rerun pending** |
| Clean checkout has no `Option/Config2.yaml`, while MCP loads it directly | Repository contents + startup code | `Config.from_yaml_file()` falls back to checked-in/default config resolution | MCP initialization from checkout without local YAML | **Implemented; canary run pending** |
| Preferred MCP path dependencies missing from minimal install | Import trace vs `requirements-minimal.txt` | Add MCP SDK 1.x, `igraph`, `lazy-object-proxy`, CLI color dependency | Minimal install can import/start core MCP path | **Implemented; install run pending** |
| Optional embedding backends imported eagerly | `EmbeddingFactory.py` | Lazy-load Ollama/HF providers only when selected | OpenAI/default startup does not require optional embedding packages | **Implemented; run pending** |
| VDB registration log checks nonexistent `_vdbs` | `GraphRAGContext` uses public `vdbs` / `list_vdbs()` | Log through `list_vdbs()` | VDB build log reflects registered ID | **Implemented** |
| Expanded entity search uses `_replace()` on third-party score object | `entity_vdb_search_tool` source | Carry adjusted score as a plain scalar | Expansion branch returns ranked results without object mutation | **Implemented; canary run pending** |

## Priority order

1. clean installation / import failures;
2. canonical MCP initialization failures;
3. corpus/ER graph build or load failures;
4. entity VDB build/search failures;
5. relationship/chunk/source retrieval failures;
6. answer-generation/grounding failures;
7. incorrect failure swallowing or misleading success results;
8. only then broader composition/resource/provenance cleanup required by observed behavior.

## Current execution state

The repository now has:

- a portable cached MCP canary;
- a separate clean-rebuild canary mode;
- a small deterministic composition contract suite;
- a lean blocking CI core job plus advisory style job;
- `workflow_dispatch` declared in CI;
- bootstrap/dependency fixes above.

GitHub Actions has not created a run for the connector-generated commits, and the available connector does not expose workflow dispatch. The connected development machine is also temporarily unavailable through its automation tool quota. Therefore the next meaningful evidence is the **first actual run**, not more planning.

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

Create/generalize an abstraction only when multiple observed failures share the same underlying cause and the abstraction removes real duplication or inconsistency.

## Done when

The canonical canary and maintained-core CI gate are green enough that the next failures come from breadth/features rather than bootstrap or fundamental path correctness.