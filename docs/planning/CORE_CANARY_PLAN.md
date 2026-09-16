# Core Canary Plan

**Status:** Active  
**Priority:** P0  
**Planning level:** Execution stub  
**Updated:** 2026-09-16  
**Owner:** DIGIMON maintainers

## Goal

Establish whether the existing canonical MCP path is reproducibly working from a clean environment. Do **not** redesign the path unless the canary exposes a real failure.

Target path:

```text
source documents
  -> corpus preparation
  -> ER graph build/load
  -> entity VDB build/load
  -> entity search
  -> relationship/chunk retrieval
  -> grounded answer
```

## Current evidence

- `tests/e2e/test_mcp_smoke.py` already exercises initialization, ER graph load/build, entity VDB, resource discovery, entity search, one-hop relationship/chunk retrieval, answer generation, and `basic_local` execution.
- Historical runtime logs show graph/VDB/community loading and real retrieval succeeding on earlier builds.
- The current smoke test calls MCP tool functions directly rather than exercising the stdio protocol and normally reuses the pre-built `Fictional_Test` artifacts.

## In scope

1. Make the existing smoke test portable and explicit about cached-vs-rebuild mode.
2. Add a clean-build mode that starts from the small fictional source corpus when credentials are available.
3. Keep a fast cached mode for local/CI smoke coverage.
4. Produce a clear PASS/FAIL summary with the failing step.

## Non-goals

- No new resource-lifecycle framework.
- No new planner/orchestrator.
- No benchmark optimization.
- No attempt to certify every graph type or retrieval method in this checkpoint.
- No formal reasoning DAG.

## Checkpoint 1 — Portable cached smoke

**Change:** ensure `tests/e2e/test_mcp_smoke.py` uses repository-relative paths and current MCP wrappers only.

**Test:**

```bash
python tests/e2e/test_mcp_smoke.py
```

**Success criteria:**

- initializes DIGIMON;
- registers/loads `Fictional_Test_ERGraph`;
- builds/loads entity VDB;
- entity search returns results;
- relationship and chunk retrieval return evidence;
- answer generation returns a non-empty answer;
- `basic_local` executes without an unhandled exception.

## Checkpoint 2 — Clean-build mode

**Change:** add an environment/CLI switch that forces the canary to prepare corpus/build the ER graph from `Data/Fictional_Test` instead of relying on existing artifacts.

**Test:**

```bash
DIGIMON_CANARY_REBUILD=1 python tests/e2e/test_mcp_smoke.py
```

**Success criteria:** the same assertions pass after rebuilding the corpus/graph/index path.

## Checkpoint 3 — Protocol smoke, only if needed

If direct tool-function smoke passes but real MCP clients fail, add one small stdio protocol test. Do not add it preemptively.

## Stop / branch rule

- If a checkpoint passes, record it and move on.
- If it fails, create the smallest fix for the observed failure and rerun.
- Do not introduce a general abstraction unless at least one observed failure requires it.

## Done when

The cached canary passes reliably, and the clean-build canary passes in an environment with valid provider credentials.