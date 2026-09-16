# CI Recovery Plan

**Status:** Active  
**Priority:** P0  
**Planning level:** Execution stub  
**Updated:** 2026-09-16  
**Owner:** DIGIMON maintainers

## Goal

Restore CI as a useful product-health signal with the smallest reliable gate. CI should answer **"does the maintained core install and run its deterministic checks?"** before it tries to certify research/UI/package surfaces.

## Current failure evidence

Latest inspected `main` CI run failed before tests:

- `requirements.txt` contains `umap==0.1.1`; pip reports no compatible distribution.
- `umap-learn==0.5.7` is also present and supplies the normal `umap` Python package.
- Black currently blocks the workflow before product tests even though the repository contains large historical/legacy areas.
- test jobs install the entire research requirements set before running any tests.
- package/Docker jobs are downstream of a workflow that currently never reaches meaningful product verification.

## In scope

1. Remove the known invalid dependency pin.
2. Make the first CI gate use the maintained/core dependency path.
3. Run blocking syntax/import/core contract checks before broad research tests.
4. Keep style checks advisory until the maintained scope is formatted consistently.
5. Add `workflow_dispatch` so the current branch can be verified on demand.
6. Expand coverage only after the core gate is green.

## Non-goals

- No repository-wide formatting campaign.
- No dependency-platform redesign.
- No requirement to make every experimental test green immediately.
- No Docker/package-release ceremony before core verification works.

## Checkpoint 1 — Installation reaches tests

**Changes:**

- remove `umap==0.1.1` from `requirements.txt`;
- make core CI install from `requirements-minimal.txt` plus explicit test tooling;
- use current GitHub Actions versions.

**Success criteria:** dependency installation succeeds and CI reaches Python checks.

## Checkpoint 2 — Small blocking core gate

**Blocking checks:**

```bash
python -m compileall Core digimon_mcp_stdio_server.py
pytest <small deterministic maintained-core selection>
```

Select tests based on the current operator/composition/MCP core, not legacy memory/orchestrator tests merely because they live under `tests/unit/`.

**Success criteria:** failures reflect maintained product code rather than environment/bootstrap problems.

## Checkpoint 3 — Add the canary appropriately

- cached/no-network canary may become blocking if it is deterministic;
- credentialed clean-build canary stays manual/scheduled/non-blocking until stable.

## Checkpoint 4 — Expand only when useful

After the core gate is green, decide separately whether to restore broader integration, package, Docker, formatting, typing, and live-provider checks.

## Stop / branch rule

Fix the first concrete red failure and rerun. Do not add CI layers to compensate for failures that have not occurred.

## Done when

A normal or manually triggered workflow can install the maintained core and produce a meaningful green/red result for deterministic core behavior.