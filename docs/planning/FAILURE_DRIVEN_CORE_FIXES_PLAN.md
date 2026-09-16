# Failure-Driven Core Fixes Plan

**Status:** Active  
**Priority:** P0/P1  
**Planning level:** Execution stub  
**Updated:** 2026-09-16  
**Owner:** DIGIMON maintainers

## Goal

Fix fundamental defects exposed by the canary and CI before adding architecture ceremony. Let observed failures determine the next code change.

## Operating rule

For each failure, record only:

| Failure | Evidence | Smallest fix | Regression test | Status |
|---|---|---|---|---|
| _populate as failures appear_ |  |  |  |  |

Then implement the smallest fix and rerun the failing check.

## Priority order

1. clean installation / import failures;
2. canonical MCP initialization failures;
3. corpus/ER graph build or load failures;
4. entity VDB build/search failures;
5. relationship/chunk/source retrieval failures;
6. answer-generation/grounding failures;
7. incorrect failure swallowing or misleading success results;
8. only then broader composition/resource/provenance cleanup required by observed behavior.

## Current known fixes

### F1 — invalid full requirements pin

**Evidence:** CI cannot resolve `umap==0.1.1`; `umap-learn` is already present.

**Fix:** remove the invalid `umap` pin and retain `umap-learn`.

**Regression check:** dependency installation reaches tests.

### F2 — CI gates style/bootstrap before product signal

**Evidence:** Black and full research-environment installation prevent meaningful core execution checks.

**Fix:** narrow the first blocking CI gate to maintained core bootstrap + deterministic checks; keep broad/style checks advisory until useful.

**Regression check:** CI reaches and runs maintained-core tests.

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