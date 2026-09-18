# DIGIMON Planning

**Updated:** 2026-09-17 planning cycle; batches 0–1 now have scoped execution evidence.

## Active Execution Authority

[NORTH_STAR_VERTICAL_SLICE_PLAN.md](NORTH_STAR_VERTICAL_SLICE_PLAN.md), revision 2, is the single active execution plan for the governed-IR north-star slice. It owns the batch sequence, additional LOC ranges, focused checks, trace/lineage expectations and unresolved assumptions.

**Latest execution checkpoint:** [PROJECTION_BATCH_01.md](../reports/PROJECTION_BATCH_01.md). The selected baseline and saved-project increment passed 45 focused checks; SQL evidence lookup, GraphML reconstruction, separate-process reopening, and failure-preserving rebuilds executed. This supersedes the initial source-only baseline for those selected files, not the unverified whole-repository or real-harness gates. **Current next action: Batch 2, maintained graph retrieval adoption**, then actual vector integration. Do not repeat Batch 0 as though no scoped test run exists.

Its operating model is **brief contract → coherent implementation batch → execute → inspect trace/counterexample → repair → cumulative rerun → coherent commit**. The approximately 1,000 LOC/hour authoring target is measured separately from verified progress. Do not build another dependent batch on an unexecuted one.

[FOUNDATION_PROPERTY_GRAPH_DESIGN.md](FOUNDATION_PROPERTY_GRAPH_DESIGN.md) records the two-graph design intent. Round-trip fidelity and runtime adoption require tests; the design's use of “lossless” is not universal runtime certification.

[The planning-path record](supporting/north-star-speedrun-path.json) selects Company Planning's durable_solo route for one-writer continuity. It is consumed by the planning-path validator, not a work-claim registry or dispatcher.

## Supporting Workstreams, Not Competing Priorities

- [CORE_CANARY_PLAN.md](CORE_CANARY_PLAN.md) retains the maintained MCP/core reuse and rebuild checks.
- [CI_RECOVERY_PLAN.md](CI_RECOVERY_PLAN.md) retains CI recovery work when an actual execution blocker requires it.
- [FAILURE_DRIVEN_CORE_FIXES_PLAN.md](FAILURE_DRIVEN_CORE_FIXES_PLAN.md) retains the concrete defect/evidence ledger.

They support the active frontier rather than each selecting a different next project. Historical source-only progress records do not imply test success; the scoped receipt does not imply all core tests passed.

## Relationship To Canonical Documentation

[../VISION.md](../VISION.md) owns the full Represent/Retrieve/Analyze thesis and ecosystem boundary. [../ARCHITECTURE.md](../ARCHITECTURE.md) owns target invariants. [../ROADMAP.md](../ROADMAP.md) retains the longer capability horizon. [../CURRENT_STATE.md](../CURRENT_STATE.md) and [../IMPLEMENTATION_MAP.md](../IMPLEMENTATION_MAP.md) own system-wide code status; read the linked execution receipt for this newer projection checkpoint. [../PLANNING_SUMMARY.md](../PLANNING_SUMMARY.md) is a concise navigation view.

Older plans are historical unless the active authorities explicitly retain them. Prefer a useful implementation result over expanding plans. Update evidence when results change status, scope, assumptions or the next action; add no second tracker or per-fix packet merely for ceremony.
