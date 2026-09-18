# DIGIMON Planning

## Active execution authority

[NORTH_STAR_VERTICAL_SLICE_PLAN.md](NORTH_STAR_VERTICAL_SLICE_PLAN.md), revision 3, owns the batch sequence, boundaries, evidence and exact next action. The newest checkpoint is [Batch 02](../reports/PROJECTION_BATCH_02.md): 69 selected cumulative tests passed; saved Foundation graphs feed maintained storage/operators and exact evidence recovery. [Batch 01](../reports/PROJECTION_BATCH_01.md) is the preceding saved-project checkpoint.

The operating model is **brief expected behavior → coherent batch → execute → inspect trace/counterexample → repair → cumulative rerun → coherent commit**. Approximately 1,000 authored LOC/hour is a measured target, not a line-count gate. Do not stack dependent unexecuted batches.

The next boundary is **Batch 3, real vector indexing/query/reload** through the existing consumer, not a new planning exercise. The reports support the living plan; they do not create competing priorities. Runtime and real-input limitations remain explicit.

[FOUNDATION_PROPERTY_GRAPH_DESIGN.md](FOUNDATION_PROPERTY_GRAPH_DESIGN.md) records the two-graph intent. Tested supported-field reconstruction is not a universal lossless guarantee. The runtime entity view preserves parallel assertion records and explicitly describes association/weight/polarity semantics in the newest receipt.

[Planning-path record](supporting/north-star-speedrun-path.json) retains the durable_solo route. No work-claim registry or fictitious coordination is introduced.

## Supporting workstreams

[CORE_CANARY_PLAN.md](CORE_CANARY_PLAN.md), [CI_RECOVERY_PLAN.md](CI_RECOVERY_PLAN.md), and [FAILURE_DRIVEN_CORE_FIXES_PLAN.md](FAILURE_DRIVEN_CORE_FIXES_PLAN.md) support the active boundary. Their historical observations do not imply current test success.

[../VISION.md](../VISION.md) owns the whole Represent/Retrieve/Analyze thesis; [../ARCHITECTURE.md](../ARCHITECTURE.md) owns target invariants; [../ROADMAP.md](../ROADMAP.md) retains the broader horizon; [../CURRENT_STATE.md](../CURRENT_STATE.md) distinguishes tested and source-only capabilities. [../PLANNING_SUMMARY.md](../PLANNING_SUMMARY.md) is navigation. Older plans are historical unless the active authorities explicitly retain them.
