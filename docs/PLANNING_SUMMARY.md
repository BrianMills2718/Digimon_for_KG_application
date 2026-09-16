# DIGIMON Planning Summary

**Updated:** 2026-09-16  
**Status:** current planning summary

The current priority is **architecture completion and consolidation**, not benchmark optimization, novelty positioning, a new UI, or a larger internal cognitive architecture.

## Current direction

DIGIMON is being organized around a **harness-first capability architecture**:

- the external intelligent harness owns adaptive reasoning, decomposition, tool selection, sequencing, retries/fallbacks and stopping;
- DIGIMON owns typed capabilities, retrieval/build operations, resource/prerequisite facts, evidence/provenance and bounded model-assisted transformations;
- AoT/GoT/ReAct and method-routing prompts are optional heuristics, not mandatory runtime state machines;
- reference GraphRAG methods remain useful compositions, shortcuts and later evaluation baselines rather than the system's architectural identity.

The strongest modern implementation center is the typed 26-operator registry/composition layer plus the FastMCP stdio server.

## Active plan

The authoritative implementation sequence is in `docs/ROADMAP.md`:

1. stabilize the canonical capability contract and MCP parity;
2. unify resource identities, lifecycle and prerequisites;
3. make provenance/evidence an end-to-end contract;
4. make the harness-first execution boundary operationally clean;
5. consolidate legacy internal planners/orchestrators/AoT code;
6. normalize cross-modal graph/table/vector capabilities;
7. standardize machine-actionable failure/recovery semantics;
8. make architectural contract tests blocking and trustworthy in CI;
9. add incremental-update and temporal/conflict semantics once the resource/evidence foundation exists;
10. perform broad benchmarking, ablation and research validation later.

## Largest current gaps

The architectural bottlenecks are not lack of retrieval algorithms. They are consistency and contracts:

- capability discovery spans the 26-operator registry plus additional MCP build/analysis/conversion tools;
- resource state is not yet represented by one typed catalog with dependencies/fingerprints/invalidation;
- prerequisite behavior is useful but distributed across descriptors/server helpers;
- source identifiers exist, but lineage is not yet a universal end-to-end evidence contract;
- CLI/internal planner paths still coexist with the newer harness-first MCP architecture;
- graph/table/vector conversion is substantive but not fully normalized into the same capability/resource/provenance model;
- error/recovery semantics vary between tool families;
- old planners, orchestrators, AoT code and historical plan documents remain in the repository and require explicit classification/consolidation.

See `docs/GAP_ANALYSIS.md` for the complete gap matrix.

## What is deliberately deferred

The following remain useful future work but should not distort the current architecture:

- KG versus BM25/vector/hybrid ablations;
- router/method-selection calibration;
- novelty/research-positioning claims;
- benchmark-score optimization;
- production-scale latency/token tuning;
- generalized multi-agent coordination;
- additional dashboards/UI shells.

The important future validation questions are preserved in `docs/FUTURE_EVALUATION_QUESTIONS.md`.

## Historical planning material

Earlier planning in this repository explored:

- UKRF/general agent frameworks;
- multi-agent coordination;
- programmed AoT/Markov decomposition;
- WebSocket MCP migration checkpoints;
- confidence/meta-cognition subsystems;
- production/performance phases.

Those documents are valuable project lineage, but they are **historical unless a current canonical document restates the requirement**.

The current source-of-truth set is:

- `docs/CURRENT_STATE.md`
- `docs/ARCHITECTURE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`
- `docs/adr/002-harness-first-capability-architecture.md`

## Immediate next code work

When implementation resumes, start at the top of the roadmap rather than adding another orchestration abstraction:

1. inventory/map all harness-facing capabilities against the canonical descriptor model;
2. design the typed resource descriptor/catalog;
3. connect prerequisites/builders to that resource model;
4. define and propagate the evidence/provenance record;
5. then clean entry points and legacy orchestration around those stable contracts.

That sequence turns the existing breadth of DIGIMON into a coherent architecture without attempting to program the harness's intelligence directly.