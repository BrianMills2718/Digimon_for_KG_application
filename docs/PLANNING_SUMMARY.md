# DIGIMON Planning Summary

**Updated:** 2026-09-16  
**Status:** current planning summary

The current priority is **architecture completion and consolidation**, not benchmark optimization, novelty positioning, a new UI, or a larger internal cognitive architecture.

## Current direction

DIGIMON is being organized around a **harness-first capability/resource/evidence architecture**:

- the external intelligent harness owns adaptive reasoning, decomposition, tool selection, sequencing, retries/fallbacks and stopping;
- DIGIMON owns typed capabilities, retrieval/build operations, resource/prerequisite facts, evidence/provenance and bounded model-assisted transformations;
- AoT/GoT/ReAct and method-routing prompts are optional heuristics, not mandatory runtime state machines;
- reference GraphRAG methods remain useful compositions, shortcuts and later evaluation baselines rather than the system's architectural identity.

The strongest modern implementation center is the typed 26-operator registry/composition layer plus the FastMCP stdio server.

For the exact module map and code-level caveats, see `docs/IMPLEMENTATION_MAP.md`.

## Important finding from the current reconciliation

The capability core is real, but **typed does not yet mean architecturally closed**.

Current implementation facts that shape the next work:

- slot-kind chain discovery does not prove that resources/prerequisites are available;
- static `ChainValidator` behavior is permissive in places;
- `OperatorComposer` can currently log validation failure and proceed best-effort;
- `PipelineExecutor` performs stricter dispatch-time checks;
- descriptor semantics still need a full implementation audit;
- dependency-aware decomposition currently carries sub-question text through the generic `ENTITY_SET` slot;
- decomposition/synthesis prompt policy exists in both YAML and operator-local prompt text, so source-of-truth/parity must be made explicit;
- resource state is broader than the graphs/VDBs directly modeled by `GraphRAGContext`;
- evidence identifiers exist, but universal lineage propagation is incomplete;
- error/failure representation varies across composition, individual operators and MCP/build tools.

These are the near-term architecture gaps. They are not reasons to add another planner.

## Active plan

The authoritative sequence is in `docs/ROADMAP.md`:

1. **capability/descriptor/MCP inventory and parity**;
2. **validation semantics** — make strict versus explicit best-effort behavior unambiguous;
3. **prompt ownership/parity** — prevent YAML/operator-local prompt drift;
4. **resource catalog/lifecycle/prerequisites**;
5. **end-to-end evidence/provenance**;
6. **clean harness-first entry points**;
7. **legacy planner/orchestrator/AoT/MCP consolidation**;
8. **cross-modal normalization**;
9. **machine-actionable error/recovery semantics**;
10. **blocking architectural contract tests/CI**;
11. incremental/temporal/conflict semantics once the foundation exists;
12. broader benchmarking/research validation later.

## Largest current gaps

### Capability discovery and descriptor precision

The 26-operator registry is the strongest machine-readable core, but MCP also exposes builders, configuration/resource inspection, graph analysis and cross-modal tools outside that descriptor model. Existing descriptors also need an implementation-level parity audit.

### Composition validation policy

Validation currently happens at multiple levels with different strictness. The architecture needs a deliberate contract: strict execution should fail before running an invalid plan; best-effort execution should be an explicit caller choice with structured warnings.

### Resource lifecycle

There is no one typed catalog covering corpus, graphs, VDBs, communities, sparse matrices, converted artifacts, build fingerprints, staleness and dependency/invalidation links.

### Evidence/provenance

`source_id`, `chunk_id`, producer metadata and evidence-aware synthesis are useful foundations. The missing piece is guaranteed lineage propagation through every relevant transformation.

### Error semantics

Missing prerequisite, empty retrieval, invalid plan, extraction incompleteness and provider failure still use different conventions. A harness needs structured failure facts in order to recover intelligently.

### Legacy architecture coexistence

CLI/internal planner paths, multiple orchestrators, programmed AOT, older MCP modules and historical planning material remain in the repository. They must be maintained as clearly classified compatibility/history rather than equal architectural authorities.

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

## Reasoning-policy decision

Do not program the harness's full reasoning graph into DIGIMON.

Dependency-aware decomposition is useful as an advisory heuristic, for example:

```text
q1: identify an intermediate entity
q2: retrieve facts about <q1.entity>
q3: resolve the answer against source evidence
```

The harness may merge, reorder, branch, parallelize, revise or ignore the suggestion.

A formal dependency DAG is justified only if a concrete system capability—scheduling, resumability, caching, provenance or auditing—needs it.

## Historical planning material

Earlier planning explored UKRF/general-agent frameworks, multi-agent coordination, programmed AoT/Markov decomposition, WebSocket MCP migration checkpoints, memory/meta-cognition systems, and production/performance phases.

Those documents are project lineage, not current implementation mandates. `docs/CHECKPOINT_PROGRESS.md` and the root MCP plans are explicitly marked historical; the original details remain in Git history.

## Current source-of-truth set

- `docs/CURRENT_STATE.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/ARCHITECTURE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`
- `docs/adr/002-harness-first-capability-architecture.md`

## Immediate next code work

When implementation resumes, do not begin with a new agent abstraction. Begin with the contracts the harness needs:

1. enumerate every harness-facing capability and compare descriptor/MCP/implementation semantics;
2. make plan validation policy explicit and test it;
3. establish prompt source-of-truth/parity for meta operators;
4. design the typed resource catalog and prerequisite links;
5. define and propagate the evidence/provenance record;
6. then clean entry points and legacy orchestration around those stable contracts.

That sequence turns DIGIMON's existing breadth into a coherent architecture without attempting to reproduce the harness's intelligence inside the library.