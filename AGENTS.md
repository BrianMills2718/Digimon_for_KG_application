# AGENTS.md — DIGIMON Contributor and Coding-Agent Guide

**Updated:** 2026-09-16

This file gives implementation guidance to coding agents working in this repository. It points to the canonical documentation rather than duplicating a dated checkpoint plan.

## Read these first

The source-of-truth documentation is:

1. `docs/CURRENT_STATE.md` — what is actually implemented now.
2. `docs/IMPLEMENTATION_MAP.md` — module classification and concrete implementation caveats.
3. `docs/ARCHITECTURE.md` — target architecture.
4. `docs/GAP_ANALYSIS.md` — concrete current→target gaps.
5. `docs/ROADMAP.md` — ordered architecture-completion plan and exit criteria.
6. `docs/adr/002-harness-first-capability-architecture.md` — accepted orchestration decision.

`README.md` and `FUNCTIONALITY.md` are concise public views. Older status trackers, UKRF plans, MCP checkpoint plans and implementation reports are historical unless the canonical docs explicitly restate them.

## Current architectural direction

DIGIMON is **harness-first**.

### External intelligent harness owns

- goal interpretation;
- whether/how to decompose a problem;
- tool/capability selection and sequencing;
- retries, fallbacks, branching and parallel work;
- adapting after observations;
- stopping criteria.

### DIGIMON owns

- corpus/graph/index/resource construction;
- typed retrieval and analysis capabilities;
- capability metadata and compatibility;
- resource/prerequisite facts and lifecycle;
- source/evidence lineage;
- bounded model-assisted operations where an individual capability requires semantic judgment;
- reference method plans as optional conveniences.

Do **not** add another general-purpose planner/orchestrator or mandatory cognitive state machine unless a new ADR explicitly changes this decision.

## Canonical code center

The strongest current core is:

- `Core/Schema/SlotTypes.py` — typed slot/dataflow records;
- `Core/Schema/OperatorDescriptor.py` — operator metadata;
- `Core/Operators/registry.py` — 26-operator registry;
- `Core/Operators/` — operator implementations;
- `Core/Composition/` — validation/execution/composition;
- `Core/Methods/` — 10 reference plans;
- `digimon_mcp_stdio_server.py` — current external-harness MCP facade.

When adding functionality, prefer extending or mapping into this capability/resource/evidence model rather than creating a parallel registry/execution abstraction.

## Important current implementation caveats

Do not infer more guarantees than the code currently provides:

- registry compatibility helpers are primarily slot-kind discovery aids, not proof that prerequisites/resources make a chain executable;
- `ChainValidator` is permissive in places and can warn about implicit same-kind wiring;
- `OperatorComposer.execute()` currently logs static validation failures and may proceed best-effort;
- `PipelineExecutor` performs stricter pre-dispatch slot checks and defaults to fail-fast operator execution;
- strict versus best-effort composition must become an explicit contract;
- `GraphRAGContext` directly tracks graphs/VDBs, not every derived resource type;
- `meta.decompose_question` currently uses `ENTITY_SET`/`EntityRecord` as a transitional carrier for sub-question text;
- decomposition/synthesis policy exists in both YAML and operator-local prompt text, so prompt ownership/parity is not finished;
- error representation differs across composition, individual operators and MCP/build tools;
- cross-modal DataFrame/array/dictionary payloads are not yet normalized into the core slot/resource/provenance system.

See `docs/IMPLEMENTATION_MAP.md` before changing these areas.

## Current implementation priorities

Follow `docs/ROADMAP.md`. The active sequence is:

1. audit capability descriptors, implementations and MCP parity;
2. make strict-vs-best-effort validation semantics explicit;
3. establish prompt source-of-truth/parity for meta operators;
4. unify resource identities/lifecycle/prerequisites;
5. make evidence/provenance an end-to-end contract;
6. clean the harness-first execution boundary;
7. consolidate legacy internal planning/AoT/MCP layers;
8. normalize cross-modal capabilities;
9. standardize machine-actionable errors/recovery;
10. harden architectural contract tests and CI.

Benchmark optimization, novelty claims, router calibration and new UI surfaces are not the current priority.

## Status vocabulary

Use these terms consistently:

- **Implemented** — substantive code exists and is wired into a current surface.
- **Partial** — code exists but lifecycle/integration/contracts/reliability are incomplete.
- **Legacy** — retained for compatibility/history, not target architecture.
- **Planned** — not materially complete yet.

Do not call something “complete” merely because a module/file exists.

## Legacy/transitional areas

Treat these carefully:

- `Core/AOT/` — legacy programmed atomic-state/transition approach;
- `Core/AgentBrain/` — older broad internal planning layer;
- `Core/AgentOrchestrator/` — multiple older/transitional orchestrators;
- `Core/Memory/` — earlier strategy/memory architecture, not current priority;
- much of `Core/MCP/` — older MCP/coordination lineage; the root stdio server is the current preferred facade;
- `digimon_cli.py` — still calls internal `PlanningAgent`/`AgentOrchestrator`;
- historical MCP/checkpoint/UKRF/multi-agent documents.

Before deleting legacy code, identify live callers/tests. Before extending it, verify the canonical capability architecture cannot serve the need more cleanly.

## AoT / GoT / ReAct guidance

These are **reasoning heuristics**, not mandatory DIGIMON runtimes.

Dependency-aware subgoals may refer to prior discoveries, for example `<q1.entity>`. The harness may merge, skip, reorder, branch, parallelize or revise them.

Do not formalize a reasoning DAG merely because the prompt can express dependencies. Only add such structure when it enables a concrete function such as scheduling, resumability, caching, provenance or auditing.

If better typing is needed for decomposition output, consider a reusable text/task-list record before building a cognitive-runtime abstraction.

## Prompt maintenance

Current decomposition/synthesis behavior exists in both:

- `prompts/*.yaml` templates; and
- typed meta-operator prompt text under `Core/Operators/meta/`.

They are aligned in the current snapshot but are duplicate sources. Until prompt ownership is centralized, changes to one must be reconciled with the other and covered by parity tests where practical.

## Capability design guidance

For a new canonical capability, prefer:

1. explicit typed inputs/outputs;
2. machine-readable descriptor/metadata;
3. explicit resource prerequisites;
4. stable resource identifiers;
5. clear strict/best-effort and failure semantics;
6. source/evidence lineage preservation;
7. deterministic behavior where possible;
8. bounded/documented LLM use where semantic judgment is intrinsic;
9. MCP exposure/discovery synchronized with the capability definition;
10. contract tests.

Avoid hiding prerequisites or resource-building side effects from the caller.

## Evidence rules

Current records provide `source_id`, `chunk_id`, producer and metadata foundations. Preserve them whenever possible.

Do not:

- treat missing graph evidence as proof a claim is false;
- strip evidence identifiers unnecessarily;
- silently select one source when sources conflict;
- invent confidence values as a substitute for evidence;
- convert retrieved evidence into unsupported inference during synthesis.

## Testing guidance

Prefer deterministic contract tests for:

- descriptor↔implementation parity;
- slot/field compatibility;
- strict/best-effort validation behavior;
- operator execution boundaries;
- MCP discovery/execution parity;
- prompt semantic parity while duplicate prompt sources remain;
- resource registration/prerequisite/invalidation behavior;
- provenance propagation;
- standardized errors;
- graph-build→retrieve→evidence flows.

Keep live-LLM/provider tests clearly separated because they have cost, network and model-variance concerns.

Do not cite an old test result as current runtime truth unless it has been rerun or explicitly labeled historical.

## Documentation maintenance

When implementation changes architectural status:

1. update `docs/CURRENT_STATE.md`;
2. update `docs/IMPLEMENTATION_MAP.md` when module/contracts change;
3. reconcile `docs/GAP_ANALYSIS.md`;
4. update `docs/ROADMAP.md` if priorities/exit criteria change;
5. update `docs/ARCHITECTURE.md` only if the target design changes;
6. create/update an ADR for a real architectural decision;
7. reconcile `README.md`, `FUNCTIONALITY.md`, `AGENTS.md`, and `CLAUDE.md` when guidance changes.

Do not create another competing “current status” or checkpoint document.

## Default decision rule

When choosing between:

- making the internal agent brain more elaborate, or
- making a capability/resource/evidence/validation contract clearer,

prefer the **capability/resource/evidence/validation contract** unless the concrete task explicitly requires otherwise.