# AGENTS.md — DIGIMON Contributor and Coding-Agent Guide

**Updated:** 2026-09-16

This file gives implementation guidance to coding agents working in this repository. It intentionally points to the canonical documentation rather than duplicating a dated checkpoint plan.

## Read these first

The source-of-truth documentation is:

1. `docs/CURRENT_STATE.md` — what is actually implemented now.
2. `docs/ARCHITECTURE.md` — target architecture.
3. `docs/GAP_ANALYSIS.md` — concrete current→target gaps.
4. `docs/ROADMAP.md` — ordered architecture-completion plan and exit criteria.
5. `docs/adr/002-harness-first-capability-architecture.md` — accepted orchestration decision.

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

When adding functionality, prefer extending or mapping into this capability model rather than creating a parallel registry/execution abstraction.

## Current implementation priorities

Follow `docs/ROADMAP.md`. The active sequence is:

1. stabilize the capability contract and MCP parity;
2. unify resource identities/lifecycle/prerequisites;
3. make evidence/provenance an end-to-end contract;
4. clean the harness-first execution boundary;
5. consolidate legacy internal planning/AoT layers;
6. normalize cross-modal capabilities;
7. standardize machine-actionable errors/recovery;
8. harden architectural contract tests and CI.

Benchmark optimization, novelty claims, router calibration and new UI surfaces are not the current priority.

## Status vocabulary

Use these terms consistently in docs/issues/code comments:

- **Implemented** — substantive code exists and is wired into a current surface.
- **Partial** — code exists but lifecycle/integration/contracts/reliability are incomplete.
- **Legacy** — retained for compatibility/history, not target architecture.
- **Planned** — not materially complete yet.

Do not call something “complete” merely because a module/file exists.

## Legacy/transitional areas

Treat the following carefully:

- `Core/AOT/` — legacy programmed atomic-state/transition approach;
- `Core/AgentBrain/` — older broad internal planning layer;
- `Core/AgentOrchestrator/` — multiple older/transitional orchestrators;
- `digimon_cli.py` — still calls internal `PlanningAgent`/`AgentOrchestrator`;
- root MCP checkpoint/tracker documents — historical 2025 planning lineage;
- older UKRF/multi-agent planning documents — research history, not current mandate.

Before deleting legacy code, identify live callers and tests. Before extending it, verify that the target capability architecture cannot serve the same need more cleanly.

## AoT / GoT / ReAct guidance

These are **reasoning heuristics**, not mandatory DIGIMON runtimes.

`prompts/decompose_question.yaml` may suggest dependency-aware subgoals. The harness may merge, skip, reorder, branch or revise them.

Only formalize a reasoning DAG/state object when it enables a concrete system function such as scheduling, resumability, caching, provenance or auditing.

## Capability design guidance

For a new canonical capability, prefer:

1. explicit typed inputs/outputs;
2. machine-readable descriptor/metadata;
3. explicit resource prerequisites;
4. stable resource identifiers;
5. structured error/failure semantics;
6. source/evidence lineage preservation;
7. deterministic behavior where possible;
8. bounded/documented LLM use where semantic judgment is intrinsic;
9. MCP exposure/discovery that stays synchronized with the capability definition;
10. contract tests.

Avoid hiding prerequisites or resource-building side effects from the caller.

## Evidence rules

Current `EntityRecord`/`RelationshipRecord` include `source_id`; `ChunkRecord` includes `chunk_id`. Preserve these identifiers whenever possible.

Do not:

- treat missing graph evidence as proof a claim is false;
- strip evidence identifiers unnecessarily;
- silently select one source when sources conflict;
- invent confidence values as a substitute for evidence;
- convert retrieved evidence into unsupported inference during synthesis.

The target evidence contract is documented in `docs/ARCHITECTURE.md` and `docs/ROADMAP.md`.

## Testing guidance

Prefer deterministic contract tests for:

- slot/descriptor compatibility;
- operator execution boundaries;
- MCP discovery/execution parity;
- resource registration/prerequisite behavior;
- provenance propagation;
- standardized errors;
- graph-build → retrieve → evidence flows.

Keep live-LLM/provider tests clearly separated because they have cost, network and model-variance concerns.

Do not cite an old test result in documentation as current runtime truth unless it has been rerun or the text explicitly labels it historical.

## Documentation maintenance

When implementation changes architectural status:

1. update `docs/CURRENT_STATE.md`;
2. update `docs/GAP_ANALYSIS.md` if a gap closes/changes;
3. update `docs/ROADMAP.md` if exit criteria/priorities change;
4. update `docs/ARCHITECTURE.md` only if the target design changes;
5. create/update an ADR for a real architectural decision;
6. reconcile `README.md`, `FUNCTIONALITY.md`, `AGENTS.md`, and `CLAUDE.md` when user/agent guidance changes.

Do not create another competing “current status” document.

## Default decision rule

When choosing between:

- making the internal agent brain more elaborate, or
- making a capability/resource/evidence contract clearer,

prefer the **capability/resource/evidence contract** unless the task explicitly requires otherwise.
