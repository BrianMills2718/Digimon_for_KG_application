# CLAUDE.md — DIGIMON Implementation Guide

**Updated:** 2026-09-16

This repository contains multiple generations of DIGIMON architecture. Use the canonical documentation below as the source of truth rather than inferring current intent from older planners, checkpoint files, or historical reports.

## Canonical documentation

Read in this order:

1. `docs/CURRENT_STATE.md` — implemented/partial/legacy/planned status of the public codebase.
2. `docs/ARCHITECTURE.md` — current target architecture.
3. `docs/GAP_ANALYSIS.md` — concrete gaps between code and target.
4. `docs/ROADMAP.md` — ordered architecture-completion plan.
5. `docs/adr/002-harness-first-capability-architecture.md` — accepted orchestration decision.
6. `docs/README.md` — documentation hierarchy and maintenance rules.

`README.md` and `FUNCTIONALITY.md` are concise public views of the same architecture.

## Current architectural decision

DIGIMON is **harness-first**.

> Program capabilities, typed contracts, resource lifecycle, and evidence boundaries. Use prompts for useful reasoning heuristics. Let the external harness remain intelligent.

### External intelligent harness owns

- understanding the user's goal and conversational context;
- deciding whether/how to decompose the task;
- choosing graph, vector, text, table, community or hybrid operations;
- sequencing, branching, retrying, parallelizing and stopping;
- revising strategy after observations;
- deciding when a reference method or auto-selection shortcut is useful.

### DIGIMON owns

- corpus preparation and retrieval-resource construction;
- typed retrieval/analysis capabilities;
- stable capability metadata and I/O contracts;
- resource identities, prerequisites, dependencies and lifecycle;
- source/evidence lineage;
- bounded model-assisted operations when semantic judgment is intrinsic to that operation;
- inspectable reference method plans.

Do not expand DIGIMON into a second general-purpose agent brain unless a new ADR explicitly changes this decision.

## Canonical code center

The modern architectural center is:

- `Core/Schema/SlotTypes.py` — seven typed dataflow slot kinds and records;
- `Core/Schema/OperatorDescriptor.py` — machine-readable capability metadata;
- `Core/Operators/registry.py` — registry of 26 operators;
- `Core/Operators/` — operator implementations;
- `Core/Composition/` — validation, execution and composition;
- `Core/Methods/` — 10 reference operator plans;
- `digimon_mcp_stdio_server.py` — preferred external harness/tool facade.

`OperatorComposer` deliberately profiles/builds/executes reference plans without owning the global LLM selection policy. A capable caller may compose capabilities directly.

## MCP execution hierarchy

The stdio MCP surface supports three useful modes:

1. **Individual capabilities/operators** — conceptual default for a capable harness.
2. **Reference methods** — execute a known composition through `execute_method`.
3. **Auto selection** — optional prompt/model chooses a reference method.

Do not make mode 3 the hidden mandatory control plane. Modes 2/3 are conveniences; mode 1 defines the preferred architecture boundary.

## AoT / GoT / ReAct

Treat Atom-of-Thought, Graph-of-Thought and ReAct as **optional reasoning heuristics**.

For example:

```text
q1: identify an intermediate entity
q2: retrieve facts about <q1.entity>
q3: verify the resulting claim against source evidence
```

This is advisory. The harness may merge, skip, reorder, branch, parallelize or revise these subgoals.

Do not create a mandatory reasoning DAG/state machine merely because dependency-aware prompting is useful. A formal reasoning graph is justified only when it enables a concrete capability such as scheduling, resumability, caching, provenance or auditing.

## Legacy/transitional architecture

The following areas are present but are not the target architectural center:

- `Core/AOT/` — older programmed atomic-state/transition reasoning;
- `Core/AgentBrain/` — older broad internal planning logic;
- `Core/AgentOrchestrator/` — several older/transitional orchestrators;
- `digimon_cli.py` — still uses `PlanningAgent`/`AgentOrchestrator` and experimental ReAct mode;
- old MCP WebSocket checkpoint plans at repository root;
- old UKRF/multi-agent/cognitive-architecture planning material.

Before deleting legacy code, identify live callers/tests. Before extending it, ask whether the canonical typed capability/resource architecture should own the requirement instead.

## Current implementation priority

Follow `docs/ROADMAP.md`. The intended sequence is:

1. stabilize the canonical capability contract and MCP parity;
2. unify resource identities, lifecycle and prerequisites;
3. make evidence/provenance an end-to-end contract;
4. clean the harness-first execution boundary;
5. consolidate legacy planner/orchestrator/AoT layers;
6. normalize cross-modal capabilities;
7. standardize machine-actionable errors/recovery;
8. harden architecture contract tests and CI;
9. add incremental/temporal/conflict semantics after the resource/evidence model exists;
10. perform broader benchmarking/research validation later.

Benchmark score optimization, novelty positioning and additional UI shells are intentionally not the current architectural priority.

## Capability implementation rules

For new canonical capabilities, prefer:

- typed explicit inputs and outputs;
- a machine-readable descriptor;
- explicit prerequisites/resources;
- stable resource identity;
- evidence/provenance propagation;
- a documented cost/model requirement when relevant;
- standardized failure/error semantics;
- deterministic behavior where possible;
- MCP discovery/execution parity;
- contract tests.

Avoid hidden build side effects and tool-specific prerequisite knowledge when the same information can be represented in the shared resource/capability model.

## Evidence rules

Current typed records already contain useful provenance foundations:

- `EntityRecord.source_id`;
- `RelationshipRecord.source_id`;
- `ChunkRecord.chunk_id`.

Preserve them.

Do not:

- treat a missing KG relation as proof the source corpus denies the relation;
- erase source/chunk identifiers unnecessarily;
- silently collapse contradictory sources;
- manufacture confidence values in place of evidence;
- let synthesis bridge facts that retrieval did not support.

The target is an explicit claim/evidence/source lineage contract described in `docs/ARCHITECTURE.md`.

## Cross-modal work

`Core/AgentTools/cross_modal_tools.py` contains substantive graph/table/vector conversion code. Treat it as implemented but not yet fully normalized into the operator/resource/provenance model.

When extending it, prefer explicit conversion descriptors, resource fingerprints, lossiness metadata and provenance mappings rather than adding new ad hoc payload conventions.

## Testing

Prioritize deterministic architecture/contract tests for:

- operator descriptor and slot compatibility;
- custom composition execution;
- MCP discovery/execution parity;
- resource registration/prerequisite/invalidation behavior;
- provenance propagation;
- graph-build → retrieve → source-evidence flows;
- standardized errors/recovery;
- cross-modal conversion contracts.

Keep live-provider/LLM tests separately classified because they depend on credentials, network, cost and model variance.

Historical benchmark/test numbers in old documents are not current guarantees unless rerun.

## Documentation maintenance

When implementation changes the architecture status:

1. update `docs/CURRENT_STATE.md`;
2. reconcile `docs/GAP_ANALYSIS.md`;
3. update `docs/ROADMAP.md` if priorities/exit criteria change;
4. change `docs/ARCHITECTURE.md` only if the target itself changes;
5. record meaningful design decisions under `docs/adr/`;
6. align `README.md`, `FUNCTIONALITY.md`, `AGENTS.md` and this file.

Do not create another competing current-state or roadmap document.

## Default implementation judgment

When the choice is between adding more internal reasoning machinery and making the capability/resource/evidence contract clearer, prefer the **capability/resource/evidence contract** unless the concrete task requires otherwise.
