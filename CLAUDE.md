# CLAUDE.md — DIGIMON Implementation Guide

**Updated:** 2026-09-16

This repository contains multiple generations of DIGIMON architecture. Use the canonical documentation below as the source of truth rather than inferring current intent from older planners, checkpoint files, or historical reports.

## Canonical documentation

Read in this order:

1. `docs/CURRENT_STATE.md` — implemented/partial/legacy/planned status of the public codebase.
2. `docs/IMPLEMENTATION_MAP.md` — exact module classification and code-level caveats.
3. `docs/ARCHITECTURE.md` — current target architecture.
4. `docs/GAP_ANALYSIS.md` — concrete gaps between code and target.
5. `docs/ROADMAP.md` — ordered architecture-completion plan.
6. `docs/adr/002-harness-first-capability-architecture.md` — accepted orchestration decision.
7. `docs/README.md` — documentation hierarchy and maintenance rules.

`README.md` and `FUNCTIONALITY.md` are concise public views of the same architecture.

## Current architectural decision

DIGIMON is **harness-first**.

> Program capabilities, typed contracts, resource lifecycle, validation semantics, and evidence boundaries. Use prompts for useful reasoning heuristics. Let the external harness remain intelligent.

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
- explicit validation/execution behavior;
- resource identities, prerequisites, dependencies and lifecycle;
- source/evidence lineage;
- bounded model-assisted operations when semantic judgment is intrinsic to that operation;
- inspectable reference method plans.

Do not expand DIGIMON into a second general-purpose agent brain unless a new ADR explicitly changes this decision.

## Canonical code center

The modern architectural center is:

- `Core/Schema/SlotTypes.py` — seven typed dataflow slot kinds and records;
- `Core/Schema/OperatorDescriptor.py` — machine-readable operator metadata;
- `Core/Operators/registry.py` — registry of 26 operators;
- `Core/Operators/` — operator implementations;
- `Core/Composition/` — validation, execution and composition;
- `Core/Methods/` — 10 reference operator plans;
- `digimon_mcp_stdio_server.py` — preferred external harness/tool facade.

`OperatorComposer` profiles/builds/executes reference plans without owning the global method-selection policy. A capable caller may compose capabilities directly.

## Important current implementation caveats

Before modifying the core, understand these verified details:

- registry compatibility/chain helpers primarily reason over slot kinds; they do not prove resource/prerequisite availability;
- `ChainValidator` can accept implicit same-kind availability with warnings;
- `OperatorComposer.execute()` currently logs static validation errors and may continue best-effort;
- `PipelineExecutor` performs stricter pre-dispatch slot checks and defaults to fail-fast execution;
- strict versus best-effort execution therefore needs one explicit caller-visible contract;
- some descriptor semantics need a full parity audit against implementations;
- `GraphRAGContext` directly models graph/VDB instances, not every derived resource type;
- `meta.decompose_question` uses `ENTITY_SET`/`EntityRecord` as a transitional carrier for sub-question strings;
- decomposition/synthesis prompt policy exists in both YAML and typed operator-local text;
- error representation differs between pipeline, individual operators and MCP/build tools;
- cross-modal DataFrame/NumPy/dictionary payloads are substantive but outside the canonical slot/resource/provenance model.

Use `docs/IMPLEMENTATION_MAP.md` for details instead of guessing from names.

## MCP execution hierarchy

The stdio MCP surface supports three useful modes:

1. **Individual capabilities/operators** — conceptual default for a capable harness.
2. **Reference methods** — execute a known composition.
3. **Auto selection** — optional prompt/model chooses a reference method.

Do not make mode 3 a hidden mandatory control plane. Modes 2/3 are conveniences; mode 1 defines the preferred architecture boundary.

## AoT / GoT / ReAct

Treat Atom-of-Thought, Graph-of-Thought and ReAct as **optional reasoning heuristics**.

For example:

```text
q1: identify an intermediate entity
q2: retrieve facts about <q1.entity>
q3: verify the resulting claim against source evidence
```

This is advisory. The harness may merge, skip, reorder, branch, parallelize or revise these subgoals.

Do not create a mandatory reasoning DAG/state machine merely because dependency-aware prompting is useful. A formal graph is justified only when it enables a concrete capability such as scheduling, resumability, caching, provenance or auditing.

If decomposition output needs better typing, consider a reusable text/task-list record before a cognitive-runtime abstraction.

## Prompt ownership

The current decomposition/synthesis policy exists in two implementation surfaces:

- `prompts/decompose_question.yaml` and `prompts/synthesize_answers.yaml`;
- `Core/Operators/meta/decompose_question.py` and `Core/Operators/meta/synthesize_answers.py`.

They are aligned in this snapshot, but they are duplicate sources. Until centralized, changes must reconcile both paths and should gain parity tests.

Do not claim the YAML prompt changed runtime behavior unless the relevant execution path actually loads it.

## Legacy/transitional architecture

The following are present but are not the target architectural center:

- `Core/AOT/` — older programmed atomic-state/transition reasoning;
- `Core/AgentBrain/` — older broad internal planning logic;
- `Core/AgentOrchestrator/` — several older/transitional orchestrators;
- `Core/Memory/` — earlier strategy/memory subsystem;
- older `Core/MCP/` clients/servers/coordination experiments;
- `digimon_cli.py` — still uses `PlanningAgent`/`AgentOrchestrator` and experimental ReAct mode;
- historical MCP WebSocket/checkpoint and UKRF/multi-agent material.

Before deleting legacy code, identify live callers/tests. Before extending it, ask whether the canonical typed capability/resource architecture should own the requirement instead.

## Current implementation priority

Follow `docs/ROADMAP.md`. The intended sequence is:

1. capability/descriptor/MCP inventory and parity;
2. explicit strict-vs-best-effort validation semantics;
3. prompt source-of-truth/parity;
4. resource identities, lifecycle and prerequisites;
5. end-to-end evidence/provenance;
6. clean harness-first execution boundary;
7. legacy planner/orchestrator/AoT/MCP consolidation;
8. cross-modal normalization;
9. machine-actionable errors/recovery;
10. blocking architecture contract tests and CI;
11. incremental/temporal/conflict semantics after the resource/evidence foundation exists;
12. broader benchmarking/research validation later.

Benchmark score optimization, novelty positioning and additional UI shells are intentionally not the current architectural priority.

## Capability implementation rules

For new canonical capabilities, prefer:

- typed explicit inputs and outputs;
- machine-readable descriptor metadata that matches implementation behavior;
- explicit resource prerequisites and producers;
- stable resource identity;
- explicit strict/best-effort behavior where relevant;
- evidence/provenance propagation;
- documented cost/model/side-effect/lossiness characteristics;
- standardized failure/error semantics;
- deterministic behavior where possible;
- MCP discovery/execution parity;
- contract tests.

Avoid hidden build side effects and tool-specific prerequisite knowledge when the same facts can live in the shared resource/capability model.

## Evidence rules

Current records provide provenance foundations:

- `EntityRecord.source_id`;
- `RelationshipRecord.source_id`;
- `ChunkRecord.chunk_id`;
- `SlotValue.producer` and metadata.

Preserve them.

Do not:

- treat a missing KG relation as proof the source corpus denies the relation;
- erase source/chunk identifiers unnecessarily;
- silently collapse contradictory sources;
- manufacture confidence values in place of evidence;
- let synthesis bridge facts that retrieval did not support.

## Cross-modal work

`Core/AgentTools/cross_modal_tools.py` contains substantive graph/table/vector conversion code. Treat it as implemented but not yet fully normalized into the operator/resource/provenance model.

When extending it, prefer explicit conversion descriptors, resource fingerprints, lossiness metadata and provenance mappings rather than new ad hoc payload conventions.

## Testing

Prioritize deterministic architecture/contract tests for:

- operator descriptor↔implementation parity;
- slot and field compatibility;
- strict/best-effort validation behavior;
- custom composition execution;
- MCP discovery/execution parity;
- prompt semantic parity while duplicate prompt sources remain;
- resource registration/prerequisite/invalidation behavior;
- provenance propagation;
- graph-build→retrieve→source-evidence flows;
- standardized errors/recovery;
- cross-modal conversion contracts.

Keep live-provider/LLM tests separately classified because they depend on credentials, network, cost and model variance.

Historical benchmark/test numbers are not current guarantees unless rerun.

## Documentation maintenance

When implementation changes architecture status:

1. update `docs/CURRENT_STATE.md`;
2. update `docs/IMPLEMENTATION_MAP.md` for module/contract changes;
3. reconcile `docs/GAP_ANALYSIS.md`;
4. update `docs/ROADMAP.md` if priorities/exit criteria change;
5. change `docs/ARCHITECTURE.md` only if the target changes;
6. record meaningful design decisions under `docs/adr/`;
7. align `README.md`, `FUNCTIONALITY.md`, `AGENTS.md` and this file.

Do not create another competing current-state or roadmap document.

## Default implementation judgment

When the choice is between adding more internal reasoning machinery and making the capability/resource/evidence/validation contract clearer, prefer the **capability/resource/evidence/validation contract** unless the concrete task requires otherwise.