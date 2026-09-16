# DIGIMON Documentation Index

**Canonical snapshot date:** 2026-09-16

This directory contains both current architecture documentation and older research/planning material. The files listed under **Canonical documentation** are the source of truth for the public snapshot. Older reports, checkpoint plans, handoffs, and exploratory notes remain useful as project history but must not override the canonical set.

## Canonical documentation

1. **[CURRENT_STATE.md](CURRENT_STATE.md)** — what the codebase actually contains and how complete each major capability is.
2. **[IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)** — module-by-module classification, operator/method inventory, and concrete implementation caveats.
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** — target harness-first capability/resource/evidence architecture and design invariants.
4. **[GAP_ANALYSIS.md](GAP_ANALYSIS.md)** — concrete gaps between present code and target architecture.
5. **[ROADMAP.md](ROADMAP.md)** — architecture-completion sequence and exit criteria.
6. **[PLANNING_SUMMARY.md](PLANNING_SUMMARY.md)** — concise current planning summary derived from the roadmap.
7. **[../FUNCTIONALITY.md](../FUNCTIONALITY.md)** — concise implemented-capability view.
8. **[QUICK_START.md](QUICK_START.md)** — current public-snapshot setup and entry points.
9. **[AGENT_INTELLIGENCE_ENHANCEMENTS.md](AGENT_INTELLIGENCE_ENHANCEMENTS.md)** — reasoning-policy detail: harness-first orchestration and AoT/GoT as a heuristic rather than a programmed brain.
10. **[FUTURE_EVALUATION_QUESTIONS.md](FUTURE_EVALUATION_QUESTIONS.md)** — deliberately deferred benchmarking, ablation, and research questions.

## Architecture decisions

- **[adr/002-harness-first-capability-architecture.md](adr/002-harness-first-capability-architecture.md)** — accepted current decision: DIGIMON owns capabilities/contracts/resources/evidence boundaries; the external harness owns adaptive orchestration.
- `adr/001-agent-orchestration-architecture.md` is retained as a superseded decision record documenting the earlier dual-brain design.

## Documentation status vocabulary

Canonical status documents use four labels consistently:

- **Implemented** — meaningful code exists and is wired into a current execution surface.
- **Partial** — meaningful code exists, but integration, consistency, contracts, lifecycle, or reliability remain incomplete.
- **Legacy** — code/documentation remains for compatibility or history but is not the preferred target architecture.
- **Planned** — target behavior is not materially complete in this public snapshot.

`Implemented` does **not** mean every provider/environment combination was executed during the documentation review. Where live runtime verification was not performed, the status documents say so explicitly.

## Current architectural reading order

For a new contributor or reviewer:

```text
README.md
   ↓
docs/CURRENT_STATE.md
   ↓
docs/IMPLEMENTATION_MAP.md
   ↓
docs/ARCHITECTURE.md
   ↓
docs/GAP_ANALYSIS.md
   ↓
docs/ROADMAP.md
```

Use `FUNCTIONALITY.md` for a shorter capability inventory, `PLANNING_SUMMARY.md` for the concise current plan, and `QUICK_START.md` to run a current entry point.

## Current implementation truths worth keeping visible

The 2026-09-16 reconciliation verified several details that should not be simplified away in future summaries:

- the 26-operator typed core is real, but composition validation still needs hardening;
- compatibility/chain discovery based on slot kinds is not proof that prerequisites/resources are available;
- `OperatorComposer` currently has a best-effort path after validation errors, so strict-vs-best-effort semantics remain a gap;
- `GraphRAGContext` directly models graphs/VDBs, not every derived resource type;
- dependency-aware decomposition is advisory and currently uses `ENTITY_SET` as a transitional carrier for sub-question text;
- decomposition/synthesis policy exists in both YAML and typed operator-local prompts, so prompt source-of-truth/parity is still an architectural concern;
- evidence identifiers exist, but universal lineage propagation is not finished;
- error behavior is not yet uniform across operators, composition and MCP/build tools;
- cross-modal graph/table/vector code is substantive but not normalized into the same typed resource/provenance system;
- CLI/internal planners remain transitional even though the stdio MCP facade is the preferred harness boundary.

See `IMPLEMENTATION_MAP.md` for details.

## Historical and supporting material

The repository contains substantial earlier planning/research material, including MCP checkpoint plans, UKRF-era plans, internal-agent/memory proposals, implementation reports and handoffs. These files document project evolution and may contain useful ideas, but many describe architectures or priorities that have changed.

In particular, old documents that prescribe:

- a hand-built general-purpose cognitive/agent state machine;
- mandatory AoT/Markov preprocessing;
- a WebSocket MCP checkpoint sequence;
- multi-agent coordination as the immediate priority; or
- benchmark/production targets as the current implementation priority

must be treated as **historical unless restated in the canonical documentation**.

`CHECKPOINT_PROGRESS.md` and the root MCP planning/tracker files are explicitly historical stubs; their original details remain available in Git history.

## Maintenance rule

When implementation or architecture changes:

1. record a real architectural decision in `docs/adr/` when appropriate;
2. update `CURRENT_STATE.md` to reflect code reality;
3. update `IMPLEMENTATION_MAP.md` when module classifications/contracts change;
4. update `ARCHITECTURE.md` only when the target design changes;
5. reconcile `GAP_ANALYSIS.md`;
6. update `ROADMAP.md` exit criteria/priorities;
7. update `PLANNING_SUMMARY.md` if the immediate plan changed;
8. reconcile `README.md`, `FUNCTIONALITY.md`, `AGENTS.md`, and `CLAUDE.md` if user/agent-facing guidance changed.

Do not create another competing “current status,” “active checkpoint,” or roadmap document.