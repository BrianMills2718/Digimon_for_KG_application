# DIGIMON Documentation Index

**Canonical snapshot date:** 2026-09-16

This directory contains both current architecture documentation and older research/planning material. The files listed under **Canonical documentation** are the source of truth for the public snapshot. Older reports, checkpoint plans, handoffs, and exploratory notes remain useful as project history but must not override the canonical set.

## Canonical documentation

1. **[CURRENT_STATE.md](CURRENT_STATE.md)** — what the codebase actually contains and how complete each major capability is.
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** — the target harness-first architecture and design boundaries.
3. **[GAP_ANALYSIS.md](GAP_ANALYSIS.md)** — concrete gaps between the present code and target architecture.
4. **[ROADMAP.md](ROADMAP.md)** — architecture-completion sequence and exit criteria.
5. **[../FUNCTIONALITY.md](../FUNCTIONALITY.md)** — concise capability-oriented description of the implemented system.
6. **[QUICK_START.md](QUICK_START.md)** — current public-snapshot setup and entry points.
7. **[AGENT_INTELLIGENCE_ENHANCEMENTS.md](AGENT_INTELLIGENCE_ENHANCEMENTS.md)** — reasoning-policy detail: harness-first orchestration and AoT/GoT as a heuristic rather than a programmed brain.
8. **[FUTURE_EVALUATION_QUESTIONS.md](FUTURE_EVALUATION_QUESTIONS.md)** — deliberately deferred benchmarking, ablation, and research questions.

## Architecture decisions

- **[adr/002-harness-first-capability-architecture.md](adr/002-harness-first-capability-architecture.md)** — current decision: DIGIMON owns capabilities/contracts/evidence boundaries; the external harness owns adaptive orchestration.
- `adr/001-agent-orchestration-architecture.md` is retained as a superseded decision record documenting the earlier dual-brain design.

## Documentation status vocabulary

Canonical status documents use four labels consistently:

- **Implemented** — meaningful code exists and is wired into a current execution surface.
- **Partial** — meaningful code exists, but integration, consistency, contracts, lifecycle, or reliability remain incomplete.
- **Legacy** — code/documentation remains in the repository for compatibility or history but is not the preferred target architecture.
- **Planned** — target behavior is not materially complete in this public snapshot.

`Implemented` does **not** mean that every provider/environment combination was executed during the documentation review. Where live runtime verification was not performed, the status documents say so explicitly.

## Current architectural reading order

For a new contributor or reviewer:

```text
README.md
   ↓
docs/CURRENT_STATE.md
   ↓
docs/ARCHITECTURE.md
   ↓
docs/GAP_ANALYSIS.md
   ↓
docs/ROADMAP.md
```

Use `FUNCTIONALITY.md` when you want a shorter capability inventory and `QUICK_START.md` when you want to run one of the current entry points.

## Historical and supporting material

The repository contains substantial earlier planning and research material, including MCP checkpoint plans, UKRF-era plans, older agent-intelligence proposals, implementation reports, and handoffs. These files document the project's evolution and may contain still-useful ideas, but many describe architectures or priorities that have since changed.

In particular, old documents that prescribe:

- a hand-built cognitive/agent state machine,
- mandatory AoT/Markov preprocessing,
- a WebSocket MCP checkpoint sequence,
- multi-agent coordination as the immediate priority, or
- benchmark/production targets as the current implementation priority

should be treated as **historical unless restated in the canonical documentation**.

## Maintenance rule

When the architecture changes, update the canonical set in this order:

1. record the architectural decision in `docs/adr/` when appropriate;
2. update `CURRENT_STATE.md` to reflect code reality;
3. update `ARCHITECTURE.md` if the target design changed;
4. reconcile `GAP_ANALYSIS.md`;
5. update `ROADMAP.md` exit criteria/priorities;
6. update `README.md`, `FUNCTIONALITY.md`, `AGENTS.md`, and `CLAUDE.md` if user-facing or agent-facing guidance changed.

Do not create a new status document that competes with this hierarchy.