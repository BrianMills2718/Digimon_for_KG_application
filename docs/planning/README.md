# DIGIMON Planning

**Updated:** 2026-09-16

This directory contains multiple generations of planning material. The active rapid-implementation plans are:

1. [CORE_CANARY_PLAN.md](CORE_CANARY_PLAN.md) — establish reproducible core MCP behavior; fix only observed failures.
2. [CI_RECOVERY_PLAN.md](CI_RECOVERY_PLAN.md) — restore a small meaningful automated health signal.
3. [FAILURE_DRIVEN_CORE_FIXES_PLAN.md](FAILURE_DRIVEN_CORE_FIXES_PLAN.md) — drive subsequent fixes from canary/CI failures.

These execution stubs follow the repository planning convention: **goal → evidence → checkpoints → tests → success criteria → execution order/stop rule**.

They are subordinate to the canonical architecture/status documents in `docs/`, especially `CURRENT_STATE.md`, `ARCHITECTURE.md`, and `ROADMAP.md`.

Older files in this directory document project history unless a canonical document or one of the active plans above explicitly restates the work.

## Execution rule

Prefer implementation over plan expansion. Update a plan only when a result changes the next action; do not add process artifacts unless they remove real ambiguity.