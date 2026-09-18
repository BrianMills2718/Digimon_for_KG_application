# Foundation projection batches 0–1: execution receipt

**Plan:** [North-star batch-and-converge plan](../planning/NORTH_STAR_VERTICAL_SLICE_PLAN.md), revision 2.  
**Upstream base:** `8ea142d36b642a26c2baf35bf10ad6ed29f4ed34`.  
**Disposition:** focused baseline repaired and saved-project increment executed. This is an evidence checkpoint under the existing plan, not a new roadmap or a claim that the full north-star outcome is complete.

## Observed result

The five explicitly selected test files below passed: **45 passed, zero skipped**, in **2.42 seconds** in the final recorded run. Compilation/import checks passed. A separate command-line build and a subsequent fresh-process `--reuse` both returned the same two exact assertion/passage results for the synthetic Alice example.

The working path is now:

```text
Foundation assertions + passage companion
→ validated identities, source scope, and both file digests
→ SQLite + assertion GraphML + binary entity GraphML
→ saved generation + validated manifest
→ fresh-process reopening
→ exact SQL entity/assertion/provenance/passage join
→ persisted evidence result + execution/artifact lineage
```

Graphs are built, saved, and checked for source-field reconstruction. They are **not yet adopted by the existing graph retrieval runtime**. Vectors, the progressive-disclosure catalog, analytic composition, and the external harness journey remain unintegrated and are explicitly listed as such in the project manifest.

## Execution environment and scope

Direct Git/DNS access and package installation were unavailable in this session. Selected repository files were transferred through the GitHub connector and checked against their Git blob hashes before local execution. This was a **scoped source snapshot**, not a complete clone or a clean installation of `requirements-minimal.txt`.

Observed environment: Python 3.13.5; SQLite 3.46.1; NetworkX 3.6.1; pytest 9.0.2; pytest-cov 7.0.0; pytest-asyncio 1.3.0; pydantic 2.13.4; loguru 0.7.3.

The normal repository `pytest.ini` and root fixture hooks were retained. No `--noconftest`, skipped projection checks, or blanket dependency stubs were used. Provider/orchestrator imports were moved into the fixtures that actually require them, so pure projection tests do not import the unrelated legacy LLM stack during collection. Tests using those fixtures still load their real dependencies.

```bash
python -m compileall -q Core/Projection scripts/run_foundation_demo.py
python -m pytest \
  tests/core/test_foundation_ir_contract.py \
  tests/core/test_relational_projection_contract.py \
  tests/core/test_foundation_property_graph_projection.py \
  tests/core/test_projection_boundary_repairs.py \
  tests/core/test_foundation_project.py -q
```

The recorded final run also wrote a JUnit XML receipt. Passing these selected tests is not a claim that all of `tests/core`, the complete dependency installation, MCP canaries, CI, provider calls, or other supported Python versions passed.

## Failure-driven changes

The first collection attempt exposed eager optional-runtime fixture imports. After their isolation, the 19 existing projection checks passed. Fifteen additional counterexamples then failed before repairs; the repaired baseline passed 34 checks. Additional persistence/lineage checks brought the final selected suite to 45.

Repairs include:

- **Rebuild safety:** SQLite is constructed and checked in a separate same-directory file, then published. An injected serialization failure leaves the previous database byte-for-byte intact. A failed whole-project generation or pre-publication validation likewise leaves the previous active manifest usable.
- **Producer compatibility:** `content_hash` remains opaque producer metadata rather than an invented SHA-256 requirement. Empty optional text is preserved; actual source and companion file digests remain separately verified SHA-256 values.
- **Evidence scope:** candidate-reference closure alone is insufficient. Assertion and companion namespace/registry scope must match; mismatches fail explicitly.
- **Field preservation:** the assertion graph retains occurrence-local filler data and empty/null/missing distinctions. A reconstruction check executes after real GraphML write/reload. SQLite retains the original assertion payload alongside normalized query tables.
- **Identity/error cases:** colliding entity/assertion or projection-local graph IDs fail instead of overwriting nodes. Duplicate reference values no longer cause SQL uniqueness failures. Nonfinite JSON/confidence is rejected. Imported nested source data is detached from the caller's mutable input.
- **Freshness:** both input files are fingerprinted, companion sidecars are checked, stale input expectations and changed artifact bytes are rejected, and manifest paths cannot escape the project directory.

The graph's source-field roundtrip is a bounded supported-input check, not certification of every possible IR shape or arbitrary edited graph. Collision cases currently fail closed rather than receiving automatic reminted IDs.

## Added callable and command surface

`Core/Projection/Project.py` exposes `build_foundation_project(...)`, `FoundationProject.open(...)`, and `evidence_for_entity(...)`. It reuses the existing Foundation, SQLite, and graph projectors rather than adding a second retrieval stack. `Core/Projection/Execution.py` supplies a local artifact-reference and execution-record helper.

Successful material transformations have their own execution IDs, input references, implementation digest, and actual output digests. Failed operations advertise no successful outputs; their local input generation remains an inspectable reproducer. Trace events and retained analytical lineage share identity but are not claimed to be a complete provenance platform.

```bash
python scripts/run_foundation_demo.py \
  --ir tests/fixtures/foundation_demo/foundation.json \
  --passages tests/fixtures/foundation_demo/passages.json \
  --output /tmp/digimon-foundation-demo \
  --entity-id entity:alice

# Run the same command with --reuse in a new process.
```

Use a new output directory for the first command. The checked-in example is explicitly **synthetic regression data**, not a production onto-canon export. A missing entity or unavailable evidence yields an explicit status rather than a fabricated answer. This narrow command does not modernize the old planner-coupled CLI.

## Upstream compatibility probe

The actual onto-canon exporter was inspected at blob `9d86393a35c1c135be1ebab7c5670e6fa157463a`. Its static compatibility fixture at blob `a4038fdbdb7c661448a3f224ca49cae8ca49f9ca` was transferred byte-for-byte and executed locally through import, assertion-graph reconstruction, and SQLite identity/alias/payload checks.

That probe used the fixture's bare assertion wrapped in the documented 1.3 envelope. It did **not** run onto-canon extraction/export against a real corpus and had no production passage companion. The private upstream fixture was not copied into this public repository.

## Measured implementation surface

Compared with the byte-verified selected baseline, authored Python implementation/tests changed by **844 additions and 46 deletions**: 890 changed lines, net 798. Generated data, fixture JSON, documentation, transferred unchanged files, coverage output, and repeated intermediate rewrites are excluded. An achieved hourly authoring rate was not measured; this does not certify the experimental 1,000-LOC/hour target.

## Continuation

The original plan's Batch 0 initial snapshot is now historical for these selected files. The baseline and saved-project checks above are observed; the real-corpus and whole-repository gates remain open. Continue with **Batch 2: feed the existing binary projection into the maintained graph retrieval consumer**, preserving parallel assertion identity and evidence, and rerun this same growing command path. Batch 3 connects actual vectors through the existing provider/index seam. Do not treat saved GraphML or an embedding-document list as consumer adoption.

Keep the broader Represent/Retrieve/Analyze, wiki/catalog, identity, harness boundary, and derivation goals unchanged. No new deployment, spending, upstream semantic authority, or production-readiness claim is introduced.
