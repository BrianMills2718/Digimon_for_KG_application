# DIGIMON North-Star Vertical Slice Plan

**Planning path:** durable_solo  
**Adopted:** 2026-09-17  
**Authority:** `docs/VISION.md`, `docs/ROADMAP.md`, `docs/CURRENT_STATE.md`  
**Method:** Company Planning — initiative-roadmap → bounded design → evidence-first implementation. No work-unit graph is required for the current single-contributor reversible path.

## Outcome

For a researcher/analyst using governed text-derived knowledge, change the current graph/vector-centered transitional runtime into an inspectable DIGIMON project that can consume onto-canon6 Foundation IR, preserve canonical identity, project the same governed semantic core into complementary representations, retrieve bounded working sets, run analytical transformations, and trace derived results back to source evidence.

## Maturity and investment boundary

Target this sequence as an **internal-product vertical slice**, not a production platform. Estimated focused engineering surface: roughly **4–7 weeks**, with a plausible **6–9 week** upper band if runtime or IR-compatibility failures are substantial.

Non-goals for this slice:
- production scaling or multi-tenant infrastructure;
- a new internal agent brain;
- generalized multi-agent coordination;
- geospatial representations;
- exhaustive statistical tooling;
- benchmark optimization before the vertical slice works;
- wrapping file open/search/wiki navigation already supplied by capable harnesses.

## Canonical outcome probe

Using one real Foundation IR export and its passage companion:

1. load the governed IR without re-extraction;
2. preserve entity/assertion/predicate/source/evidence identity;
3. materialize at least relational, property-graph, vector, and progressive-disclosure wiki/catalog projections;
4. discover one entity through the wiki/catalog;
5. query exact structured attributes through the relational projection;
6. retrieve a bounded graph neighborhood using the same canonical ID;
7. run at least one structural analytic transformation (Leiden and/or centrality);
8. recover exact supporting evidence;
9. inspect derivation lineage from source/IR → projection → retrieval → analytic result → finding.

**Inspectable artifact:** a small checked-in or reproducibly generated demo project containing the projection outputs, a script/test exercising the workflow, and a lineage record.

**Non-claim:** passing this probe does not establish production readiness, general benchmark superiority, or complete support for every representation/analytic family.

## Current truth

Already working source-side:
- typed retrieval/composition core and ten reference methods;
- five graph-build surfaces;
- graph/vector/community/sparse resources;
- grounded evidence recovery and fail-closed answer generation;
- practical graph-source manifests and derived-artifact invalidation;
- significant graph analytics/transforms (PPR, communities, PCST, Steiner, path/subgraph operations);
- documentation reconciled around Represent → Retrieve → Analyze/Transform.

Known missing boundaries:
- current-head runtime execution is not freshly certified;
- Foundation IR is not yet a canonical first-class DIGIMON input model;
- no canonical IR→relational projection;
- no progressive-disclosure wiki/catalog projection;
- analytics are not yet organized as a complete first-class plane;
- no first-class cross-representation derivation graph;
- CLI/Python/MCP do not yet converge on one maintained public runtime.

## Backward path and critical path

```text
observed mixed-method evidence-to-action workflow
  <- derivation lineage over a real analytic result
  <- analytic transformation over a retrieved working set
  <- graph/vector/relational/wiki projections sharing canonical IDs
  <- canonical DIGIMON semantic model loaded from Foundation IR
  <- fresh runtime proof that the maintained core still executes
```

The **first missing boundary** is fresh runtime proof. Because the available connected environment cannot currently execute the repository and connector-created commits are not triggering Actions, that boundary is externally blocked. The first implementation-ready boundary behind it is therefore **Foundation IR ingestion + canonical identity preservation**, which can proceed source-first with deterministic contract tests ready for the next runner.

## Risk-ordered stages and estimates

| Stage | Outcome | Estimate |
|---|---|---:|
| 1 | Fresh deterministic suite + MCP clean-rebuild canary; fix first real reds | 0.5–2 days |
| 2 | Foundation IR 1.3 + passage companion become canonical DIGIMON input records | 2–4 days |
| 3 | Cross-representation identity contract tested | 1–2 days |
| 4 | Canonical DuckDB/relational projection + schema/catalog metadata | 2–4 days |
| 5 | Existing graph/vector projections normalized around the same IR identities | 2–5 days |
| 6 | Static Markdown progressive-disclosure wiki/catalog projection | 3–6 days |
| 7 | Graph/SNA analytics promoted to first-class typed capability plane; minimal table analytics | 4–8 days |
| 8 | Artifact/derivation lineage for projection → retrieval → analysis → finding | 4–8 days |
| 9 | Python runtime, CLI, and MCP converge on the same maintained core | 3–6 days |
| 10 | Selective RDF/BM25/tree expansion where evidence shows value | 3–8 days/family |
| 11 | Integrated evaluation/hardening of the actual mixed-method thesis | 5–10 days |

## Execution frontier

### Active slice A — runtime proof
Blocked on an available runner. Required commands remain:
```bash
pip install -r requirements-minimal.txt
pytest tests/core -q
python tests/e2e/test_mcp_smoke.py
DIGIMON_CANARY_REBUILD=1 python tests/e2e/test_mcp_smoke.py
```
When available: first red → smallest fix → regression → rerun.

### Active slice B — Foundation IR seam
Proceed now:
- model the Foundation IR 1.3 envelope, assertions, role fillers, qualifiers, source attribution, and 1.0 passage companion;
- fail closed on unsupported format/producer or malformed identities;
- preserve unknown additive qualifier keys without interpreting them;
- expose canonical entity/assertion/source/evidence indexes useful to downstream projections;
- add deterministic contract tests using a representative fixture;
- do not project to graph/SQL/vector in this slice yet.

## Continuation / stop / reset triggers

**Continue** while each slice produces a directly inspectable artifact and reduces a known boundary.

**Change tactics** if Foundation IR cannot express a target projection without invented semantics; document the exact missing field and reconcile upstream rather than silently enriching it.

**Pause for material decision** only if a change would move semantic authority from onto-canon6 into DIGIMON, introduce an irreversible public contract, or require a new external dependency with significant operational cost.

**Reset** if runtime evidence shows the maintained typed core is fundamentally inconsistent with the documented current state; repair runtime truth before expanding representation families.

## Next action

Implement Active slice B now while keeping Active slice A visibly blocked on a real runner.

## Progress log

### 2026-09-17 — Foundation IR seam implemented source-side

Implemented:
- Core/Projection/FoundationIR.py — strict Foundation IR 1.3 and passage 1.0 consumer;
- Core/Projection/__init__.py — projection-layer public exports;
- tests/core/test_foundation_ir_contract.py — deterministic contract coverage prepared for the next runner;
- exact producer/version/count validation;
- canonical assertion/entity IDs preserved verbatim, including Unicode;
- value fillers preserved for later relational/graph projections;
- additive qualifier keys preserved without DIGIMON interpreting upstream semantic policy;
- assertion/entity/provenance/passages indexes;
- assertion-to-passage lookup through candidate provenance refs;
- snapshot SHA-256 recomputation and optional sidecar verification;
- passage companion closure: every selected assertion provenance ref must resolve and no orphaned companion provenance refs are accepted.

Evidence:
- planning commit 0b05893177f5c1360abbdac8bbbd9915ce918da5
- projection package commit 89f5c39d2aeeda1619671ac70135793db64b2d56
- consumer commit 6fa83beaf23887c247cbb70c33f36fe1f6473df0
- contract-test commit 05a5913001330f64737b5b64ce691ef0141651e6
- provenance-closure commits b47efd6802873043b644eea730e66fc0291038ec and 0d146615a8b00789a584d3e4df595c2300dd0358

Verification boundary:
- source reviewed only;
- tests have not executed in the available environment;
- do not promote this slice to runtime-certified until the Stage-1 runner commands pass.

Next implementation-ready slice:
- make the cross-representation identity contract explicit over FoundationIR before building the first relational projection.


### 2026-09-17 — Identity + first projections implemented source-side

Cross-representation identity:
- Core/Projection/Identity.py defines the immutable projection identity manifest.
- entity/assertion/predicate/provenance/passage/source/namespace/registry IDs are exposed without reminting.
- contract test locks exact ID reuse.

Relational projection:
- Core/Projection/Relational.py materializes Foundation IR into normalized SQLite with no new dependency.
- tables preserve entities, names, types, aliases, assertions, n-ary roles, qualifiers, provenance, source URLs, passages, and passage support.
- relational_schema_manifest() provides agent/catalog-facing schema descriptions.
- contract test proves canonical entity lookup → assertion role → provenance → exact passage join.
- SQLite is intentionally the first proof backend; DuckDB can be added later if analytical workload evidence justifies the dependency.

Property graph bounded design:
- docs/planning/FOUNDATION_PROPERTY_GRAPH_DESIGN.md records the n-ary mapping decision.
- canonical assertion graph = lossless MultiDiGraph with assertion nodes and role edges.
- retrieval entity graph = binary-only undirected MultiGraph; non-binary assertions are reported as skipped rather than clique-expanded.
- parallel assertions preserve assertion identity through MultiGraph edge keys.
- these pure projectors are not yet wired into GraphRAGContext/reference methods.

Verification boundary remains source-review only. No new runtime-green claim is made.

Next boundary:
- execute the prepared contract tests when a runner becomes available;
- then adapt the binary entity projection into the maintained ER runtime only after defining the exact MultiGraph→current simple-Graph merge behavior, or evolve the runtime storage deliberately.
