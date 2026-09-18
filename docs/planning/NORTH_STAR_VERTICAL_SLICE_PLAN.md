# DIGIMON North-Star Vertical Slice — Batch and Converge

**Revision:** 3 — execution checkpoint after Batch 2; outcome and operating model unchanged.  
**Planning path:** durable_solo, one writer, reversible local implementation.  
**Authority:** [VISION](../VISION.md), [ARCHITECTURE](../ARCHITECTURE.md), [ROADMAP](../ROADMAP.md), and the contributor's batch-generation/trace-driven-repair direction.  
**Consumer:** the contributor and the next execution session. This remains the sole active execution plan; reports below are evidence, not competing priorities.  
**Method record:** [PlanningPathDecisionV1](supporting/north-star-speedrun-path.json).

## Outcome And Boundaries

For a researcher/analyst using governed text-derived knowledge, produce one inspectable project where an external harness navigates content and representation schemas, retrieves across complementary structures using canonical IDs, analyzes a bounded working set, and recovers source evidence and derivation.

The full thesis remains **Represent → Retrieve → Analyze/Transform → grounded findings → action**, downstream of onto-canon6. The first integrated prototype demonstrates the internal-product direction; tests alone do not promote maturity or redefine the full product as a demo.

**Canonical probe:** one real, authorized Foundation export plus exact passage companion → SQLite, graph, vector, and Markdown catalog → native harness discovery → exact SQL and aggregation → graph neighborhood and real vector query → Leiden/centrality on a selected view/subgraph → selected entities → exact evidence → inspectable finding and lineage. The harness chooses its sequence. A fixed script is regression evidence, not autonomous composition; a real harness trace must also show a cross-representation move and an analytic-result-to-evidence move.

**Review surface:** saved/reopenable project, generated catalog, actual outputs, source passages, finding, and execution/lineage records. No new UI server.

**Negative case:** missing/partial evidence, unknown identity, or stale projection is explicit, never fabricated source or a plausible unsupported finding.

**Non-claims:** no general benchmark superiority, complete method/IR support, causal/social influence from centrality, or production readiness.

**Limits:** no deployment, canonical-store migration, upstream semantic-contract mutation, new provider spend, or public release of private exports/traces. The repository is public; source-bearing real artifacts stay in authorized local storage. Synthetic fixtures are labeled.

## Operating Model And Surface

Target approximately **1,000 authored code/test LOC per active authoring hour** experimentally. It is not a quota, observed rate, correctness claim, or delivery promise. Prefer fewer lines that close the same boundary. Keep authoring, execution, repair, and transfer/access overhead separate.

Original remaining-slice allowance at revision 2: **3,000–5,000 additional authored implementation/test LOC**, excluding generated data/wiki pages, formatting churn, planning prose, and repeatedly rewritten lines. Re-estimate from actual integration failures rather than treating the allowance as a ceiling. Earlier calendar guesses do not schedule the frontier.

```text
brief expected behavior → coherent batch, usually 500–1,500 code/test lines
→ compile/import + focused checks → same growing integration journey
→ inspect first divergence and smallest reproducer → repair → cumulative rerun
→ record exact evidence and commit coherent change
```

Do not stack dependent unexecuted batches. Independent work may proceed only against a stable interface with its own execution. Collect related failures under a costly shared setup before repeating it. Final integration is not deferred until the end.

## Architecture And Capability Invariants

- onto-canon6 owns meaning, governance, identity, ontology binding, and canonical export. DIGIMON does not re-extract or recanonicalize governed inputs. Raw ingestion/chunking is standalone compatibility, not the ecosystem center.
- Preserve entity/assertion/passage IDs and roles/literals/qualifiers/aliases; bind derived artifacts to both input files and namespace/registry scope. Matching counts/shapes is not freshness.
- Complementary representations include relational, graph, vector, lexical where useful, semantic/RDF where useful, hierarchy/tree, evidence, and wiki. Projection-local state remains downstream.
- N-ary structure is retained in the assertion graph. No implicit clique expansion or direction from dictionary order. The binary runtime view is an explicit association projection with unit-per-pair weights and all parallel assertion records retained, not affirmative truth or confidence-weighted social ties.
- Analytics must declare predicate/qualifier selection, polarity treatment, direction, weights, parallel-edge policy, self-loops, and whole-graph versus subgraph domain. Derived scores/communities/findings are not original evidence or causal claims.
- Reuse maintained storage, typed operators, indexes, and analytic engines. A separate demo-only retrieval stack is not adoption. Use existing SUBGRAPH/SCORE_VECTOR/COMMUNITY_SET semantics rather than cognitive-state types.
- The harness owns planning, sequencing, adaptation, retries, and stopping. Reuse its file navigation, ordinary search, and SQL tooling. Add only missing specialized access; a narrow callable/CLI bridge suffices. MCP is optional transport, not the product.
- The wiki is a progressive-disclosure semantic **and operational** map: content, actual representations, schemas/ontology references, cross-representation join keys, capabilities, availability, and evidence. It describes choices, not a mandatory workflow. Observed vocabulary is not a complete supplied ontology.
- Keep source-backed state, retrieved working sets, analytics, and findings distinct. Distinguish evidence provenance, semantic provenance, and derivation lineage; domain and provenance graphs are different.
- Begin lightweight execution/lineage with the first material artifact. Record exact inputs/outputs, method/implementation, parameters/seed/model where applicable, omissions, and failures. Do not fabricate observations from native tools outside the recorded path.
- Publish replacements only after validation; failed generation preserves prior usable outputs. No telemetry platform, generic lifecycle service, new internal brain, or geospatial expansion.

## Milestone Horizon

Ranges are the revision-2 allowances, not freshly claimed measured effort. Later details remain reversible until observed integration fixes them.

| Batch | Inspectable boundary | State after Batch 2 | Original additional LOC |
|---|---|---|---:|
| 0 | Execute and repair IR/SQLite/graph/evidence baseline | Focused checks observed; real-corpus and full-install evidence open | 300–600 |
| 1 | One saved/reopened project, growing runner, minimal trace/lineage | Observed in Batch 01 receipt | 400–700 |
| 2 | Saved graph → maintained storage/operators → bounded subgraph → exact evidence | Observed in Batch 02 receipt; not global MCP/all-method registration | 500–800 |
| 3 | Real entity/assertion/passage embedding/index query and reload | Next; environment/provider route not verified | 400–700 |
| 4 | Progressive-disclosure catalog linking real content, schemas, and resources | Conditional on actual representation metadata | 500–800 |
| 5 | Selected subgraph → Leiden/centrality → entity IDs → evidence; SQL aggregation | Conditional on analytic adapter/engine availability; graph input now real | 500–800 |
| 6 | Clean/reuse/failure checks plus actual external-harness finding/journey | Conditional on 3–5; extends same integration runner | 400–600 |

Graph/vector lanes were independent after the shared project seam; analytics may follow graph before catalog polish. One writer remains the default. Introduce coordination only for actual concurrent writers/shared mutation, not imagined parallelism.

## Evidence And Current State

- [Batch 01 receipt](../reports/PROJECTION_BATCH_01.md): repaired baseline and saved-project path; 45 selected tests passed, exact SQL evidence, safe rebuild, source/GraphML checks. Scoped source snapshot, not full repository certification.
- [Batch 02 receipt](../reports/PROJECTION_BATCH_02.md): **69 selected tests passed, zero skipped**, final run **5.58 seconds**. Real NetworkXStorage/OperatorContext, existing khop/materialize functions, parallel assertion retention, filters, one/two/three hops, exact source bytes/scope, partial-evidence reporting, fresh-process CLI reuse, and injected failure/recovery were exercised.
- Batch 2 authored Python: **636 additions / 60 deletions; 696 changed / 576 net lines**. This excludes docs, generated outputs, unchanged transferred modules, and repeat edits. Isolated authoring throughput was not measured.
- Observed implementation base for Batch 2: `2b2a35322187951008759cb33ec127c055076225`; the receipt and committed file hashes bind the result. Successful graph adoption is through the narrow local Python/CLI operator path. Broader GraphRAGContext/MCP, reference methods, vectors, analytics, and autonomous harness observation remain unproved.
- Direct Git DNS failed. Execution used the mounted byte-verified scoped snapshot plus pinned connector-transferred dependencies, normal pytest configuration/root fixtures, and no dependency stubs. This is real bounded execution, not a complete checkout/install.

## Active Slice — Batch 3

**Visible result:** semantic vector retrieval over the **same** saved Foundation project, with persisted reload and canonical identities in the returned records.

**Reuse:** current Foundation records/identity, Project/Execution artifacts, and the existing embedding/provider/index owner. Inspect the actual VDB interface before adding an adapter. Document lists and hash-embedding stand-ins cannot satisfy semantic retrieval.

**Likely surfaces:** narrow `Core/Projection/Vector.py`, current VDB/provider seams, the existing growing runner, focused vector tests, and this plan's evidence checkpoint. Do not invent a new vector service, router, or planner.

**Checks:** entity/assertion/passage keys preserved; nonempty correctly rendered input text; collection/model/dimension/metric and both source digests recorded; actual index query; source-linked results; save/reload; incompatible or stale index rejected; empty/missing resources explicit. Controlled deterministic fixtures may test plumbing, but label them separately from a genuine embedding/index run.

**Known prerequisite:** FAISS/llama-index/provider packages were absent from the scoped Batch 2 runner. Recheck available authorized execution routes and configured embedding access. No new provider spending is authorized by this plan. Missing access is an evidence/operational gate, not a reason to write the remaining dependent features source-only.

## Observability And Acceptance

A small local JSONL stream plus referenced artifacts is enough. Instrument boundaries, not every function. Record start/terminal status, exact revision/input/output digests, parameters, relevant counts/selected IDs, omissions, expected/actual mismatch and minimal reproducer. Bound diagnostic samples and do not dump corpus text or credentials into logs.

**Traces explain what happened; independent checks test correctness.** Use exact ID/SQL joins, producer fixtures, field reconstruction, understandable graph oracles/library results, and intentionally broken inputs. Self-reported success flags are insufficient. Diagnostic logs and retained analytic lineage have different retention/meaning even when they share execution IDs.

Integration continues to require identity across SQL/graph/vectors/catalog, bounded retrieval, valid analytic output domains and finite scores, exact evidence, real artifact parentage, safe clean/reuse/failure behavior, and an actual harness journey. Existing affected core tests/MCP canaries remain regression obligations; missing dependencies/credentials/skipped suites remain visible.

## Decisions, Assumptions, And Human Gates

**Human-set:** the full text-derived Represent/Retrieve/Analyze vision; onto-canon upstream authority; harness-native reuse; approximately 1,000-LOC/hour authoring target.

**Reversible implementation choices:** SQLite first; local manifest/JSONL; one growing runner; a simple association view retaining all parallel records over the unchanged MultiGraph; lazy public operator imports.

**Still unresolved:** a representative real export/companion; configured embedding route and suitable dependencies; complete ontology context in upstream export; broader method/runtime compatibility; observed throughput; external-harness usefulness. These are evidence questions, not grounds to reopen the accepted vision.

**Human-required only when activated:** new spend, public data release, upstream authority change, deployment, or consequential migration. No new human decision is required for ordinary authorized code/test continuation.

## Later Horizon And Course Correction

Retain RDF/SPARQL, specialized lexical/BM25 beyond native search, hierarchy/tree, broader graph/SNA and non-graph methods, all-reference-method coverage, richer derivation queries/invalidation, Python/CLI/MCP convergence, legacy retirement, and later comparative evaluation/scaling. None is erased by the first demonstration's scope.

Course-correct when supporting work repeats without a capability, repeated failures provide no new evidence, or the adapter becomes another platform. Return to the smallest authentic integration or demonstrated blocker; do not add a new roadmap/approval checkpoint. Change architecture only for a concrete counterexample. Every analysis should make the next one easier.

**Exact next action:** inspect the existing embedding/index consumer and obtain a suitable authorized runtime/configuration for Batch 3. Reuse the saved project and the 69-test cumulative baseline; produce a real query/reload receipt before advancing dependent vector/catalog claims.
