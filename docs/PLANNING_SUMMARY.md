# DIGIMON Planning Summary

**Updated:** 2026-09-17  
**Role:** short view of the authoritative roadmap and active execution plan; not a second schedule.

## Unchanged North Star

**Governed semantic IR → Represent → Retrieve → Analyze/Transform → grounded evidence/findings → action**, with shared canonical identity and derivation lineage.

onto-canon6 owns semantic authority and governance. DIGIMON owns complementary derived representations and specialized retrieval/analytics. The external harness owns planning, composition, adaptation and stopping; native file navigation, ordinary search and existing SQL tools are reused rather than reimplemented.

The wiki/catalog is a progressive-disclosure map of both semantic content and the operational environment: representations, actual schemas/ontology references, canonical IDs, capabilities, availability and source evidence. It describes possibilities rather than prescribing a retrieval sequence.

Analytics remain first-class. The representative workflow is retrieve a graph/subgraph, compute Leiden/centrality, use the output to retrieve again, and produce a finding with evidence and derivation. Analytical scores, communities and interpretations are derived state, not original evidence. The provenance graph is distinct from the domain graph.

## Active Execution Plan

Use [planning/NORTH_STAR_VERTICAL_SLICE_PLAN.md](planning/NORTH_STAR_VERTICAL_SLICE_PLAN.md), revision 2, for batch contracts, proposed file surfaces, acceptance checks, telemetry/lineage facts, unresolved assumptions and the exact next action.

The contributor's approximately **1,000 authored code/test LOC per active authoring hour** is a target to measure, not a quota or correctness claim. The shortest integrated slice is provisionally **3,000–5,000 additional implementation/test lines**. Generated data/wiki pages, formatting churn and planning prose are excluded. Calendar guesses from earlier discussion no longer schedule the frontier.

The next sequence is **baseline execution → one saved/reopened project with trace/lineage → maintained graph retrieval → actual vector indexing/query → progressive-disclosure catalog → subgraph analytics and evidence → observed integrated harness journey**.

Graph and vector work are independent after the shared seam is stable. Analytics can follow graph integration before catalog polish. Every batch extends and reruns the same journey; final integration is not deferred until everything has been written.

## Current Baseline

At inspected base `d548f0c1f84e3450c252eedea8f3a02abf0f3520`, `Core/Projection/` already contains the Foundation IR consumer, shared identity manifest, normalized SQLite projection and two graph projectors. Their source and tests exist; this planning revision does not claim those tests passed or that the projectors are adopted by maintained runtime consumers.

Still missing/unverified are actual execution, graph runtime adoption, vectors from the same IR, generated catalog, bounded analytic access, persisted artifact/execution lineage and the real harness workflow. The older “no runner” status must be checked against actual available tools rather than carried forward indefinitely. Local Python was available in this session; a direct Git request failed DNS, and no DIGIMON tests ran.

## Delivery Rules

Generate a coherent batch, compile/import, run focused checks and the growing integration path, inspect the first divergence, repair and rerun, then record/commit a coherent result. Do not accumulate dependent unexecuted batches.

Use bounded machine-readable traces with exact revisions, input/output artifact identities, parameters, counts, omission reasons, expected/actual failures and a small reproducer. Traces explain execution; producer fixtures, exact joins/ID checks, graph oracles, and negative controls test correctness. Do not create a telemetry service.

Start minimal derivation records at the first artifact, rather than reconstructing lineage later. Keep diagnostic logs and retained analytical lineage distinct. Preserve prior usable outputs when rebuilds fail.

Keep fresh focused checks, broad deterministic regression, provider-dependent canaries, and stakeholder observation separate. A fixed demo script, fake embedding, or test-file presence does not prove the authentic outcome.

## What Remains Beyond The First Proof

The [ROADMAP](ROADMAP.md) retains RDF/semantic graphs, hierarchy/tree projections, specialized lexical/BM25 where native search is insufficient, broader graph/SNA and non-graph analytical methods, remaining reference-method coverage, incremental/resource correctness, derivation queries, Python/CLI/MCP convergence, legacy cleanup and later comparative evaluation.

Raw-source chunking/custom ontology remains standalone compatibility work, not the canonical ecosystem input. Geospatial, an internal agent brain, extra UI shells, generic multi-agent infrastructure and production scaling are not first-proof requirements.

## Exact Next Action

Run the three existing Foundation/relational/property-graph contract test files listed in Batch 0 of the living plan on a usable checkout. Record the exact revision/environment and the first failure or scoped pass. Only then extend the growing demo.

No human decision is required to begin that baseline. Runtime access, real authorized input, embedding/provider availability and observed throughput are still evidence questions. This planning update does not authorize new spending, public release of private data or deployment.
