# DIGIMON Planning Summary

**Updated:** 2026-09-17 planning cycle, following projection batches 0–1.  
**Role:** short view of the authoritative roadmap and active execution plan; not a second schedule.

## Unchanged North Star

**Governed semantic IR → Represent → Retrieve → Analyze/Transform → grounded evidence/findings → action**, with shared canonical identity and derivation lineage.

onto-canon6 owns semantic authority and governance. DIGIMON owns complementary derived representations and specialized retrieval/analytics. The external harness owns planning, composition, adaptation and stopping; native file navigation, ordinary search and existing SQL tools are reused rather than reimplemented.

The wiki/catalog is a progressive-disclosure map of both semantic content and the operational environment: representations, actual schemas/ontology references, canonical IDs, capabilities, availability and source evidence. It describes possibilities rather than prescribing a retrieval sequence.

Analytics remain first-class. The representative workflow is retrieve a graph/subgraph, compute Leiden/centrality, use the output to retrieve again, and produce a finding with evidence and derivation. Analytical scores, communities and interpretations are derived state, not original evidence. The provenance graph is distinct from the domain graph.

## Active Execution Plan

Use [planning/NORTH_STAR_VERTICAL_SLICE_PLAN.md](planning/NORTH_STAR_VERTICAL_SLICE_PLAN.md), revision 2, for batch contracts, proposed file surfaces, acceptance checks, telemetry/lineage facts and unresolved assumptions. Its initial execution position is superseded by the [batches 0–1 receipt](reports/PROJECTION_BATCH_01.md); the current next action is below.

The contributor's approximately **1,000 authored code/test LOC per active authoring hour** is a target to measure, not a quota or correctness claim. The shortest integrated slice was provisionally **3,000–5,000 additional implementation/test lines** before this batch. Generated data/wiki pages, formatting churn and planning prose are excluded. Calendar guesses from earlier discussion no longer schedule the frontier.

The sequence remains **baseline execution → one saved/reopened project with trace/lineage → maintained graph retrieval → actual vector indexing/query → progressive-disclosure catalog → subgraph analytics and evidence → observed integrated harness journey**.

Graph and vector work are independent after the shared seam is stable. Analytics can follow graph integration before catalog polish. Every batch extends and reruns the same journey; final integration is not deferred until everything has been written.

## Current Execution Evidence

At upstream base `8ea142d36b642a26c2baf35bf10ad6ed29f4ed34`, selected projection files were transferred with matching Git blob hashes into a local runnable snapshot. The normal pytest configuration was retained; optional LLM/orchestrator imports now occur only inside fixtures that need them. After repairs and the saved-project increment, **45 selected tests passed with zero skips**. The last recorded focused run took 2.42 seconds. This was not a complete clone or a full `tests/core` run.

The working increment builds SQLite and two GraphML projections from Foundation assertions/passages, preserves source fields through a graph roundtrip, saves a generation with artifact/input hashes, reopens it in a new process, and returns exact SQL-backed source evidence. Failed rebuilds preserve the previous output. Execution records connect actual transformations and evidence results to saved artifacts. The change includes 844 authored Python additions and 46 deletions; achieved hourly throughput was not measured.

Still missing/unverified are maintained graph runtime adoption, vectors from the same IR, generated catalog, bounded analytic access, broader derivation queries, the real-corpus/harness workflow, full dependency installation and broad core/MCP regression. The saved manifest advertises those missing integrations explicitly. See the receipt for commands, producer compatibility probe, negative cases and limits.

## Delivery Rules

Generate a coherent batch, compile/import, run focused checks and the growing integration path, inspect the first divergence, repair and rerun, then record/commit a coherent result. Do not accumulate dependent unexecuted batches.

Use bounded machine-readable traces with exact revisions, input/output artifact identities, parameters, counts, omission reasons, expected/actual failures and a small reproducer. Traces explain execution; producer fixtures, exact joins/ID checks, graph oracles, and negative controls test correctness. Do not create a telemetry service.

Start minimal derivation records at the first artifact, rather than reconstructing lineage later. Keep diagnostic logs and retained analytical lineage distinct. Preserve prior usable outputs when rebuilds fail.

Keep fresh focused checks, broad deterministic regression, provider-dependent canaries, and stakeholder observation separate. A fixed demo script, fake embedding, or test-file presence does not prove the authentic outcome.

## What Remains Beyond The First Proof

The [ROADMAP](ROADMAP.md) retains RDF/semantic graphs, hierarchy/tree projections, specialized lexical/BM25 where native search is insufficient, broader graph/SNA and non-graph analytical methods, remaining reference-method coverage, incremental/resource correctness, derivation queries, Python/CLI/MCP convergence, legacy cleanup and later comparative evaluation.

Raw-source chunking/custom ontology remains standalone compatibility work, not the canonical ecosystem input. Geospatial, an internal agent brain, extra UI shells, generic multi-agent infrastructure and production scaling are not first-proof requirements.

## Exact Next Action

Continue Batch 2: connect the existing binary entity projection to the maintained graph retrieval consumer, explicitly preserving parallel assertions, canonical IDs and evidence. Add that consumer's dependencies to the runnable snapshot or use an authorized complete checkout; a missing dependency is not permission to bypass the consumer with a demo-only stack. Rerun the five focused files in the receipt and extend the same saved-project command with an actual graph retrieval check.

No new human decision is required for that bounded integration. Real authorized input, embedding/provider availability, broader regression and observed throughput remain evidence questions. No new spending, public release of private data or deployment is authorized.
