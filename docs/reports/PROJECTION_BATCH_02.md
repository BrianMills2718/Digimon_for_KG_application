# Foundation graph runtime: Batch 2 execution receipt

**Parent:** [North-star batch-and-converge plan](../planning/NORTH_STAR_VERTICAL_SLICE_PLAN.md).  
**Inspected base:** `2b2a35322187951008759cb33ec127c055076225`.  
**Disposition:** maintained graph storage and typed operators now consume the saved Foundation projection through a narrow local adapter. This is not full reference-method, MCP, vector, analytics, or autonomous-harness certification.

## Observed path

```text
saved Foundation assertions + passage companion + binary GraphML
→ checked snapshot/artifact hashes
→ explicit simple entity-association view, retaining parallel assertion records
→ real NetworkXStorage and OperatorContext
→ existing subgraph.khop_paths
→ bounded typed SUBGRAPH + persisted GraphML
→ existing subgraph.materialize
→ exact source passages + selected assertion identities + saved execution lineage
```

The original assertion graph and binary MultiGraph are unchanged. SQL can retrieve an n-ary assertion that the binary graph deliberately omits. Graph-selected evidence is restricted to selected edges, not every claim incident on any selected node.

## Checks actually executed

The seven selected files below passed: **69 passed, zero skipped**, in **5.58 seconds** in the final recorded focused run. The prior 45-test baseline was rerun successfully before implementation. One new materializer test failed first because the existing operator trimmed original passage whitespace; it passed after repair.

```bash
python -m compileall -q Core/Projection Core/Operators Core/Storage scripts/run_foundation_demo.py
python -m pytest \
  tests/core/test_foundation_ir_contract.py \
  tests/core/test_relational_projection_contract.py \
  tests/core/test_foundation_property_graph_projection.py \
  tests/core/test_projection_boundary_repairs.py \
  tests/core/test_foundation_project.py \
  tests/core/test_graph_materialize_exactness.py \
  tests/core/test_foundation_graph_runtime.py -q
```

The tests use actual NetworkXStorage, OperatorContext, khop/materialize functions, SQLite, GraphML, files, and fresh subprocesses. No provider, storage, or retrieval mocks substitute for the successful integration path. Intentional injected failures test error handling. Synthetic source fixtures are explicitly regression data, not production onto-canon exports.

Observed cases include parallel assertions, same label/different canonical IDs, one/two/three hops, pre-aggregation predicate filtering, n-ary omissions, isolates, self-loops, opaque passage IDs containing the legacy separator, missing/partial evidence, invalid hop counts, unknown seeds, changed artifact bytes, and a deliberately failing traversal. Failure events advertise no successful retrieval output. The unchanged saved project remains queryable after the injected failure is removed.

Separate build and fresh-process `--reuse` runs of the growing demo returned identical graph evidence. A richer synthetic example retrieved three assertions over A–B–C at two hops; a predicate-filtered query retained just the matching assertion from a parallel pair.

## Semantics and compatibility

- The retrieval view is an **undirected association between binary assertion participants**, not an inferred affirmative truth graph. Polarity/qualifiers remain in the retained assertion records; they are not silently interpreted.
- One edge per entity pair has unit weight. This is not confidence, assertion frequency, or an automatic social tie-strength measure. Self-loops are retained and declared. Future analytics must use these declared semantics or request another explicit view.
- Each aggregated edge retains every assertion ID, predicate/role/provenance record, and exact passage ID. Predicate filtering happens before aggregation. N-ary assertions are not clique-expanded.
- Canonical IDs, not display labels, populate the maintained `entity_name` key. Separate display names are retained. No recanonicalization occurs.
- The materializer preserves original text and available source scope. Non-text payloads are no longer stringified into evidence. Explicit passage ID arrays take precedence over delimiter-separated compatibility fields.
- Partial evidence is explicitly reported with `assertions_without_passages`; the CLI does not report that as complete success.
- Existing public operator exports now resolve lazily, preserving names while avoiding eager unrelated vector/LLM imports for direct graph use. The khop operator and storage use the same loguru logger object without forcing legacy global configuration initialization. No dependency stubs or `--noconftest` bypass are introduced.

## Public local surface and lineage

`Core/Projection/GraphRuntime.py` exposes `open_foundation_graph_context(...)` for direct composition with maintained operators and `retrieve_foundation_subgraph(...)` for typed SUBGRAPH/ENTITY_SET/CHUNK_SET outputs. `SUBGRAPH.nx_graph` holds the selected attributed graph for subsequent analytics.

`FoundationProject.graph_neighborhood(...)` returns the persisted JSON result. The growing CLI adds `--graph-hops` and optional repeated `--predicate` arguments:

```bash
python scripts/run_foundation_demo.py \
  --ir tests/fixtures/foundation_demo/foundation.json \
  --passages tests/fixtures/foundation_demo/passages.json \
  --output /tmp/digimon-foundation-graph-demo \
  --entity-id entity:alice --graph-hops 2
# Repeat with --reuse in another process.
```

Each runtime view, subgraph, and evidence result has a content hash and actual producing execution. Records link input graph → selected subgraph → evidence, as well as the source/IR artifacts inherited from Batch 1. Consumer implementation hashes supplement the projection implementation fingerprint. This is local execution lineage, not a complete provenance service.

## Environment and limits

Direct Git/DNS access remained unavailable. The previous mounted source bundle was verified against its recorded Git blob hashes; additional relevant modules were fetched from the pinned GitHub commit and byte-verified locally. Execution used that **scoped snapshot**, not a full clone/install.

Python 3.13.5; NetworkX 3.6.1; NumPy 2.3.5; pydantic 2.13.4; pytest 9.0.2; pytest-cov 7.0.0; pytest-asyncio 1.3.0. Normal repository test configuration and root fixtures were retained.

Not claimed: all `tests/core`, full dependency installation, other Python versions, complete lazy-export/registry compatibility, GraphRAGContext global registration, MCP canaries, live embedding/index queries, all ten reference methods, real-corpus export integration, first-class analytic wrappers, or external-harness observation. No new provider spending or deployment occurred.

## Surface and next action

Authored Python changes: **636 additions, 60 deletions**, **696 changed lines**, **576 net growth**. Excludes unchanged transferred modules, fixtures/data, generated artifacts, documentation, and repeated rewrites. No isolated authoring-hour rate was measured.

**Next:** Batch 3, connect entity/assertion/passage records from the same Foundation snapshot to the existing embedding/index consumer, with persisted reload and a genuine query. Recheck an authorized execution environment and configured route; FAISS/llama-index/provider packages are absent from this scoped runner. Do not replace real semantic retrieval with a hash embedding or document-list surrogate. Catalog, analytics, and broader derivation capabilities remain in the same north-star plan.
