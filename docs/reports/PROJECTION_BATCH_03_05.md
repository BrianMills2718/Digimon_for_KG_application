# Foundation catalog, analytics, lineage and finding: execution receipt

**Parent:** [North-star batch-and-converge plan](../planning/NORTH_STAR_VERTICAL_SLICE_PLAN.md).  
**Inspected committed base:** `8404330797ed099e701b35df9acfba4987259fdc`.  
**Disposition:** independent catalog + executable centrality/SQL analytics + queryable derivation lineage + persisted finding are now exercised over the same saved Foundation project. Vector retrieval and Leiden remain blocked by missing authentic runtime dependencies and are not substituted.

## Observed path

```text
Foundation IR + passages
→ SQLite + assertion graph + binary graph
→ saved/reopened project
→ generated Markdown catalog (content + actual representation/schema map)
→ graph retrieval / SQL aggregation
→ centrality on bounded retrieved SUBGRAPH
→ selected entities → exact source passages
→ analytic artifact
→ recursive execution/artifact lineage query
→ persisted finding with evidence, method, parameters, scope and limitations
```

The catalog reports vectors as unavailable rather than fabricating an address. The analytics result is derived state, not evidence; the finding carries an explicit warning that centrality does not establish causal influence or importance outside the retrieved working set.

## Checks actually executed

The cumulative selected suite now passes **95 tests, zero skipped**, final run **9.02 seconds**. The focused lineage/finding suite passes **7/7**. Normal repository pytest configuration/root fixtures were retained.

```bash
python -m compileall -q Core/Projection scripts/run_foundation_demo.py
python -m pytest \
  tests/core/test_foundation_ir_contract.py \
  tests/core/test_relational_projection_contract.py \
  tests/core/test_foundation_property_graph_projection.py \
  tests/core/test_projection_boundary_repairs.py \
  tests/core/test_foundation_project.py \
  tests/core/test_graph_materialize_exactness.py \
  tests/core/test_foundation_graph_runtime.py \
  tests/core/test_foundation_catalog.py \
  tests/core/test_foundation_analytics.py \
  tests/core/test_foundation_lineage_finding.py -q
```

Fresh build and fresh-process `--reuse` execute the catalog, SQL aggregation, graph retrieval, centrality, finding creation and lineage traversal over the same project.

## Failure-driven repairs

- Catalog reuse validates the existing generated catalog rather than treating it as an overwrite failure.
- Lineage query honors execution-qualified artifact references. Content-addressed artifacts may legitimately be reproduced by multiple executions; a bare hash is ambiguous when several successful executions produced identical bytes.
- Parent/coordinator execution records that merely reference child-produced artifacts are not misclassified as second producers.
- Failed executions never become successful artifact producers.
- Findings refuse unsuccessful/unobserved analytical inputs and require evidence.

## New capability surface

- `Core/Projection/Catalog.py` — deterministic progressive-disclosure Markdown environment map with entity/assertion/source pages, actual SQLite/graph schema addresses, link/hash validation, and truthful representation availability.
- `Core/Projection/Analytics.py` — typed degree/betweenness/PageRank over the exact retrieved NetworkX subgraph plus exact SQL predicate aggregation. Leiden availability is reported truthfully; no substitute algorithm is mislabeled.
- `Core/Projection/Lineage.py` — on-demand recursive read model over the existing execution JSONL; no second provenance database.
- `Core/Projection/Finding.py` — deterministic derived finding over one successful analytic artifact, including method/parameters, graph scope, ranked entities, exact evidence references, limitations, and lineage execution IDs.
- `FoundationProject` exposes catalog, analytics, lineage and finding helpers; the growing CLI adds catalog/centrality/aggregation/finding flags.

## Verification boundary

Execution remains a byte-verified scoped source snapshot on Python 3.13.5, not a full clean install or full-repository certification. The synthetic fixtures are regression data, not production onto-canon exports. No provider spending or deployment occurred.

Authentic **semantic vector retrieval is still blocked** in this runner: FAISS/llama-index and a configured semantic embedding route are absent. Hash vectors, TF-IDF, or document lists were deliberately not substituted. **Leiden** is likewise not claimed here because the repository's graspologic/igraph path is unavailable in this runner.

External harness observation remains unperformed. The fixed CLI demonstrates the capability chain and replay, not autonomous strategy selection.

## Implementation surface

Relative to the Batch 2 local tested tree, this tranche adds roughly **1,500 authored Python implementation/test lines** (catalog, analytics, lineage/finding, tests and runner integration) before documentation. This is a surface estimate from the local diff, not a measured LOC/hour rate.

## Next action

The shortest remaining first-demo blockers are now operational rather than architectural:
1. obtain an authorized runner/configuration for the existing semantic embedding/index path and execute entity/assertion/passage query + persisted reload;
2. obtain the real Leiden dependency path and run it over the same bounded working set;
3. run one actual external harness journey that uses the catalog to move across representations and from analytic output back to evidence.

Do not add a substitute vector or community algorithm merely to close the checklist.
