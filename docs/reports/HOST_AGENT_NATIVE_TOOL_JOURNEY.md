# Host-Agent Native-Tool Journey Receipt

**Date:** 2026-09-18  
**Code basis:** locally executed 95-test Batch 3–5 tree, subsequently published as `93d65b3f5b5193f4f0616646478ce2d8e12559e1`  
**Scope:** one synthetic governed Foundation project; host-agent/native file + SQLite tools plus DIGIMON specialized graph/analytic runtime.

## What was observed

The host agent began from the generated catalog rather than from a hard-coded retrieval sequence.

1. Opened the generated catalog entry point.
2. Discovered Alice and the canonical identity `entity:alice`.
3. Read the entity page, which exposed:
   - exact SQL address;
   - binary-graph node identity;
   - assertion links;
   - source/evidence links;
   - vector status as unavailable.
4. Used SQLite directly, guided by the catalog schema, to inspect governed assertions for Alice:
   - binary `org:employs`;
   - n-ary `org:transferred`, which is intentionally absent from the binary entity graph;
   - exact passage/source identities and text.
5. Used the specialized DIGIMON graph runtime with the same canonical entity ID.
6. Ran degree centrality over the retrieved graph working set.
7. Recovered only the exact source passage supporting the selected graph edge.
8. Persisted a finding that retained method, parameters, graph scope, evidence references, limitations, analytic artifact identity, and recursive derivation lineage.

Observed lineage included:

```text
projection.graphs
→ project.build
→ graph.runtime_adapter
→ subgraph.khop_paths
→ analytics.centrality.degree
→ finding.from_analytic_result
```

## Important semantic observation

SQL exposed the n-ary transfer assertion while the binary entity graph did not. This is expected and demonstrates why complementary representations matter rather than forcing all governed semantics into one graph.

The graph analytic result was explicitly scoped to the retrieved binary-association view. Degree centrality was treated as derived structural state, not source evidence, causal influence, or global importance.

## What this proves

- the generated Markdown catalog is usable as a progressive-disclosure semantic and operational map;
- canonical identity supports cross-representation movement without fuzzy rediscovery;
- native file/SQLite capabilities can compose with specialized DIGIMON graph/analytic capabilities;
- an analytic result can lead back to exact source evidence and retained derivation;
- the harness can choose the sequence rather than DIGIMON prescribing one.

## What this does not prove

- this was not a qagent-specific certification;
- semantic vector retrieval was unavailable in this runner;
- Leiden was unavailable because the repository's actual dependency stack was absent;
- the fixture was synthetic, not a production corpus export;
- this does not certify all reference methods, MCP surfaces, providers, Python versions, or autonomous planning quality.

## Remaining first-proof blockers

1. real semantic embedding/index/query/reload through the repository's supported vector stack;
2. a real Leiden execution through the supported community dependency path;
3. a qagent-specific journey only if that exact harness is required as an acceptance target.
