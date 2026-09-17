# DIGIMON Functionality

**Snapshot:** 2026-09-17  
**Purpose:** concise implementation inventory. For the north star use `docs/VISION.md`; for current caveats use `docs/CURRENT_STATE.md`.

## One-liner

DIGIMON is evolving toward a **general text-derived representation, retrieval, and analytics runtime** downstream of governed semantic IR. The current implementation is strongest in graph/vector retrieval, graph-oriented transformations and evidence-grounded answer generation.

The external harness owns adaptive reasoning/composition. DIGIMON owns specialized representations, retrieval/analytic capabilities, typed contracts, shared identity/evidence facts and bounded local model-assisted transformations.

## Current functionality by capability plane

### REPRESENT

#### Property graphs and graph-derived structures — **Implemented / strongest representation family**

Current build surfaces include:

- Entity-Relationship graph;
- Relationship-Keyword graph;
- hierarchical tree graph;
- balanced hierarchical tree;
- passage graph.

Recent hardening includes truthful build results, zero-node failure, graph-specific namespaces, source-chunk manifests and pragmatic dependent-artifact invalidation.

#### Vector indexes — **Implemented**

FAISS-backed entity/relationship indexing and retrieval exists. Current behavior includes:

- dimensions inferred from actual embeddings;
- L2 distance normalized in the correct higher-is-better direction;
- typed seed handling;
- stable graph/VDB metadata mapping;
- canonical entity/relation VDB selection;
- exact graph entity linking before approximate vector linking.

#### Community / sparse graph structures — **Implemented / pragmatic lifecycle**

Community and sparse propagation resources exist. Community materialization and singleton Leiden handling are implemented. Rebuilt graphs invalidate known stale community artifacts; stale in-memory community use fails closed.

Sparse propagation now validates matrix dimensions, though same-shaped cross-graph matrix identity remains a known gap.

#### Raw document/chunk representation — **Implemented standalone path**

Standalone corpus preparation supports common text/document formats, and maintained `ChunkFactory` now actually applies configured chunking. This remains useful for independent use, experiments and benchmarks.

It is **not the conceptual ecosystem authority path**: the target canonical upstream seam is governed semantic IR from onto-canon6.

#### Graph/table/vector conversion — **Implemented / experimental integration**

Substantive conversion code exists using NetworkX, pandas/NumPy and embedding adapters. It demonstrates multi-representation intent but is not yet the complete canonical projection system.

#### Relational/tabular canonical projection — **Partial / target gap**

Table conversion exists, but there is not yet one maintained Foundation-IR→relational database projection with stable entity/assertion/evidence IDs and an agent-readable schema.

#### Wiki / progressive-disclosure catalog — **Planned**

The target is a generated agent-readable knowledge/environment map describing semantic organization, representations, schemas, canonical IDs, specialized capabilities and source/evidence paths. It should rely on native harness file/link/search abilities rather than inventing wrapper APIs where unnecessary.

#### Semantic/RDF projection — **Planned / not canonical**

The governed IR appears semantically suitable, but a maintained RDF/SPARQL projection is not currently part of the core.

#### Specialized lexical/BM25 representation — **Planned selectively**

Only justified where fielded/ranked lexical retrieval adds capability beyond native harness text/file search.

---

### RETRIEVE

#### Typed operator/composition core — **Implemented / materially hardened**

The maintained core includes typed values for:

- query text;
- entity sets;
- relationship sets;
- chunk sets;
- subgraphs;
- community sets;
- score vectors.

Static validation now requires explicit required-slot wiring, invalid plans fail closed by default, generic loop/conditional ownership no longer double-executes body steps, and carried loop outputs retain their real slot kinds.

The operator catalog is extensible/dynamic; documentation should not rely on a permanent fixed count.

#### Reference methods — **Implemented by source; fresh runtime certification pending**

Ten maintained reference compositions remain:

- `basic_local`
- `basic_global`
- `lightrag`
- `fastgraphrag`
- `hipporag`
- `tog`
- `gr`
- `dalk`
- `kgp`
- `med`

They are examples/shortcuts, not DIGIMON's identity. Major wiring/evidence defects have been repaired, including explicit ToG/KGP hop state, structural materialization, Basic Global completion and FastGraphRAG/HippoRAG PPR-mode separation.

#### Entity retrieval — **Implemented**

Includes:

- entity vector search;
- exact graph linking;
- one-hop expansion;
- PPR;
- TF-IDF ranking;
- model-assisted extraction/linking paths.

#### Relationship retrieval — **Implemented**

Includes:

- relationship vector search;
- one-hop relation retrieval;
- score propagation/aggregation;
- model-assisted relation selection.

Relationship vector text now preserves endpoint identity as well as semantic relation fields.

#### Chunk/evidence retrieval — **Implemented / evidence-safe**

Includes:

- chunks from entity occurrences;
- chunks from relationships;
- score→chunk propagation;
- direct source/evidence recovery.

Maintained paths resolve exact stored IDs and do not fabricate pseudo-evidence for unresolved objects.

#### Subgraph/path retrieval — **Implemented**

Includes:

- k-hop/neighborhood/path operations;
- Steiner approximation;
- PCST-style optimization;
- model-assisted path filtering;
- structural materialization back to source evidence.

#### Community retrieval — **Implemented**

Community selection/materialization supports global retrieval while preserving source lineage. Community artifacts are treated as graph-derived resources rather than independent truth.

#### Grounded answering — **Implemented / hardened**

Final generation:

- refuses zero-evidence answering;
- avoids the LLM call when evidence is absent;
- passes exact evidence IDs;
- validates returned citations;
- preserves evidence IDs/provenance in result metadata;
- returns an explicit insufficient-evidence result rather than plausible unsupported prose.

---

### ANALYZE / TRANSFORM

#### Graph/SNA-oriented analytics — **Substantive but not yet fully cataloged**

DIGIMON already contains meaningful graph analytical/transformation machinery, including:

- Personalized PageRank / diffusion scores;
- community detection/materialization;
- connected/structural graph operations;
- k-hop/path transformations;
- PCST optimization;
- Steiner approximation;
- score aggregation/propagation;
- graph/table/vector conversion utilities.

The target is to promote a coherent typed analytics plane from real existing capabilities first, then fill high-value gaps such as centrality families, cohesion/brokerage and related SNA measures.

#### General tabular/statistical analytics — **Partial / target gap**

A complete typed analytical suite over relational/tabular projections is not yet established. Future additions should come from concrete reusable needs such as aggregation, distributions, group comparisons, clustering, anomaly detection and trend calculations—not an indiscriminate toolbox.

#### Derived analytic artifacts — **Partial**

Communities and score vectors are already derived outputs. The broader target requires uniform method/parameter/input lineage for derived values such as centrality, bridge classifications, diffusion paths, model results and findings.

---

## Cross-representation identity — **Partial / foundational target**

Current graph/vector/evidence paths preserve many canonical IDs, but the full target is explicit identity continuity across all projections:

```text
entity_id
assertion_id
predicate_id
source_ref
evidence identity
```

The goal is for an agent to discover an entity in a wiki/catalog, use the same ID in SQL, traverse the corresponding graph node, query vector metadata and recover exact evidence without fuzzy rediscovery.

---

## Provenance and derivation

### Evidence provenance — **Implemented strongly on maintained answer paths**

Source/chunk IDs, graph-source materialization and answer citation validation are meaningful current functionality.

### Semantic provenance — **Primarily upstream / consumed downstream**

Onto-canon6 owns governed assertion/source semantics. DIGIMON should preserve those identities/provenance through projections.

### Artifact/derivation lineage — **Partial foundations / target gap**

Current foundations include producer metadata, graph-source manifests and pragmatic invalidation. The full target is a first-class lineage chain:

```text
source artifact
→ governed semantic IR
→ representation projection
→ bounded retrieval artifact
→ analytic transformation + parameters
→ derived artifact
→ finding
```

The derivation graph is distinct from the domain/property graph.

---

## Access surfaces

### MCP — **Implemented in code / agent-facing**

`digimon_mcp_stdio_server.py` is currently the strongest modern tool-protocol surface for specialized capabilities.

MCP is an interface, not the product thesis.

### CLI — **Implemented / Transitional**

`digimon_cli.py` remains a human-facing entry point but currently uses the older `PlanningAgent` / `AgentOrchestrator` / optional ReAct stack.

### Python runtime — **Partial**

The target is a clean developer/application library over the same maintained representation/retrieval/analytics core.

The long-term shape is **CLI + Python + MCP over one core**.

---

## Harness boundary

DIGIMON should not duplicate native harness capabilities merely for symmetry.

If the harness already handles:

- file reading;
- text search/grep;
- link following;
- wiki/directory navigation;
- planning/sequencing/retries/branching/stopping;

DIGIMON should generally provide good artifacts and specialized engines rather than wrapper tools for those behaviors.

AoT/GoT/ReAct/decomposition remain optional reasoning heuristics, not mandatory DIGIMON runtime architecture.

---

## Resource/freshness functionality — **Implemented pragmatically / not generalized**

Current maintained behavior includes:

- active dataset/graph selection;
- canonical entity/relation VDB preference;
- in-memory VDB eviction when a graph is replaced;
- source-chunk manifests triggering graph rebuild on changed/added chunks;
- removal of canonical stale VDB/community artifacts after successful rebuild;
- ER sparse-matrix invalidation;
- fail-closed stale community use.

This is intentionally concrete. A generalized resource governance system should only be added when real representation/analytic artifacts require it.

---

## Current verification boundary

Many deterministic tests have been added around repaired contracts, but the current head has not been freshly executed in the available environment. Current connector-created commits/PRs have not generated new GitHub Actions runs.

Do not describe source-reviewed tests as a green runtime certification.

---

## Current priorities

See `docs/ROADMAP.md`. In compact form:

1. get a real current-head core/canary execution signal;
2. finish custom ontology wiring and first runtime reds;
3. verify the canonical onto-canon/Foundation IR handoff;
4. enforce cross-representation identity;
5. implement a canonical relational projection;
6. generate the progressive-disclosure wiki/catalog;
7. inventory/promote graph analytics into a first-class typed analytic plane;
8. add minimal derivation lineage through projection→retrieval→analysis;
9. bind remaining graph-derived resources to exact graph identity;
10. converge Python/CLI/MCP on one maintained core;
11. broaden deterministic architecture tests;
12. evaluate/benchmark after these seams are real.

For authoritative context:

- `docs/VISION.md`
- `docs/CURRENT_STATE.md`
- `docs/ARCHITECTURE.md`
- `docs/GAP_ANALYSIS.md`
- `docs/ROADMAP.md`
- `docs/DOCUMENTATION_COVERAGE.md`
