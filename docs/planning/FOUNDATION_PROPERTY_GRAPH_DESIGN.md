# Foundation IR → Property Graph Bounded Design

**Status:** implementation design  
**Date:** 2026-09-17  
**Planning path:** Company Planning durable_solo / bounded design  
**Parent:** NORTH_STAR_VERTICAL_SLICE_PLAN.md

## Decision pressure

Foundation IR is role-aware and may be n-ary. The maintained DIGIMON ER runtime is an undirected entity–entity NetworkX graph whose retrieval/analytics assume entity nodes and ordinary edges.

A direct universal collapse would either:
- invent pairwise relationships for n-ary assertions; or
- insert assertion/value nodes into the existing ER graph and change the semantics of PPR, neighborhoods, centrality, communities, sparse matrices, and reference methods.

Neither is acceptable as an implicit compatibility mapping.

## Decision

Maintain two distinct graph projections with explicit semantics.

### A. Canonical assertion graph — lossless

Purpose: preserve the governed semantic IR structurally without inventing binary relations.

Node kinds:
- entity — node id is the exact Foundation entity_id;
- assertion — node id is the exact Foundation assertion_id;
- value — deterministic projection-local node id scoped to assertion/role/ordinal when a value filler must be represented structurally.

Edges:
- assertion → filler, with role_name and filler_ordinal;
- edge metadata identifies the role and preserves the Foundation predicate on the assertion node.

Assertion-node metadata:
- assertion_id;
- predicate;
- claim_text;
- qualifiers;
- confidence;
- provenance_refs;
- source/passage identities when resolved.

Properties:
- lossless with respect to the supported Foundation assertion structure;
- canonical IDs are preserved;
- n-ary assertions remain n-ary;
- useful for semantic inspection, provenance/navigation, and future algorithms designed for assertion graphs;
- MUST NOT be silently passed to existing ER operators that assume entity-only nodes.

### B. Retrieval entity-relation graph — derived/lossy where declared

Purpose: feed the existing entity-graph retrieval/SNA machinery without pretending that every Foundation assertion is binary.

An assertion may collapse to an entity edge only when:
1. it contains exactly two entity filler occurrences total;
2. both canonical entity IDs are present;
3. no projection policy claims a direction that Foundation IR itself does not establish;
4. the edge records the Foundation predicate and the two role names;
5. the projection records that structural/value detail was summarized into an entity edge.

Because the maintained NetworkX ER runtime is undirected, the first adapter should remain undirected. It must not infer subject/object direction from role ordering.

For assertions with fewer or more than two entity fillers:
- keep them in the canonical assertion graph;
- do not fabricate pairwise clique edges;
- optionally project them later only through an explicit predicate/role projection policy.

Edge identity/provenance:
- endpoint IDs are exact Foundation entity IDs;
- relation_name = Foundation predicate;
- source evidence uses resolved Foundation passage IDs where available;
- assertion_id is preserved as edge metadata;
- role mapping is preserved as edge metadata;
- multiple assertions between the same entity pair must not silently erase assertion identity. If the existing simple Graph storage merges them, metadata must retain the contributing assertion IDs/predicates/roles or the adapter must fail until a safe merge representation is defined.

## Why not a single graph?

A single graph would force one representation to serve two incompatible purposes:
- semantic fidelity to governed n-ary assertions;
- compatibility with entity-only retrieval and social-network algorithms.

DIGIMON's vision explicitly allows complementary representations. Keeping both makes lossiness inspectable instead of hidden.

## First implementation slice

1. Add a pure FoundationIR → NetworkX assertion-graph projector.
2. Add contract tests for:
   - exact entity/assertion IDs;
   - n-ary role preservation;
   - value filler preservation;
   - provenance/passage linkage.
3. Add a pure FoundationIR → binary entity-graph projector that:
   - projects only exactly-two-entity assertions;
   - records predicate, assertion ID and role mapping;
   - reports skipped non-binary assertion IDs.
4. Do not wire either graph into GraphRAGContext/MCP/reference methods until source-level contract tests exist and the merge behavior for multiple assertions on one entity pair is explicit.

## Acceptance probe

Given one binary and one ternary assertion:
- assertion graph contains both assertions with all roles;
- entity graph contains exactly one relation edge from the binary assertion;
- ternary assertion is reported as skipped, not converted to three pairwise edges;
- both graphs reuse canonical entity IDs;
- source/passages can be recovered from graph metadata.

## Rejected alternatives

- **Clique-expand every n-ary assertion:** rejected because it fabricates pairwise relations.
- **Use assertion nodes inside the existing ER graph:** rejected because current analytics/retrieval would treat assertions as ordinary entities.
- **Choose source/target by dictionary/role order:** rejected because direction would be an implementation accident.
- **Keep only the lossless assertion graph:** rejected for now because it would strand the substantial maintained entity-graph retrieval/SNA toolchain.

## Future extension

Predicate-specific projection policies may declare safe binary direction or richer transformations later, but those policies must be explicit, inspectable, and provenance-bearing. They belong to DIGIMON projection semantics, not onto-canon semantic authority.
