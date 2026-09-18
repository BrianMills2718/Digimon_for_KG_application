"""Property-graph projections from governed Foundation IR.

Two graphs are intentionally produced:
- an assertion graph that preserves n-ary role structure;
- a binary entity graph for existing entity-network retrieval/analytics.

The binary projection never clique-expands n-ary assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import networkx as nx

from Core.Common.Constants import GRAPH_FIELD_SEP

from .FoundationIR import FoundationIR, FoundationRoleFillerRecord


PROPERTY_GRAPH_PROJECTION_VERSION = "1.0"


@dataclass(frozen=True)
class BinaryEntityGraphProjection:
    graph: nx.MultiGraph
    skipped_assertion_ids: tuple[str, ...]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _passage_ids(ir: FoundationIR, assertion_id: str) -> tuple[str, ...]:
    return tuple(
        passage.passage_id for passage in ir.passages_for_assertion(assertion_id)
    )


def project_foundation_ir_to_assertion_graph(ir: FoundationIR) -> nx.MultiDiGraph:
    """Build a lossless role-aware assertion graph.

    Entity and assertion node IDs are reused verbatim from Foundation IR.
    Value nodes use deterministic projection-local IDs because literals have no
    canonical entity identity.
    """

    graph = nx.MultiDiGraph(
        projection_kind="foundation_assertion_graph",
        projection_version=PROPERTY_GRAPH_PROJECTION_VERSION,
        foundation_format_version=ir.format_version,
        foundation_producer=ir.producer,
        source_sha256=ir.source_sha256 or "",
    )

    for entity in ir.entities_by_id.values():
        graph.add_node(
            entity.entity_id,
            node_kind="entity",
            entity_id=entity.entity_id,
            entity_name=entity.names[0] if entity.names else entity.entity_id,
            names_json=_json(entity.names),
            entity_types_json=_json(entity.entity_types),
            alias_ids_json=_json(entity.alias_ids),
        )

    for assertion in ir.assertions:
        passages = _passage_ids(ir, assertion.assertion_id) if ir.passages else ()
        graph.add_node(
            assertion.assertion_id,
            node_kind="assertion",
            assertion_id=assertion.assertion_id,
            predicate=assertion.predicate,
            claim_text=assertion.claim_text or "",
            confidence=assertion.confidence if assertion.confidence is not None else "",
            qualifiers_json=_json(dict(assertion.qualifiers)),
            provenance_refs_json=_json(assertion.provenance_refs),
            passage_ids_json=_json(passages),
            source_urls_json=_json(assertion.source_urls),
            namespace_id=assertion.namespace_id or "",
            source_registry_id=assertion.source_registry_id or "",
        )

        for role_name, fillers in assertion.roles.items():
            for ordinal, filler in enumerate(fillers):
                if filler.entity_id is not None:
                    target_id = filler.entity_id
                else:
                    target_id = (
                        f"value::{assertion.assertion_id}::{role_name}::{ordinal}"
                    )
                    graph.add_node(
                        target_id,
                        node_kind="value",
                        value_kind=filler.value_kind or "",
                        value_json=_json(filler.value)
                        if filler.value is not None
                        else "",
                        normalized_json=_json(filler.normalized)
                        if filler.normalized is not None
                        else "",
                        raw=filler.raw or "",
                    )

                graph.add_edge(
                    assertion.assertion_id,
                    target_id,
                    key=f"{role_name}:{ordinal}",
                    edge_kind="role_filler",
                    role_name=role_name,
                    filler_ordinal=ordinal,
                    filler_kind=filler.kind,
                )

    return graph


def _entity_occurrences(
    roles: dict[str, tuple[FoundationRoleFillerRecord, ...]] | Any,
) -> list[tuple[str, int, FoundationRoleFillerRecord]]:
    result: list[tuple[str, int, FoundationRoleFillerRecord]] = []
    for role_name, fillers in roles.items():
        for ordinal, filler in enumerate(fillers):
            if filler.entity_id is not None:
                result.append((role_name, ordinal, filler))
    return result


def project_foundation_ir_to_binary_entity_graph(
    ir: FoundationIR,
) -> BinaryEntityGraphProjection:
    """Project only exactly-two-entity assertions into an undirected MultiGraph.

    The MultiGraph keeps assertion identity when several governed assertions
    connect the same entity pair. Assertions that are unary or n-ary are
    reported as skipped instead of being clique-expanded.
    """

    graph = nx.MultiGraph(
        projection_kind="foundation_binary_entity_graph",
        projection_version=PROPERTY_GRAPH_PROJECTION_VERSION,
        foundation_format_version=ir.format_version,
        foundation_producer=ir.producer,
        source_sha256=ir.source_sha256 or "",
        lossiness="binary-only; non-binary assertions skipped",
    )

    for entity in ir.entities_by_id.values():
        graph.add_node(
            entity.entity_id,
            node_kind="entity",
            entity_id=entity.entity_id,
            entity_name=entity.names[0] if entity.names else entity.entity_id,
            entity_type=GRAPH_FIELD_SEP.join(entity.entity_types),
            alias_ids_json=_json(entity.alias_ids),
            assertion_ids_json=_json(entity.assertion_ids),
        )

    skipped: list[str] = []
    for assertion in ir.assertions:
        occurrences = _entity_occurrences(assertion.roles)
        if len(occurrences) != 2:
            skipped.append(assertion.assertion_id)
            continue

        left_role, left_ordinal, left = occurrences[0]
        right_role, right_ordinal, right = occurrences[1]
        assert left.entity_id is not None
        assert right.entity_id is not None

        passages = _passage_ids(ir, assertion.assertion_id) if ir.passages else ()
        source_id = GRAPH_FIELD_SEP.join(passages)
        role_map = [
            {
                "role_name": left_role,
                "filler_ordinal": left_ordinal,
                "entity_id": left.entity_id,
            },
            {
                "role_name": right_role,
                "filler_ordinal": right_ordinal,
                "entity_id": right.entity_id,
            },
        ]

        graph.add_edge(
            left.entity_id,
            right.entity_id,
            key=assertion.assertion_id,
            assertion_id=assertion.assertion_id,
            relation_name=assertion.predicate,
            predicate=assertion.predicate,
            weight=1.0,
            source_id=source_id,
            passage_ids_json=_json(passages),
            provenance_refs_json=_json(assertion.provenance_refs),
            role_map_json=_json(role_map),
            claim_text=assertion.claim_text or "",
            qualifiers_json=_json(dict(assertion.qualifiers)),
            projection_lossy=True,
        )

    return BinaryEntityGraphProjection(
        graph=graph,
        skipped_assertion_ids=tuple(skipped),
    )
