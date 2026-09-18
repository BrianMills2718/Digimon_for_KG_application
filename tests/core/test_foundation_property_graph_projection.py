from __future__ import annotations

import json

from Core.Projection.FoundationIR import parse_foundation_ir
from Core.Projection.PropertyGraph import (
    project_foundation_ir_to_assertion_graph,
    project_foundation_ir_to_binary_entity_graph,
)


def _ir():
    return parse_foundation_ir(
        {
            "format_version": "1.3",
            "producer": "onto-canon6",
            "assertion_count": 2,
            "assertions": [
                {
                    "assertion_id": "gassert2_binary",
                    "predicate": "org:employs",
                    "claim_text": "Acme employs Alice.",
                    "roles": {
                        "employer": [
                            {
                                "kind": "entity",
                                "entity_id": "entity:acme",
                                "name": "Acme",
                                "entity_type": "org:Organization",
                            }
                        ],
                        "employee": [
                            {
                                "kind": "entity",
                                "entity_id": "entity:alice",
                                "name": "Alice",
                                "entity_type": "org:Person",
                            }
                        ],
                        "since": [
                            {
                                "kind": "value",
                                "value_kind": "date",
                                "value": "2025-01-01",
                            }
                        ],
                    },
                    "qualifiers": {"sys:tier": "public"},
                    "provenance_refs": ["cand_binary"],
                },
                {
                    "assertion_id": "gassert2_ternary",
                    "predicate": "org:transferred",
                    "claim_text": "Alice transferred a report from Acme to Beta.",
                    "roles": {
                        "actor": [
                            {
                                "kind": "entity",
                                "entity_id": "entity:alice",
                                "name": "Alice",
                                "entity_type": "org:Person",
                            }
                        ],
                        "source": [
                            {
                                "kind": "entity",
                                "entity_id": "entity:acme",
                                "name": "Acme",
                                "entity_type": "org:Organization",
                            }
                        ],
                        "destination": [
                            {
                                "kind": "entity",
                                "entity_id": "entity:beta",
                                "name": "Beta",
                                "entity_type": "org:Organization",
                            }
                        ],
                    },
                    "qualifiers": {},
                    "provenance_refs": ["cand_ternary"],
                },
            ],
        },
        passage_payload={
            "format_version": "1.0",
            "producer": "onto-canon6",
            "passage_count": 2,
            "passages": [
                {
                    "passage_id": "gpassage1_binary",
                    "text": "Acme employs Alice.",
                    "source_ref": "source-window:1",
                    "supporting_provenance_refs": ["cand_binary"],
                },
                {
                    "passage_id": "gpassage1_ternary",
                    "text": "Alice transferred a report from Acme to Beta.",
                    "source_ref": "source-window:2",
                    "supporting_provenance_refs": ["cand_ternary"],
                },
            ],
        },
    )


def test_assertion_graph_preserves_nary_roles_and_value_fillers():
    graph = project_foundation_ir_to_assertion_graph(_ir())

    assert graph.nodes["entity:alice"]["node_kind"] == "entity"
    assert graph.nodes["gassert2_ternary"]["node_kind"] == "assertion"

    ternary_edges = list(
        graph.out_edges("gassert2_ternary", keys=True, data=True)
    )
    assert len(ternary_edges) == 3
    roles = {edge_data["role_name"] for _, _, _, edge_data in ternary_edges}
    assert roles == {"actor", "source", "destination"}

    value_node = "value::gassert2_binary::since::0"
    assert graph.nodes[value_node]["node_kind"] == "value"
    assert json.loads(graph.nodes[value_node]["value_json"]) == "2025-01-01"

    assertion_metadata = graph.nodes["gassert2_binary"]
    assert json.loads(assertion_metadata["passage_ids_json"]) == [
        "gpassage1_binary"
    ]


def test_binary_entity_graph_skips_nary_without_clique_expansion():
    projection = project_foundation_ir_to_binary_entity_graph(_ir())
    graph = projection.graph

    assert set(graph.nodes) == {"entity:acme", "entity:alice", "entity:beta"}
    assert graph.number_of_edges() == 1
    assert projection.skipped_assertion_ids == ("gassert2_ternary",)

    edge = graph.get_edge_data(
        "entity:acme",
        "entity:alice",
        key="gassert2_binary",
    )
    assert edge is not None
    assert edge["assertion_id"] == "gassert2_binary"
    assert edge["relation_name"] == "org:employs"
    assert edge["projection_lossy"] is True
    assert json.loads(edge["passage_ids_json"]) == ["gpassage1_binary"]

    role_map = json.loads(edge["role_map_json"])
    assert {entry["role_name"] for entry in role_map} == {
        "employer",
        "employee",
    }


def test_binary_projection_preserves_parallel_assertion_identity():
    payload = {
        "format_version": "1.3",
        "producer": "onto-canon6",
        "assertion_count": 2,
        "assertions": [
            {
                "assertion_id": assertion_id,
                "predicate": predicate,
                "roles": {
                    "left": [
                        {"kind": "entity", "entity_id": "entity:a", "name": "A"}
                    ],
                    "right": [
                        {"kind": "entity", "entity_id": "entity:b", "name": "B"}
                    ],
                },
                "qualifiers": {},
                "provenance_refs": [],
            }
            for assertion_id, predicate in (
                ("gassert2_one", "rel:one"),
                ("gassert2_two", "rel:two"),
            )
        ],
    }
    projection = project_foundation_ir_to_binary_entity_graph(
        parse_foundation_ir(payload)
    )
    graph = projection.graph

    assert graph.number_of_edges() == 2
    assert set(graph["entity:a"]["entity:b"]) == {
        "gassert2_one",
        "gassert2_two",
    }
