from __future__ import annotations

import json
import sqlite3

from Core.Projection.FoundationIR import parse_foundation_ir
from Core.Projection.Relational import (
    project_foundation_ir_to_sqlite,
    relational_schema_manifest,
)


def _ir():
    return parse_foundation_ir(
        {
            "format_version": "1.3",
            "producer": "onto-canon6",
            "assertion_count": 1,
            "assertions": [
                {
                    "assertion_id": "gassert2_abc",
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
                    "provenance_refs": ["cand_001"],
                }
            ],
        },
        passage_payload={
            "format_version": "1.0",
            "producer": "onto-canon6",
            "passage_count": 1,
            "passages": [
                {
                    "passage_id": "gpassage1_abc",
                    "text": "Acme employs Alice.",
                    "source_ref": "source-window:1",
                    "supporting_provenance_refs": ["cand_001"],
                }
            ],
        },
    )


def test_relational_projection_preserves_identity_and_evidence_join(tmp_path):
    path = tmp_path / "knowledge.sqlite"
    result = project_foundation_ir_to_sqlite(_ir(), path)

    assert result.entity_count == 2
    assert result.assertion_count == 1
    assert result.passage_count == 1

    conn = sqlite3.connect(path)
    try:
        alice = conn.execute(
            """
            SELECT e.entity_id, n.name, t.entity_type
            FROM entities e
            JOIN entity_names n USING (entity_id)
            JOIN entity_types t USING (entity_id)
            WHERE e.entity_id = ?
            """,
            ("entity:alice",),
        ).fetchone()
        assert alice == ("entity:alice", "Alice", "org:Person")

        evidence = conn.execute(
            """
            SELECT a.assertion_id, a.predicate, p.passage_id, p.text
            FROM assertions a
            JOIN assertion_roles r ON r.assertion_id = a.assertion_id
            JOIN assertion_provenance ap ON ap.assertion_id = a.assertion_id
            JOIN passage_support ps ON ps.provenance_ref = ap.provenance_ref
            JOIN passages p ON p.passage_id = ps.passage_id
            WHERE r.entity_id = ?
            """,
            ("entity:alice",),
        ).fetchone()
        assert evidence == (
            "gassert2_abc",
            "org:employs",
            "gpassage1_abc",
            "Acme employs Alice.",
        )

        value_json = conn.execute(
            """
            SELECT value_json
            FROM assertion_roles
            WHERE assertion_id = ? AND role_name = ?
            """,
            ("gassert2_abc", "since"),
        ).fetchone()[0]
        assert json.loads(value_json) == "2025-01-01"
    finally:
        conn.close()


def test_relational_schema_manifest_describes_agent_visible_tables():
    schema = relational_schema_manifest()
    assert schema["entities"]["primary_key"] == "entity_id"
    assert "n-ary" in schema["assertion_roles"]["purpose"]
    assert schema["passages"]["primary_key"] == "passage_id"


def test_relational_projection_refuses_accidental_overwrite(tmp_path):
    path = tmp_path / "knowledge.sqlite"
    project_foundation_ir_to_sqlite(_ir(), path)

    try:
        project_foundation_ir_to_sqlite(_ir(), path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("projection should refuse overwrite by default")

    project_foundation_ir_to_sqlite(_ir(), path, overwrite=True)
