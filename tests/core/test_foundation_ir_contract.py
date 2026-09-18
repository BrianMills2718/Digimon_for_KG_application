from __future__ import annotations

import hashlib
import json

import pytest

from Core.Projection.FoundationIR import (
    FoundationIRContractError,
    load_foundation_ir,
    parse_foundation_ir,
)


def _assertion_bundle():
    return {
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
                            "alias_ids": ["entity:alice-smith"],
                        }
                    ],
                    "since": [
                        {
                            "kind": "value",
                            "value_kind": "date",
                            "value": "2025-01-01",
                            "raw": "January 1, 2025",
                        }
                    ],
                },
                "qualifiers": {
                    "sys:tier": "public",
                    "future:additive-qualifier": {"preserve": True},
                },
                "confidence": 0.91,
                "provenance_refs": ["cand_001"],
                "source_urls": ["https://example.test/source"],
                "namespace_id": "person:alice",
                "source_registry_id": "registry:test",
            }
        ],
    }


def _passage_bundle():
    return {
        "format_version": "1.0",
        "producer": "onto-canon6",
        "passage_count": 1,
        "passages": [
            {
                "passage_id": "gpassage1_abc",
                "text": "Acme employs Alice.",
                "source_ref": "source-window:1",
                "source_urls": ["https://example.test/source"],
                "source_registry_id": "registry:test",
                "namespace_id": "person:alice",
                "content_hash": "a" * 64,
                "supporting_provenance_refs": ["cand_001"],
            }
        ],
    }


def test_parse_preserves_identity_and_additive_qualifiers():
    ir = parse_foundation_ir(
        _assertion_bundle(),
        passage_payload=_passage_bundle(),
    )

    assertion = ir.assertions_by_id["gassert2_abc"]
    assert assertion.predicate == "org:employs"
    assert assertion.qualifiers["future:additive-qualifier"] == {"preserve": True}

    alice = ir.entities_by_id["entity:alice"]
    assert alice.names == ("Alice",)
    assert alice.entity_types == ("org:Person",)
    assert alice.alias_ids == ("entity:alice-smith",)
    assert alice.assertion_ids == ("gassert2_abc",)

    assert ir.assertions_by_entity_id["entity:acme"] == ("gassert2_abc",)
    assert ir.assertions_by_provenance_ref["cand_001"] == ("gassert2_abc",)
    assert ir.passages_by_provenance_ref["cand_001"] == ("gpassage1_abc",)
    assert [p.passage_id for p in ir.passages_for_assertion("gassert2_abc")] == [
        "gpassage1_abc"
    ]


def test_parse_preserves_value_fillers_for_later_projections():
    ir = parse_foundation_ir(_assertion_bundle())
    assertion = ir.assertions_by_id["gassert2_abc"]
    since = assertion.roles["since"][0]
    assert since.kind == "value"
    assert since.value_kind == "date"
    assert since.value == "2025-01-01"
    assert since.raw == "January 1, 2025"


def test_parse_does_not_rename_unicode_or_canonical_ids():
    bundle = _assertion_bundle()
    employee = bundle["assertions"][0]["roles"]["employee"][0]
    employee["entity_id"] = "entity:Москва/北京/José"
    employee["name"] = "Москва 北京 José"

    ir = parse_foundation_ir(bundle)
    assert "entity:Москва/北京/José" in ir.entities_by_id
    assert ir.entities_by_id["entity:Москва/北京/José"].names == (
        "Москва 北京 José",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("format_version", "1.2"),
        ("producer", "some-other-producer"),
    ],
)
def test_envelope_fails_closed_on_unsupported_contract(field, value):
    bundle = _assertion_bundle()
    bundle[field] = value
    with pytest.raises(FoundationIRContractError):
        parse_foundation_ir(bundle)


def test_entity_filler_requires_canonical_entity_id():
    bundle = _assertion_bundle()
    del bundle["assertions"][0]["roles"]["employee"][0]["entity_id"]
    with pytest.raises(FoundationIRContractError):
        parse_foundation_ir(bundle)


def test_duplicate_assertion_identity_fails_closed():
    bundle = _assertion_bundle()
    bundle["assertions"].append(dict(bundle["assertions"][0]))
    bundle["assertion_count"] = 2
    with pytest.raises(FoundationIRContractError):
        parse_foundation_ir(bundle)


def test_counts_are_verified():
    bundle = _assertion_bundle()
    bundle["assertion_count"] = 9
    with pytest.raises(FoundationIRContractError):
        parse_foundation_ir(bundle)


def test_passage_companion_must_close_over_assertion_provenance():
    passages = _passage_bundle()
    passages["passages"][0]["supporting_provenance_refs"] = ["cand_other"]
    with pytest.raises(FoundationIRContractError):
        parse_foundation_ir(_assertion_bundle(), passage_payload=passages)


def test_passage_companion_cannot_omit_selected_assertion_provenance():
    passages = _passage_bundle()
    passages["passages"] = []
    passages["passage_count"] = 0
    with pytest.raises(FoundationIRContractError):
        parse_foundation_ir(_assertion_bundle(), passage_payload=passages)



def test_passage_content_hash_must_be_sha256():
    passages = _passage_bundle()
    passages["passages"][0]["content_hash"] = "not-a-sha"
    with pytest.raises(FoundationIRContractError):
        parse_foundation_ir(_assertion_bundle(), passage_payload=passages)


def test_snapshot_sha256_is_recorded_and_sidecar_verified(tmp_path):
    path = tmp_path / "foundation.json"
    raw = (json.dumps(_assertion_bundle(), sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )

    ir = load_foundation_ir(path)
    assert ir.source_sha256 == digest

    path.with_name(path.name + ".sha256").write_text(
        f"{'0' * 64}  {path.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(FoundationIRContractError):
        load_foundation_ir(path)
