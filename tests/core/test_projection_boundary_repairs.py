"""Counterexamples independent of projector success flags (synthetic inputs)."""
import copy
import hashlib
import importlib
import json

import networkx as nx
import pytest

from Core.Projection.FoundationIR import FoundationIRContractError, load_foundation_ir, parse_foundation_ir
from tests.core.test_foundation_ir_contract import _assertion_bundle, _passage_bundle

pg = importlib.import_module("Core.Projection.PropertyGraph")
rel = importlib.import_module("Core.Projection.Relational")


def test_producer_content_hash_is_opaque_and_preserved_verbatim():
    passages = _passage_bundle()
    passages["passages"][0]["content_hash"] = "MD5:AbCd1234"
    ir = parse_foundation_ir(_assertion_bundle(), passage_payload=passages)
    assert ir.passages[0].content_hash == "MD5:AbCd1234"


def test_empty_optional_text_is_not_missing():
    bundle = _assertion_bundle()
    bundle["assertions"][0]["claim_text"] = ""
    bundle["assertions"][0]["roles"]["since"][0]["raw"] = ""
    ir = parse_foundation_ir(bundle)
    assert ir.assertions[0].claim_text == ""
    assert ir.assertions[0].roles["since"][0].raw == ""


@pytest.mark.parametrize("field", ["namespace_id", "source_registry_id"])
def test_companion_must_match_assertion_source_scope(field):
    passages = _passage_bundle()
    passages["passages"][0][field] = "different-scope"
    with pytest.raises(FoundationIRContractError, match="scope"):
        parse_foundation_ir(_assertion_bundle(), passage_payload=passages)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_confidence_rejected(number):
    bundle = _assertion_bundle()
    bundle["assertions"][0]["confidence"] = number
    with pytest.raises(FoundationIRContractError):
        parse_foundation_ir(bundle)


def test_failed_sqlite_rebuild_keeps_last_good_output(tmp_path, monkeypatch):
    ir = parse_foundation_ir(_assertion_bundle(), passage_payload=_passage_bundle())
    target = tmp_path / "knowledge.sqlite"
    rel.project_foundation_ir_to_sqlite(ir, target)
    before = target.read_bytes()

    def fail(_):
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(rel, "_json", fail)
    with pytest.raises(RuntimeError, match="injected"):
        rel.project_foundation_ir_to_sqlite(ir, target, overwrite=True)
    assert target.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["knowledge.sqlite"]


def test_repeated_reference_values_do_not_break_relational_build(tmp_path):
    bundle = _assertion_bundle()
    bundle["assertions"][0]["source_urls"] *= 2
    bundle["assertions"][0]["provenance_refs"] *= 2
    ir = parse_foundation_ir(bundle, passage_payload=_passage_bundle())
    result = rel.project_foundation_ir_to_sqlite(ir, tmp_path / "knowledge.sqlite")
    assert result.assertion_count == 1


def test_graph_refuses_collision_between_entity_and_assertion_ids():
    bundle = _assertion_bundle()
    bundle["assertions"][0]["assertion_id"] = "entity:alice"
    with pytest.raises(FoundationIRContractError, match="collision"):
        pg.project_foundation_ir_to_assertion_graph(parse_foundation_ir(bundle))


def test_graph_refuses_collision_with_projection_local_value_id():
    bundle = _assertion_bundle()
    bundle["assertions"][0]["roles"]["employee"][0]["entity_id"] = "value::gassert2_abc::since::0"
    with pytest.raises(FoundationIRContractError, match="collision"):
        pg.project_foundation_ir_to_assertion_graph(parse_foundation_ir(bundle))


def test_graph_roundtrip_retains_filler_local_data_and_null_presence(tmp_path):
    bundle = _assertion_bundle()
    first = bundle["assertions"][0]
    first["roles"]["unused"] = []
    first["roles"]["since"][0].update(value=None, normalized={"year": 2025, "precise": False})
    second = copy.deepcopy(first)
    second["assertion_id"] = "gassert2_second"
    second.pop("claim_text")
    second["roles"]["employee"][0].update(name="Alice alternate", alias_ids=[])
    second["roles"]["since"][0].pop("value")
    bundle["assertions"].append(second)
    bundle["assertion_count"] = 2
    graph = pg.project_foundation_ir_to_assertion_graph(parse_foundation_ir(bundle))
    path = tmp_path / "assertions.graphml"
    nx.write_graphml(graph, path)
    reopened = nx.read_graphml(path, force_multigraph=True)
    assert pg.assertion_graph_to_foundation_payload(reopened) == bundle


def test_source_payload_is_not_aliased_to_callers_mutable_input():
    bundle = _assertion_bundle()
    ir = parse_foundation_ir(bundle)
    bundle["assertions"][0]["qualifiers"]["future:additive-qualifier"]["preserve"] = False
    assert ir.assertions[0].qualifiers["future:additive-qualifier"]["preserve"] is True


def test_both_input_file_digests_are_retained(tmp_path):
    assertion_path = tmp_path / "foundation.json"
    passage_path = tmp_path / "passages.json"
    assertion_path.write_text(json.dumps(_assertion_bundle()))
    passage_path.write_text(json.dumps(_passage_bundle()))
    ir = load_foundation_ir(assertion_path, passage_path=passage_path)
    assert ir.source_sha256 == hashlib.sha256(assertion_path.read_bytes()).hexdigest()
    assert ir.passage_sha256 == hashlib.sha256(passage_path.read_bytes()).hexdigest()


def test_passage_file_sidecar_is_checked(tmp_path):
    assertion_path = tmp_path / "foundation.json"
    passage_path = tmp_path / "passages.json"
    assertion_path.write_text(json.dumps(_assertion_bundle()))
    passage_path.write_text(json.dumps(_passage_bundle()))
    passage_path.with_name("passages.json.sha256").write_text("0" * 64)
    with pytest.raises(FoundationIRContractError, match="mismatch"):
        load_foundation_ir(assertion_path, passage_path=passage_path)


def test_roundtrip_rejects_missing_role_edge():
    graph = pg.project_foundation_ir_to_assertion_graph(parse_foundation_ir(_assertion_bundle()))
    source, target, key = next(iter(graph.edges(keys=True)))
    graph.remove_edge(source, target, key)
    with pytest.raises(FoundationIRContractError, match="missing role"):
        pg.assertion_graph_to_foundation_payload(graph)


def test_relational_preserves_original_field_presence(tmp_path):
    bundle = _assertion_bundle()
    bundle["assertions"][0]["roles"]["since"][0]["value"] = None
    target = tmp_path / "knowledge.sqlite"
    rel.project_foundation_ir_to_sqlite(parse_foundation_ir(bundle), target)
    import sqlite3
    conn = sqlite3.connect(target)
    try:
        original = json.loads(conn.execute("SELECT payload_json FROM assertions").fetchone()[0])
    finally:
        conn.close()
    assert original == bundle["assertions"][0]
