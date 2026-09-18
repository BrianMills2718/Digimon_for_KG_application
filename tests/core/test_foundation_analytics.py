from __future__ import annotations

import asyncio
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from Core.Projection.Analytics import (
    aggregate_foundation_predicates,
    analyze_foundation_subgraph,
    centrality_from_subgraph,
    leiden_runtime_status,
)
from Core.Projection.Project import build_foundation_project
from Core.Schema.SlotTypes import SlotKind, SlotValue, SubgraphRecord


def _write_fixture(root: Path):
    assertions = []
    passages = []
    edges = [
        ("a", "b", "rel:ab"),
        ("b", "c", "rel:bc"),
        ("c", "d", "rel:cd"),
    ]
    for index, (left, right, predicate) in enumerate(edges, 1):
        aid = f"assertion:{index}"
        pref = f"cand:{index}"
        pid = f"passage:{index}"
        assertions.append({
            "assertion_id": aid,
            "predicate": predicate,
            "claim_text": f"{left.upper()} relates to {right.upper()}.",
            "roles": {
                "left": [{"kind": "entity", "entity_id": f"entity:{left}", "name": left.upper(), "entity_type": "test:Node"}],
                "right": [{"kind": "entity", "entity_id": f"entity:{right}", "name": right.upper(), "entity_type": "test:Node"}],
            },
            "qualifiers": {},
            "provenance_refs": [pref],
        })
        passages.append({
            "passage_id": pid,
            "text": f"source {left}-{right}",
            "source_ref": f"source:{index}",
            "supporting_provenance_refs": [pref],
        })
    foundation = root / "foundation.json"
    companion = root / "passages.json"
    foundation.write_text(json.dumps({
        "format_version": "1.3", "producer": "onto-canon6",
        "assertion_count": len(assertions), "assertions": assertions,
    }), encoding="utf-8")
    companion.write_text(json.dumps({
        "format_version": "1.0", "producer": "onto-canon6",
        "passage_count": len(passages), "passages": passages,
    }), encoding="utf-8")
    return foundation, companion


def _project(tmp_path):
    source, passages = _write_fixture(tmp_path)
    return build_foundation_project(source, tmp_path / "project", passage_path=passages)


def test_centrality_slot_is_typed_and_aligned():
    graph = nx.path_graph(["entity:a", "entity:b", "entity:c", "entity:d"])
    slot = SlotValue(
        SlotKind.SUBGRAPH,
        SubgraphRecord(nodes=set(graph), edges=list(graph.edges()), nx_graph=graph),
    )
    scores = centrality_from_subgraph(slot, method="betweenness")
    assert scores.kind == SlotKind.SCORE_VECTOR
    assert scores.metadata["node_ids"] == ["entity:a", "entity:b", "entity:c", "entity:d"]
    assert np.isfinite(scores.data).all()
    mapping = dict(zip(scores.metadata["node_ids"], scores.data))
    assert mapping["entity:b"] == pytest.approx(mapping["entity:c"])
    assert mapping["entity:b"] > mapping["entity:a"]


@pytest.mark.parametrize("method", ["degree", "betweenness", "pagerank"])
def test_supported_centralities_are_finite(method):
    graph = nx.path_graph(["a", "b", "c"])
    slot = SlotValue(SlotKind.SUBGRAPH, SubgraphRecord(nodes=set(graph), edges=list(graph.edges()), nx_graph=graph))
    result = centrality_from_subgraph(slot, method=method)
    assert np.isfinite(result.data).all()
    assert result.metadata["derived_state"] is True


def test_unknown_metric_and_empty_graph_fail_closed():
    graph = nx.Graph()
    slot = SlotValue(SlotKind.SUBGRAPH, SubgraphRecord(nodes=set(), edges=[], nx_graph=graph))
    with pytest.raises(ValueError, match="unsupported"):
        centrality_from_subgraph(slot, method="magic")
    with pytest.raises(ValueError, match="nonempty"):
        centrality_from_subgraph(slot, method="degree")


def test_retrieve_analyze_then_recover_exact_evidence(tmp_path):
    project = _project(tmp_path)
    result = asyncio.run(analyze_foundation_subgraph(
        project, ["entity:a"], k=3, method="betweenness", top_n=2
    ))
    assert result["status"] == "ok"
    assert [item["entity_id"] for item in result["top_entities"]] == ["entity:b", "entity:c"]
    assert {item["passage_id"] for item in result["evidence"]} == {
        "passage:1", "passage:2", "passage:3"
    }
    assert {item["text"] for item in result["evidence"]} == {
        "source a-b", "source b-c", "source c-d"
    }
    assert "not source evidence" in result["interpretation_warning"]
    artifact = project.root / result["artifact"]["path"]
    assert artifact.is_file()


def test_analysis_scope_changes_with_retrieval_scope(tmp_path):
    project = _project(tmp_path)
    result = asyncio.run(analyze_foundation_subgraph(
        project, ["entity:a"], k=1, method="degree", top_n=2
    ))
    assert result["graph_scope"]["nodes"] == ["entity:a", "entity:b"]
    assert {item["passage_id"] for item in result["evidence"]} == {"passage:1"}


def test_sql_aggregation_is_exact_and_persisted(tmp_path):
    project = _project(tmp_path)
    result = aggregate_foundation_predicates(project)
    assert result["status"] == "ok"
    assert result["rows"] == [
        {"predicate": "rel:ab", "assertion_count": 1},
        {"predicate": "rel:bc", "assertion_count": 1},
        {"predicate": "rel:cd", "assertion_count": 1},
    ]
    assert (project.root / result["artifact"]["path"]).is_file()


def test_analytic_execution_is_lineaged_from_subgraph(tmp_path):
    project = _project(tmp_path)
    result = asyncio.run(analyze_foundation_subgraph(project, ["entity:a"], k=2, method="degree", top_n=1))
    events = [json.loads(line) for line in (project.root / "executions.jsonl").read_text(encoding="utf-8").splitlines()]
    terminal = next(item for item in events if item["execution_id"] == result["execution_id"] and item["status"] == "succeeded")
    assert terminal["operation"] == "analytics.centrality.degree"
    assert terminal["inputs"][0]["kind"] == "subgraph"
    assert terminal["outputs"][0]["kind"] == "analytic_result"


def test_leiden_runtime_status_is_truthful_for_environment():
    status = leiden_runtime_status()
    assert isinstance(status["available"], bool)
    if status["available"]:
        assert "dependency present" in status["reason"]
    else:
        assert "graspologic" in status["reason"]


@pytest.mark.parametrize("method", ["closeness", "eigenvector"])
def test_additional_centralities_are_finite_and_aligned(method):
    graph = nx.path_graph(["entity:a", "entity:b", "entity:c", "entity:d"])
    slot = SlotValue(
        SlotKind.SUBGRAPH,
        SubgraphRecord(nodes=set(graph), edges=list(graph.edges()), nx_graph=graph),
    )
    result = centrality_from_subgraph(slot, method=method)
    assert result.metadata["node_ids"] == ["entity:a", "entity:b", "entity:c", "entity:d"]
    assert np.isfinite(result.data).all()


def test_structural_summary_has_expected_path_graph_metrics():
    from Core.Projection.Analytics import structural_summary_from_subgraph

    graph = nx.path_graph(["a", "b", "c", "d"])
    slot = SlotValue(
        SlotKind.SUBGRAPH,
        SubgraphRecord(nodes=set(graph), edges=list(graph.edges()), nx_graph=graph),
    )
    summary = structural_summary_from_subgraph(slot)
    assert summary["node_count"] == 4
    assert summary["edge_count"] == 3
    assert summary["component_count"] == 1
    assert summary["components"] == [["a", "b", "c", "d"]]
    assert summary["isolates"] == []
    assert summary["density"] == pytest.approx(0.5)
    assert summary["transitivity"] == 0.0
    assert summary["average_clustering"] == 0.0
    assert summary["coreness"] == {"a": 1, "b": 1, "c": 1, "d": 1}
    assert summary["bridges"] == [("a", "b"), ("b", "c"), ("c", "d")]
    assert summary["articulation_points"] == ["b", "c"]
    assert summary["derived_state"] is True


def test_structural_summary_declares_self_loop_policy():
    from Core.Projection.Analytics import structural_summary_from_subgraph

    graph = nx.Graph()
    graph.add_edge("a", "a")
    graph.add_edge("a", "b")
    slot = SlotValue(
        SlotKind.SUBGRAPH,
        SubgraphRecord(nodes=set(graph), edges=list(graph.edges()), nx_graph=graph),
    )
    summary = structural_summary_from_subgraph(slot)
    assert summary["self_loops"] == [("a", "a")]
    assert summary["coreness"] == {"a": 1, "b": 1}
    assert "excluded" in summary["algorithm_policy"]["self_loops"]


def test_structural_summary_reports_undefined_assortativity_without_nan():
    from Core.Projection.Analytics import structural_summary_from_subgraph

    graph = nx.Graph()
    graph.add_node("only")
    slot = SlotValue(
        SlotKind.SUBGRAPH,
        SubgraphRecord(nodes={"only"}, edges=[], nx_graph=graph),
    )
    summary = structural_summary_from_subgraph(slot)
    assert summary["degree_assortativity"] is None
    assert summary["degree_assortativity_note"] == "undefined for a graph with no edges"
    assert summary["isolates"] == ["only"]


def test_retrieve_structural_summary_then_exact_evidence_and_lineage(tmp_path):
    project = _project(tmp_path)
    result = asyncio.run(project.analyze_structure(["entity:a"], k=3))
    assert result["status"] == "ok"
    assert result["metrics"]["node_count"] == 4
    assert result["metrics"]["component_count"] == 1
    assert result["metrics"]["articulation_points"] == ["entity:b", "entity:c"]
    assert {item["passage_id"] for item in result["evidence"]} == {
        "passage:1", "passage:2", "passage:3"
    }
    lineage = project.lineage_for_artifact(result["artifact"])
    operations = {item["operation"] for item in lineage["executions"]}
    assert "analytics.structural_summary" in operations
    assert "subgraph.khop_paths" in operations
