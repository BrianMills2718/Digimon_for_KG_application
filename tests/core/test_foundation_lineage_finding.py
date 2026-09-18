from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys

import pytest

from Core.Projection.Execution import ExecutionLog
from Core.Projection.Lineage import load_terminal_executions, trace_artifact_lineage
from Core.Projection.Project import FoundationProject, build_foundation_project

HERE = Path(__file__).resolve().parents[1]
FIXTURE = HERE / "fixtures" / "foundation_demo"


def _project(tmp_path: Path):
    return build_foundation_project(
        FIXTURE / "foundation.json",
        tmp_path / "project",
        passage_path=FIXTURE / "passages.json",
    )


def test_analytic_artifact_lineage_reaches_projection_inputs(tmp_path):
    project = _project(tmp_path)
    result = asyncio.run(project.analyze_graph(
        ["entity:alice"], k=1, method="degree", top_n=2
    ))
    lineage = project.lineage_for_artifact(result["artifact"])

    operations = {item["operation"] for item in lineage["executions"]}
    assert "analytics.centrality.degree" in operations
    assert "subgraph.khop_paths" in operations
    assert "graph.runtime_adapter" in operations
    artifact_kinds = {item.get("kind") for item in lineage["artifacts"]}
    assert "subgraph" in artifact_kinds
    assert "runtime_graph" in artifact_kinds
    assert "foundation" in artifact_kinds
    assert "binary_graph" in artifact_kinds


def test_failed_execution_is_not_a_successful_artifact_producer(tmp_path):
    log = ExecutionLog(tmp_path / "executions.jsonl")
    with pytest.raises(RuntimeError):
        with log.operation("test.failure", inputs=[], parameters={}) as run:
            run.outputs = [{
                "artifact_id": "fake:sha256:deadbeef",
                "kind": "fake",
                "path": "fake",
                "sha256": "deadbeef",
                "input_digests": {},
            }]
            raise RuntimeError("boom")
    records = load_terminal_executions(log.path)
    assert records[0]["status"] == "failed"
    assert records[0]["outputs"] == []
    with pytest.raises(KeyError):
        trace_artifact_lineage(log.path, "fake:sha256:deadbeef")


def test_finding_persists_analysis_evidence_limitations_and_lineage(tmp_path):
    project = _project(tmp_path)
    analytic = asyncio.run(project.analyze_graph(
        ["entity:alice"], k=1, method="betweenness", top_n=2
    ))
    finding = project.create_finding(analytic)

    assert finding["status"] == "derived_finding"
    assert finding["method"] == "betweenness"
    assert finding["analytic_artifact"]["artifact_id"] == analytic["artifact"]["artifact_id"]
    assert finding["evidence_refs"]
    assert all(item["passage_id"] for item in finding["evidence_refs"])
    assert any("causal influence" in item for item in finding["limitations"])
    assert analytic["execution_id"] in finding["lineage_execution_ids"]
    assert (project.root / finding["artifact"]["path"]).is_file()

    terminal = [
        item for item in load_terminal_executions(project.log.path)
        if item["execution_id"] == finding["execution_id"]
    ][0]
    assert terminal["operation"] == "finding.from_analytic_result"
    assert terminal["parent_execution_id"] == analytic["execution_id"]
    assert terminal["outputs"][0]["artifact_id"] == finding["artifact"]["artifact_id"]


def test_finding_refuses_unobserved_or_unsupported_analysis(tmp_path):
    project = _project(tmp_path)
    with pytest.raises(ValueError, match="successful analytic"):
        project.create_finding({"status": "insufficient_evidence"})
    with pytest.raises(ValueError, match="artifact identity"):
        project.create_finding({
            "status": "ok",
            "top_entities": [{"entity_id": "entity:alice", "score": 1.0}],
            "evidence": [{"passage_id": "x"}],
        })


def test_lineage_rejects_unknown_artifact_and_invalid_log(tmp_path):
    path = tmp_path / "executions.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line"):
        load_terminal_executions(path)

    path.write_text("", encoding="utf-8")
    with pytest.raises(KeyError):
        trace_artifact_lineage(path, "missing:sha256:abc")


def test_finding_survives_project_reopen(tmp_path):
    project = _project(tmp_path)
    analytic = asyncio.run(project.analyze_graph(
        ["entity:alice"], k=1, method="degree", top_n=1
    ))
    finding = project.create_finding(analytic)

    reopened = FoundationProject.open(project.root)
    traced = reopened.lineage_for_artifact(finding["artifact"])
    operations = {item["operation"] for item in traced["executions"]}
    assert "finding.from_analytic_result" in operations
    assert "analytics.centrality.degree" in operations


def test_cli_build_and_reuse_can_emit_finding(tmp_path):
    output = tmp_path / "cli-project"
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_foundation_demo.py"
    base = [
        sys.executable, str(script),
        "--ir", str(FIXTURE / "foundation.json"),
        "--passages", str(FIXTURE / "passages.json"),
        "--output", str(output),
        "--entity-id", "entity:alice",
        "--graph-hops", "1",
        "--centrality", "degree",
        "--top-n", "2",
        "--finding",
        "--catalog",
        "--aggregate-predicates",
    ]
    first = subprocess.run(base, cwd=script.parents[1], text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    assert payload["finding_result"]["status"] == "derived_finding"
    assert payload["finding_result"]["evidence_refs"]

    second = subprocess.run(base + ["--reuse"], cwd=script.parents[1], text=True, capture_output=True, check=False)
    assert second.returncode == 0, second.stderr
    reused = json.loads(second.stdout)
    assert reused["finding_result"]["status"] == "derived_finding"
    assert reused["catalog_result"]["reused"] is True
