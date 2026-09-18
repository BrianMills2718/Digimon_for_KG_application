"""Exercise actual files, SQLite, GraphML, subprocess reopening, and failures."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from Core.Projection.Execution import file_sha256
from Core.Projection.Project import FoundationProject, build_foundation_project
from tests.core.test_foundation_ir_contract import _assertion_bundle, _passage_bundle


@pytest.fixture
def source_files(tmp_path):
    source, passages = tmp_path / "input.json", tmp_path / "windows.json"
    source.write_text(json.dumps(_assertion_bundle()))
    passages.write_text(json.dumps(_passage_bundle()))
    return source, passages


def test_build_reopen_query_and_trace(source_files, tmp_path):
    source, passages = source_files
    root = tmp_path / "project"
    project = build_foundation_project(source, root, passage_path=passages)
    reopened = FoundationProject.open(root)
    result = reopened.evidence_for_entity("entity:alice")
    assert result["status"] == "ok"
    assert result["evidence"][0]["text"] == _passage_bundle()["passages"][0]["text"]
    assert result["evidence"][0]["passage_id"] == "gpassage1_abc"
    assert file_sha256(root / result["artifact"]["path"]) == result["artifact"]["sha256"]
    events = [json.loads(line) for line in (root / "executions.jsonl").read_text().splitlines()]
    terminal = [event for event in events if event["status"] == "succeeded"]
    assert {event["operation"] for event in terminal} == {"project.build", "projection.relational", "projection.graphs", "relational.evidence_for_entity"}
    assert all(event["implementation_sha256"] for event in terminal)
    assert project.manifest["input_digests"] == {"foundation": file_sha256(source), "passages": file_sha256(passages)}
    assert {"vectors", "external_harness"}.issubset(project.manifest["not_integrated"])


def test_unknown_entity_and_absent_evidence_are_explicit(source_files, tmp_path):
    source, passages = source_files
    project = build_foundation_project(source, tmp_path / "project")
    assert project.evidence_for_entity("entity:alice")["status"] == "insufficient_evidence"
    assert project.evidence_for_entity("entity:not-present")["status"] == "unknown_entity"


def test_reuse_rejects_changed_companion(source_files, tmp_path):
    source, passages = source_files
    root = tmp_path / "project"
    build_foundation_project(source, root, passage_path=passages)
    passages.write_text(passages.read_text() + "\n")
    with pytest.raises(ValueError, match="stale"):
        FoundationProject.open(root, expected_inputs={"foundation": file_sha256(source), "passages": file_sha256(passages)})


def test_changed_artifact_is_rejected(source_files, tmp_path):
    source, passages = source_files
    project = build_foundation_project(source, tmp_path / "project", passage_path=passages)
    artifact = project.root / project.manifest["artifacts"]["binary_graph"]["path"]
    artifact.write_text("corrupt")
    with pytest.raises(ValueError, match="changed"):
        FoundationProject.open(project.root)


def test_failed_rebuild_preserves_manifest_and_records_failure(source_files, tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("Core.Projection.Project")
    source, passages = source_files
    root = tmp_path / "project"
    build_foundation_project(source, root, passage_path=passages)
    original = (root / "manifest.json").read_bytes()

    def fail(_):
        raise RuntimeError("injected graph failure")

    monkeypatch.setattr(module, "project_foundation_ir_to_assertion_graph", fail)
    with pytest.raises(RuntimeError, match="injected"):
        build_foundation_project(source, root, passage_path=passages, overwrite=True)
    assert (root / "manifest.json").read_bytes() == original
    assert FoundationProject.open(root).evidence_for_entity("entity:alice")["status"] == "ok"
    events = [json.loads(line) for line in (root / "executions.jsonl").read_text().splitlines()]
    failures = [event for event in events if event["status"] == "failed"]
    assert {e["operation"] for e in failures} == {"projection.graphs", "project.build"}
    assert all(e["outputs"] == [] for e in failures)
    build_failure = next(e for e in failures if e["operation"] == "project.build")
    assert (root / build_failure["diagnostics"]["reproducer_ref"] / "foundation.json").exists()


def test_cli_build_and_reuse_in_new_process(source_files, tmp_path):
    source, passages = source_files
    script = Path(__file__).resolve().parents[2] / "scripts/run_foundation_demo.py"
    command = [sys.executable, str(script), "--ir", str(source), "--passages", str(passages), "--output", str(tmp_path / "project"), "--entity-id", "entity:alice"]
    first = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert first.returncode == 0, first.stderr
    second = subprocess.run(command + ["--reuse"], capture_output=True, text=True, timeout=20)
    assert second.returncode == 0, second.stderr
    assert json.loads(first.stdout)["result"]["evidence"] == json.loads(second.stdout)["result"]["evidence"]


def test_execution_outputs_have_stable_reopenable_artifact_references(source_files, tmp_path):
    source, passages = source_files
    project = build_foundation_project(source, tmp_path / "project", passage_path=passages)
    project.evidence_for_entity("entity:alice")
    events = [json.loads(line) for line in (project.root / "executions.jsonl").read_text().splitlines()]
    for event in events:
        if event["status"] != "succeeded":
            continue
        for artifact in event["outputs"]:
            assert file_sha256(project.root / artifact["path"]) == artifact["sha256"]
            assert artifact["artifact_id"] == f"{artifact['kind']}:sha256:{artifact['sha256']}"
    sql = project.manifest["artifacts"]["relational"]
    producer = next(e for e in events if e["execution_id"] == sql["producing_execution"] and e["status"] == "succeeded")
    assert producer["parent_execution_id"] == project.manifest["producing_execution"]
    assert producer["outputs"][0]["artifact_id"] == sql["artifact_id"]


def test_validation_failure_before_publish_keeps_previous_generation(source_files, tmp_path, monkeypatch):
    source, passages = source_files
    root = tmp_path / "project"
    build_foundation_project(source, root, passage_path=passages)
    original = (root / "manifest.json").read_bytes()

    def reject(*args, **kwargs):
        raise ValueError("injected manifest validation failure")

    monkeypatch.setattr(FoundationProject, "_validated", reject)
    with pytest.raises(ValueError, match="injected"):
        build_foundation_project(source, root, passage_path=passages, overwrite=True)
    assert (root / "manifest.json").read_bytes() == original


def test_manifest_cannot_point_outside_project(source_files, tmp_path):
    source, passages = source_files
    project = build_foundation_project(source, tmp_path / "project", passage_path=passages)
    manifest_path = project.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["foundation"]["path"] = "../input.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        FoundationProject.open(project.root)
