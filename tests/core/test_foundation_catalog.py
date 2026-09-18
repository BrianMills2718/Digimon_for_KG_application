from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from Core.Projection.Catalog import generate_foundation_catalog, validate_foundation_catalog
from Core.Projection.Project import FoundationProject, build_foundation_project

HERE = Path(__file__).resolve().parents[1]
FIXTURE = HERE / "fixtures" / "foundation_demo"


def _project(tmp_path):
    return build_foundation_project(
        FIXTURE / "foundation.json",
        tmp_path / "project",
        passage_path=FIXTURE / "passages.json",
    )


def _links(path: Path):
    return re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8"))


def test_catalog_is_progressive_disclosure_environment_map(tmp_path):
    project = _project(tmp_path)
    output = project.root / "catalog" / project.manifest["generation_id"]
    result = generate_foundation_catalog(project._load_ir(), project.manifest, output)

    assert result.entity_count == project.manifest["counts"]["entities"]
    index = result.index_path.read_text(encoding="utf-8")
    assert "Representations and capabilities" in index
    assert "Entities" in index
    assert "Sources / evidence" in index

    reps = (output / "representations.md").read_text(encoding="utf-8")
    assert "Relational / SQLite" in reps
    assert "Graph neighborhood retrieval" in reps
    assert "Vector index" in reps
    assert "**unavailable**" in reps
    assert "vector" not in project.manifest["artifacts"]


def test_entity_page_exposes_shared_addresses_without_prescribing_workflow(tmp_path):
    project = _project(tmp_path)
    output = project.root / "catalog" / project.manifest["generation_id"]
    generate_foundation_catalog(project._load_ir(), project.manifest, output)
    manifest = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    page = output / manifest["entity_pages"]["entity:alice"]
    text = page.read_text(encoding="utf-8")

    assert "Canonical entity ID" in text and "entity:alice" in text
    assert "SQL" in text and "assertion_roles" in text
    assert "Binary graph" in text and "node ID `entity:alice`" in text
    assert "Vector" in text and "unavailable" in text
    assert "descriptive, not a workflow policy" in text


def test_catalog_links_resolve_within_catalog(tmp_path):
    project = _project(tmp_path)
    output = project.root / "catalog" / project.manifest["generation_id"]
    generate_foundation_catalog(project._load_ir(), project.manifest, output)

    for page in output.rglob("*.md"):
        for link in _links(page):
            if "://" in link or link.startswith("#"):
                continue
            target = (page.parent / link).resolve()
            assert target.is_relative_to(output.resolve()), (page, link)
            assert target.is_file(), (page, link)


def test_catalog_manifest_detects_changed_page(tmp_path):
    project = _project(tmp_path)
    output = project.root / "catalog" / project.manifest["generation_id"]
    generate_foundation_catalog(project._load_ir(), project.manifest, output)
    manifest = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    entity_page = output / manifest["entity_pages"]["entity:alice"]
    entity_page.write_text(entity_page.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed catalog file"):
        validate_foundation_catalog(output, expected_project_manifest=project.manifest)


def test_catalog_rejects_wrong_project_generation(tmp_path):
    project = _project(tmp_path)
    output = project.root / "catalog" / project.manifest["generation_id"]
    generate_foundation_catalog(project._load_ir(), project.manifest, output)
    wrong = dict(project.manifest)
    wrong["generation_id"] = "other"
    with pytest.raises(ValueError, match="generation mismatch"):
        validate_foundation_catalog(output, expected_project_manifest=wrong)


def test_source_pages_do_not_duplicate_exact_source_text(tmp_path):
    project = _project(tmp_path)
    ir = project._load_ir()
    output = project.root / "catalog" / project.manifest["generation_id"]
    generate_foundation_catalog(ir, project.manifest, output)
    manifest = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    passage = ir.passages[0]
    page = (output / manifest["passage_pages"][passage.passage_id]).read_text(encoding="utf-8")
    assert passage.text not in page
    assert passage.passage_id in page
    assert passage.source_ref in page


def test_catalog_refuses_overwrite_by_default(tmp_path):
    project = _project(tmp_path)
    output = project.root / "catalog" / project.manifest["generation_id"]
    generate_foundation_catalog(project._load_ir(), project.manifest, output)
    with pytest.raises(FileExistsError):
        generate_foundation_catalog(project._load_ir(), project.manifest, output)


def test_project_catalog_records_lineage_and_reopens(tmp_path):
    project = _project(tmp_path)
    result = project.generate_catalog()
    assert result["status"] == "ok"
    manifest_path = project.root / result["catalog_manifest"]["path"]
    assert manifest_path.is_file()
    validate_foundation_catalog(manifest_path.parent, expected_project_manifest=project.manifest)

    events = [json.loads(line) for line in (project.root / "executions.jsonl").read_text(encoding="utf-8").splitlines()]
    event = next(item for item in events if item["execution_id"] == result["execution_id"] and item["status"] != "started")
    assert event["operation"] == "catalog.generate"
    assert event["status"] == "succeeded"
    assert event["outputs"][0]["sha256"] == result["catalog_manifest"]["sha256"]

    reopened = FoundationProject.open(project.root)
    assert reopened.manifest["generation_id"] == project.manifest["generation_id"]
    validate_foundation_catalog(manifest_path.parent, expected_project_manifest=reopened.manifest)


def test_cli_build_and_reuse_can_generate_catalog(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "cli-project"
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_foundation_demo.py"
    base = [
        sys.executable, str(script),
        "--ir", str(FIXTURE / "foundation.json"),
        "--passages", str(FIXTURE / "passages.json"),
        "--output", str(output),
        "--entity-id", "entity:alice",
        "--catalog",
    ]
    first = subprocess.run(base, cwd=script.parents[1], text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    assert payload["catalog_result"]["status"] == "ok"
    assert "catalog" not in payload["not_integrated"]
    assert "vectors" in payload["not_integrated"]

    second = subprocess.run(base + ["--reuse"], cwd=script.parents[1], text=True, capture_output=True, check=False)
    assert second.returncode == 0, second.stderr
    reused = json.loads(second.stdout)
    assert reused["catalog_result"]["status"] == "ok"
    assert reused["catalog_result"]["reused"] is True
    assert "catalog" not in reused["not_integrated"]
    assert "vectors" in reused["not_integrated"]
