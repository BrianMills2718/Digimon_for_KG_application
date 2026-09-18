"""Deterministic finding artifacts over derived analytical results.

A finding records an inspectable interpretation boundary and exact evidence /
derivation references. It does not ask an LLM to invent conclusions.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from .Execution import ArtifactRef
from .Lineage import trace_artifact_lineage

if TYPE_CHECKING:
    from .Project import FoundationProject

FINDING_FORMAT_VERSION = "1.0"


def create_analytic_finding(
    project: "FoundationProject",
    analytic_result: dict[str, Any],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    """Persist a bounded finding derived from one recorded analytic result."""
    from .Project import _write_json

    if analytic_result.get("status") != "ok":
        raise ValueError("finding requires a successful analytic result")
    analytic_ref = analytic_result.get("artifact")
    if not isinstance(analytic_ref, dict) or not analytic_ref.get("artifact_id"):
        raise ValueError("analytic result is missing its artifact identity")
    top_entities = analytic_result.get("top_entities")
    evidence = analytic_result.get("evidence")
    if not isinstance(top_entities, list) or not top_entities:
        raise ValueError("finding requires at least one analytical entity")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("finding requires source evidence")

    lineage = trace_artifact_lineage(project.log.path, analytic_ref)
    method = analytic_result.get("method")
    parameters = analytic_result.get("parameters") or {}
    ids = [str(item["entity_id"]) for item in top_entities]
    headline = title or f"Top {method} entities in the retrieved graph working set"
    summary = (
        f"Within the recorded retrieved graph scope, the highest ranked {method} "
        f"entities are: {', '.join(ids)}."
    )
    evidence_refs = [
        {
            "passage_id": item.get("passage_id"),
            "assertion_ids": list(item.get("assertion_ids") or []),
            "source_ref": item.get("source_ref"),
        }
        for item in evidence
    ]

    with project.log.operation(
        "finding.from_analytic_result",
        inputs=[analytic_ref],
        parameters={"format_version": FINDING_FORMAT_VERSION, "title": headline},
        parent_execution_id=analytic_result.get("execution_id"),
    ) as run:
        finding = {
            "format_version": FINDING_FORMAT_VERSION,
            "status": "derived_finding",
            "finding_id": f"finding:{run.execution_id}",
            "execution_id": run.execution_id,
            "title": headline,
            "summary": summary,
            "method": method,
            "parameters": parameters,
            "graph_scope": analytic_result.get("graph_scope"),
            "ranked_entities": top_entities,
            "evidence_refs": evidence_refs,
            "analytic_artifact": analytic_ref,
            "lineage_root_artifact_id": analytic_ref["artifact_id"],
            "lineage_execution_ids": [item["execution_id"] for item in lineage["executions"]],
            "limitations": [
                analytic_result.get("interpretation_warning"),
                "The result is conditional on the retrieved graph scope and declared projection semantics.",
                "Ranking does not establish causal influence, importance outside this working set, or source truth.",
            ],
            "input_digests": project.manifest["input_digests"],
        }
        directory = project.root / "findings"
        directory.mkdir(exist_ok=True)
        output = directory / f"{run.execution_id}.json"
        _write_json(output, finding)
        ref = asdict(ArtifactRef.from_file(
            project.root, output, "finding", project.manifest["input_digests"]
        ))
        ref["producing_execution"] = run.execution_id
        run.outputs = [ref]
        run.diagnostics = {
            "ranked_entity_ids": ids,
            "evidence_passage_ids": [item.get("passage_id") for item in evidence_refs],
            "lineage_execution_count": len(lineage["executions"]),
        }
        return {**finding, "artifact": ref, "lineage": lineage}
