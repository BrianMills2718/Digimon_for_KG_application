"""Query retained DIGIMON artifact/execution lineage from the local JSONL log.

The execution log remains the write seam. This module builds a small read model
on demand; it is not a second provenance store or domain graph.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Any


LINEAGE_QUERY_VERSION = "1.0"


def load_terminal_executions(path: str | Path) -> list[dict[str, Any]]:
    """Return one terminal event per execution, preserving append order."""
    path = Path(path)
    if not path.is_file():
        return []
    terminal: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid execution log JSON at line {line_number}") from exc
        if record.get("record_type") != "execution_event":
            continue
        execution_id = record.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError(f"execution event at line {line_number} has no execution_id")
        if record.get("status") not in {"succeeded", "failed"}:
            continue
        if execution_id not in terminal:
            order.append(execution_id)
        terminal[execution_id] = record
    return [terminal[item] for item in order]


def _artifact_id(ref: Any) -> str | None:
    if not isinstance(ref, dict):
        return None
    value = ref.get("artifact_id")
    return value if isinstance(value, str) and value else None


def _index(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    executions = {record["execution_id"]: record for record in records}
    produced_by: dict[str, list[str]] = {}
    for record in records:
        if record.get("status") != "succeeded":
            continue
        for output in record.get("outputs") or []:
            artifact_id = _artifact_id(output)
            if artifact_id is None:
                continue
            declared = output.get("producing_execution") if isinstance(output, dict) else None
            producer = declared if isinstance(declared, str) and declared else record["execution_id"]
            if producer != record["execution_id"]:
                continue
            bucket = produced_by.setdefault(artifact_id, [])
            if producer not in bucket:
                bucket.append(producer)
    return executions, produced_by

def trace_artifact_lineage(
    path: str | Path,
    artifact: str | dict[str, Any],
) -> dict[str, Any]:
    """Trace one execution-qualified artifact backward through successful runs.

    Content-addressed artifacts may be reproduced by more than one execution.
    Artifact references carrying ``producing_execution`` disambiguate that case.
    A bare artifact ID is accepted only when the producer is unambiguous.
    """
    artifact_id = artifact if isinstance(artifact, str) else _artifact_id(artifact)
    producer_hint = (
        artifact.get("producing_execution")
        if isinstance(artifact, dict) and isinstance(artifact.get("producing_execution"), str)
        else None
    )
    if not artifact_id:
        raise ValueError("artifact identity is required")
    records = load_terminal_executions(path)
    executions, produced_by = _index(records)

    def choose_producer(identifier: str, hint: str | None) -> str | None:
        candidates = produced_by.get(identifier, [])
        if hint is not None:
            if hint not in candidates:
                raise ValueError(
                    f"artifact {identifier} does not record producing execution {hint}"
                )
            return hint
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"artifact {identifier} has multiple producing executions; pass an execution-qualified artifact reference"
            )
        return None

    root_execution = choose_producer(artifact_id, producer_hint)
    if root_execution is None:
        raise KeyError(f"no successful producing execution for artifact {artifact_id}")

    visited_exec: set[str] = set()
    visited_artifact_exec: set[tuple[str, str | None]] = set()
    queue: deque[tuple[str, str, str | None]] = deque(
        [("artifact", artifact_id, root_execution)]
    )
    edges: list[dict[str, str]] = []
    artifact_refs: dict[tuple[str, str | None], dict[str, Any]] = {}

    while queue:
        kind, identifier, hint = queue.popleft()
        if kind == "artifact":
            key = (identifier, hint)
            if key in visited_artifact_exec:
                continue
            visited_artifact_exec.add(key)
            producer = choose_producer(identifier, hint)
            if producer is not None:
                edges.append({"from": producer, "to": identifier, "relation": "produced"})
                queue.append(("execution", producer, None))
            continue

        if identifier in visited_exec:
            continue
        visited_exec.add(identifier)
        record = executions.get(identifier)
        if record is None:
            continue
        parent = record.get("parent_execution_id")
        if isinstance(parent, str) and parent:
            edges.append({"from": parent, "to": identifier, "relation": "parent_execution"})
            queue.append(("execution", parent, None))
        for item in record.get("inputs") or []:
            input_id = _artifact_id(item)
            if input_id is None:
                continue
            input_hint = (
                item.get("producing_execution")
                if isinstance(item, dict) and isinstance(item.get("producing_execution"), str)
                else None
            )
            artifact_refs.setdefault((input_id, input_hint), dict(item))
            edges.append({"from": input_id, "to": identifier, "relation": "consumed"})
            queue.append(("artifact", input_id, input_hint))
        for item in record.get("outputs") or []:
            output_id = _artifact_id(item)
            if output_id is not None:
                output_hint = (
                    item.get("producing_execution")
                    if isinstance(item, dict) and isinstance(item.get("producing_execution"), str)
                    else record["execution_id"]
                )
                artifact_refs.setdefault((output_id, output_hint), dict(item))

    selected = [executions[item] for item in visited_exec if item in executions]
    selected.sort(key=lambda item: (item.get("at", ""), item["execution_id"]))
    unique_edges = {tuple(sorted(edge.items())) for edge in edges}
    edge_records = [dict(items) for items in sorted(unique_edges, key=repr)]
    return {
        "schema_version": "1.0",
        "query_version": LINEAGE_QUERY_VERSION,
        "root_artifact_id": artifact_id,
        "producing_execution_id": root_execution,
        "executions": selected,
        "artifacts": [artifact_refs[key] for key in sorted(artifact_refs, key=repr)],
        "edges": edge_records,
        "failed_execution_ids": sorted(
            record["execution_id"] for record in records if record.get("status") == "failed"
        ),
    }

