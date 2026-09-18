"""Persist/reopen the existing Foundation projections as one inspectable project.

A new generation is built separately. Publishing its manifest never deletes the
last usable generation. This is a single-writer local artifact API, not a new
retrieval planner or replacement graph runtime.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

import networkx as nx

from .Execution import ArtifactRef, ExecutionLog, file_sha256, implementation_digest
from .FoundationIR import load_foundation_ir
from .Identity import build_identity_manifest
from .PropertyGraph import (
    assertion_graph_to_foundation_payload,
    project_foundation_ir_to_assertion_graph,
    project_foundation_ir_to_binary_entity_graph,
)
from .Relational import project_foundation_ir_to_sqlite

PROJECT_FORMAT_VERSION = "1.0"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("project artifact path must be relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("project artifact path escapes project")
    return resolved


class FoundationProject:
    def __init__(self, root: Path, manifest: dict[str, Any]):
        self.root = root.resolve()
        self.manifest = manifest
        self.log = ExecutionLog(self.root / "executions.jsonl")

    @classmethod
    def open(cls, path: str | Path, *, expected_inputs: dict[str, str | None] | None = None) -> "FoundationProject":
        root = Path(path).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return cls._validated(root, manifest, expected_inputs=expected_inputs)

    @classmethod
    def _validated(cls, root: Path, manifest: dict[str, Any], *, expected_inputs: dict[str, str | None] | None = None) -> "FoundationProject":
        if manifest.get("format_version") != PROJECT_FORMAT_VERSION:
            raise ValueError("unsupported project manifest version")
        if expected_inputs is not None and manifest["input_digests"] != expected_inputs:
            raise ValueError("stale project: input snapshot/companion mismatch")
        required = {"foundation", "relational", "assertion_graph", "binary_graph"}
        if not required.issubset(manifest["artifacts"]):
            raise ValueError("project manifest is missing required artifacts")
        for kind in ("foundation", "passages"):
            expected = manifest["input_digests"][kind]
            artifact = manifest["artifacts"].get(kind)
            if (artifact is None) != (expected is None) or (artifact is not None and artifact["sha256"] != expected):
                raise ValueError(f"input/artifact digest mismatch: {kind}")
        for name, artifact in manifest["artifacts"].items():
            if artifact["input_digests"] != manifest["input_digests"]:
                raise ValueError(f"artifact input mismatch: {name}")
            if artifact["artifact_id"] != f"{artifact['kind']}:sha256:{artifact['sha256']}":
                raise ValueError(f"artifact identity mismatch: {name}")
            target = _inside(root, artifact["path"])
            if not target.is_file() or file_sha256(target) != artifact["sha256"]:
                raise ValueError(f"missing/changed project artifact: {name}")
        return cls(root, manifest)

    def _load_ir(self):
        foundation = _inside(self.root, self.manifest["artifacts"]["foundation"]["path"])
        passage_artifact = self.manifest["artifacts"].get("passages")
        passages = _inside(self.root, passage_artifact["path"]) if passage_artifact is not None else None
        return load_foundation_ir(foundation, passage_path=passages, validate_sidecar=False)

    def generate_catalog(self, *, overwrite: bool = False) -> dict[str, Any]:
        """Generate or revalidate the progressive-disclosure catalog for this generation."""
        from .Catalog import generate_foundation_catalog, validate_foundation_catalog

        output = self.root / "catalog" / self.manifest["generation_id"]
        existing = output / "catalog.json"
        operation = "catalog.generate" if overwrite or not existing.is_file() else "catalog.reuse"
        with self.log.operation(
            operation,
            inputs=list(self.manifest["artifacts"].values()),
            parameters={"generation_id": self.manifest["generation_id"], "overwrite": overwrite},
        ) as run:
            if existing.is_file() and not overwrite:
                catalog = validate_foundation_catalog(output, expected_project_manifest=self.manifest)
                manifest_path = existing
                index_path = output / catalog["entry_point"]
                counts = {
                    "entities": len(catalog["entity_pages"]),
                    "assertions": len(catalog["assertion_pages"]),
                    "passages": len(catalog["passage_pages"]),
                }
                reused = True
            else:
                result = generate_foundation_catalog(
                    self._load_ir(), self.manifest, output, overwrite=overwrite
                )
                manifest_path = result.manifest_path
                index_path = result.index_path
                counts = {
                    "entities": result.entity_count,
                    "assertions": result.assertion_count,
                    "passages": result.passage_count,
                }
                reused = False
            ref = ArtifactRef.from_file(
                self.root, manifest_path, "catalog_manifest", self.manifest["input_digests"]
            )
            run.outputs = [asdict(ref)]
            run.diagnostics = {
                "entry_point": index_path.relative_to(self.root).as_posix(),
                "counts": counts,
                "reused": reused,
            }
            return {
                "status": "ok",
                "execution_id": run.execution_id,
                "catalog_manifest": asdict(ref),
                "entry_point": index_path.relative_to(self.root).as_posix(),
                "counts": counts,
                "reused": reused,
            }

    def _connection(self) -> sqlite3.Connection:
        artifact = self.manifest["artifacts"]["relational"]
        path = _inside(self.root, artifact["path"])
        # Recheck the selected resource: callers may hold this object across edits.
        if file_sha256(path) != artifact["sha256"]:
            raise ValueError("stale relational artifact")
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    async def graph_neighborhood(self, entity_ids: list[str], *, k: int = 2,
                                 predicates: list[str] | None = None) -> dict[str, Any]:
        """Existing typed graph retrieval -> exact evidence over this snapshot.

        For composable typed outputs, use GraphRuntime.retrieve_foundation_subgraph.
        This convenience method returns the persisted JSON observation.
        """
        from .GraphRuntime import retrieve_foundation_subgraph
        slots = await retrieve_foundation_subgraph(self, entity_ids, k=k, predicates=predicates)
        metadata = slots["chunks"].metadata
        return {**metadata["report"], "artifact": metadata["artifact"]}

    async def analyze_graph(self, entity_ids: list[str], *, k: int = 2, method: str = "betweenness", top_n: int = 5, predicates: list[str] | None = None) -> dict[str, Any]:
        """Retrieve a bounded graph, compute centrality, and recover scoped evidence."""
        from .Analytics import analyze_foundation_subgraph
        return await analyze_foundation_subgraph(
            self, entity_ids, k=k, method=method, top_n=top_n, predicates=predicates
        )

    def aggregate_predicates(self) -> dict[str, Any]:
        """Exact relational assertion counts grouped by predicate."""
        from .Analytics import aggregate_foundation_predicates
        return aggregate_foundation_predicates(self)

    def lineage_for_artifact(self, artifact: str | dict[str, Any]) -> dict[str, Any]:
        """Trace a retained artifact backward through successful executions."""
        from .Lineage import trace_artifact_lineage
        return trace_artifact_lineage(self.log.path, artifact)

    def create_finding(self, analytic_result: dict[str, Any], *, title: str | None = None) -> dict[str, Any]:
        """Persist a bounded finding over one successful analytical result."""
        from .Finding import create_analytic_finding
        return create_analytic_finding(self, analytic_result, title=title)

    def evidence_for_entity(self, entity_id: str) -> dict[str, Any]:
        """Exact entity -> assertion -> producer reference -> original passage."""
        artifact = self.manifest["artifacts"]["relational"]
        with self.log.operation("relational.evidence_for_entity", inputs=[artifact], parameters={"entity_id": entity_id}) as run:
            conn = self._connection()
            try:
                exists = conn.execute("SELECT 1 FROM entities WHERE entity_id = ?", (entity_id,)).fetchone()
                rows = conn.execute("""
                    SELECT DISTINCT a.assertion_id, a.predicate, p.passage_id,
                           p.source_ref, p.namespace_id, p.source_registry_id, p.text
                    FROM assertion_roles r
                    JOIN assertions a USING(assertion_id)
                    JOIN assertion_provenance ap USING(assertion_id)
                    JOIN passage_support ps ON ps.provenance_ref = ap.provenance_ref
                    JOIN passages p USING(passage_id)
                    WHERE r.entity_id = ?
                    ORDER BY a.assertion_id, p.passage_id
                """, (entity_id,)).fetchall()
            finally:
                conn.close()
            result = {
                "status": "unknown_entity" if not exists else ("ok" if rows else "insufficient_evidence"),
                "entity_id": entity_id, "execution_id": run.execution_id,
                "evidence": [dict(row) for row in rows],
                "input_digests": self.manifest["input_digests"],
            }
            directory = self.root / "observations"
            directory.mkdir(exist_ok=True)
            output = directory / f"{run.execution_id}.json"
            _write_json(output, result)
            ref = ArtifactRef.from_file(self.root, output, "evidence_set", self.manifest["input_digests"])
            run.outputs = [asdict(ref)]
            run.diagnostics = {"evidence_count": len(rows), "result_status": result["status"]}
            return {**result, "artifact": asdict(ref)}


def build_foundation_project(
    source_path: str | Path, output_path: str | Path, *,
    passage_path: str | Path | None = None, overwrite: bool = False,
) -> FoundationProject:
    """Build source/SQLite/GraphML artifacts, validate them, then publish once.

    No LLM call, graph-runtime registration, vector index, catalog, or analytics
    is implied by building these artifacts. Failed generations retain inputs as
    local reproducers; they are never advertised in the active manifest.
    """
    root = Path(output_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(manifest_path)
    generation_id = str(uuid4())
    published = root / "generations" / generation_id
    published.mkdir(parents=True)
    # Stable locations keep successful child artifacts/reproducers reopenable
    # even when the enclosing project build fails. Only the manifest publishes.
    stage = published
    log = ExecutionLog(root / "executions.jsonl")
    with log.operation("project.build", inputs=[{"path": str(source_path)}, {"path": str(passage_path) if passage_path else None}], parameters={"overwrite": overwrite}) as run:
        try:
            # Capture bytes before parsing so the persisted inputs are the exact
            # inputs we validate and project, even if originals later change.
            for origin, name in [(source_path, "foundation.json"), (passage_path, "passages.json")]:
                if origin is None:
                    continue
                origin = Path(origin)
                (stage / name).write_bytes(origin.read_bytes())
                sidecar = origin.with_name(origin.name + ".sha256")
                if sidecar.is_file():
                    (stage / f"{name}.sha256").write_bytes(sidecar.read_bytes())
            ir = load_foundation_ir(stage / "foundation.json", passage_path=stage / "passages.json" if passage_path is not None else None)
            digests = {"foundation": ir.source_sha256, "passages": ir.passage_sha256}
            run.inputs[:] = [
                {"kind": kind, "artifact_id": f"{kind}:sha256:{digest}", "sha256": digest, "path": (published / f"{kind}.json").relative_to(root).as_posix()}
                for kind, digest in digests.items() if digest is not None
            ]
            # Every material transformation gets its own execution and parents.
            paths = {"foundation": stage / "foundation.json"}
            if passage_path is not None:
                paths["passages"] = stage / "passages.json"
            source_refs = [dict(item) for item in run.inputs]
            producers = {}
            with log.operation("projection.relational", inputs=source_refs, parameters={"engine": "sqlite"}, parent_execution_id=run.execution_id) as step:
                paths["relational"] = stage / "knowledge.sqlite"
                project_foundation_ir_to_sqlite(ir, paths["relational"])
                producers["relational"] = step.execution_id
                step.outputs = [asdict(ArtifactRef.from_file(root, paths["relational"], "relational", digests))]
            with log.operation("projection.graphs", inputs=source_refs, parameters={"binary_policy": "exactly_two_entity_occurrences; undirected"}, parent_execution_id=run.execution_id) as step:
                assertion_graph = project_foundation_ir_to_assertion_graph(ir)
                binary = project_foundation_ir_to_binary_entity_graph(ir)
                paths["assertion_graph"] = stage / "assertions.graphml"
                paths["binary_graph"] = stage / "entities.graphml"
                nx.write_graphml(assertion_graph, paths["assertion_graph"])
                nx.write_graphml(binary.graph, paths["binary_graph"])
                recovered = assertion_graph_to_foundation_payload(nx.read_graphml(paths["assertion_graph"], force_multigraph=True))
                if recovered != ir.to_payload():
                    raise ValueError("assertion graph field-preservation check failed")
                step.outputs = [asdict(ArtifactRef.from_file(root, paths[name], name, digests)) for name in ("assertion_graph", "binary_graph")]
                producers.update({name: step.execution_id for name in ("assertion_graph", "binary_graph")})
                step.diagnostics = {"skipped_nonbinary_assertions": list(binary.skipped_assertion_ids)}
            artifacts = {}
            for name, path in paths.items():
                ref = asdict(ArtifactRef.from_file(stage, path, name, digests))
                ref["path"] = (published / path.name).relative_to(root).as_posix()
                ref["producing_execution"] = producers.get(name)
                artifacts[name] = ref
            identity = build_identity_manifest(ir)
            manifest = {
                "format_version": PROJECT_FORMAT_VERSION, "generation_id": generation_id,
                "producing_execution": run.execution_id, "input_digests": digests,
                "implementation_sha256": implementation_digest(),
                "artifacts": artifacts,
                "scope": {"namespace_ids": identity.namespace_ids, "source_registry_ids": identity.source_registry_ids},
                "counts": {"entities": len(identity.entity_ids), "assertions": len(ir.assertions), "passages": len(ir.passages)},
                "graph_omissions": list(binary.skipped_assertion_ids),
                "not_integrated": ["graph_runtime", "vectors", "catalog", "analytics", "external_harness"],
            }
            candidate = stage / "manifest.json"
            _write_json(candidate, manifest)
            validated = FoundationProject._validated(root, json.loads(candidate.read_text(encoding="utf-8")))
            # Write a temporary manifest file, then atomically switch the front
            # door. Prior generations are deliberately retained, never deleted.
            temporary_manifest = root / f".manifest-{generation_id}.json"
            try:
                temporary_manifest.write_bytes(candidate.read_bytes())
                if overwrite:
                    os.replace(temporary_manifest, manifest_path)
                else:
                    os.link(temporary_manifest, manifest_path)
            finally:
                temporary_manifest.unlink(missing_ok=True)
            run.outputs = list(artifacts.values())
            run.diagnostics = {"counts": manifest["counts"], "generation_id": generation_id}
            return validated
        except BaseException:
            if stage.exists():
                run.diagnostics["reproducer_ref"] = stage.relative_to(root).as_posix()
                run.diagnostics["generation_state"] = "unpublished_failure"
            raise
