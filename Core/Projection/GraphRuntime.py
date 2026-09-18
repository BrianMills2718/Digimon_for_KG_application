"""Adapt saved Foundation graphs to the maintained typed graph operators.

The native MultiGraph stays intact. A simple, undirected assertion-association
view serves NetworkXStorage; every contributing assertion remains in edge data.
These are associations between claim participants, not an inferred truth graph.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Operators._context import OperatorContext
from Core.Operators.subgraph.khop_paths import subgraph_khop_paths
from Core.Operators.subgraph.materialize import subgraph_materialize
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue
from Core.Storage.NetworkXStorage import NetworkXStorage

from .Execution import ArtifactRef, file_sha256
from .FoundationIR import load_foundation_ir

if TYPE_CHECKING:
    from .Project import FoundationProject

GRAPH_RUNTIME_VERSION = "1.0"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _implementation_files() -> dict[str, str]:
    """Include actual consumers; the shared recorder fingerprints projectors."""
    from Core.Operators.subgraph import khop_paths, materialize
    from Core.Storage import NetworkXStorage as storage_module
    return {module.__name__: file_sha256(Path(module.__file__))
            for module in (khop_paths, materialize, storage_module)}


def _string_ids(encoded: str) -> list[str]:
    ids = json.loads(encoded)
    if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("invalid graph identity list")
    return ids


def simple_entity_view(binary: nx.MultiGraph, *, predicates: list[str] | None = None) -> nx.Graph:
    """One unweighted edge per pair; parallel assertion records are never erased.

    Predicate filters select assertion edges *before* pair aggregation. No role
    direction, polarity truth, confidence weight, or n-ary clique is inferred.
    Self-loops are retained explicitly. Names remain display data, never keys.
    """
    if binary.is_directed() or not binary.is_multigraph():
        raise ValueError("expected an undirected binary assertion MultiGraph")
    if binary.graph.get("projection_kind") != "foundation_binary_entity_graph":
        raise ValueError("not a Foundation binary entity projection")
    if predicates is not None and (
        not isinstance(predicates, list) or not all(isinstance(p, str) and p for p in predicates)
    ):
        raise ValueError("predicates must be a list of nonempty predicate IDs or None")
    selected = None if predicates is None else set(predicates)
    graph = nx.Graph(**{k: v for k, v in binary.graph.items() if k not in ("node_default", "edge_default")})
    graph.graph.update(
        projection_kind="foundation_runtime_entity_association",
        runtime_version=GRAPH_RUNTIME_VERSION,
        graph_semantics="undirected binary assertion association; not affirmative truth",
        parallel_policy="one edge per pair; all assertion records retained",
        weight_policy="unit per pair; not confidence or assertion count",
        self_loop_policy="retain", polarity_policy="retained in assertions; not interpreted",
        predicate_filter_json=_json(None if selected is None else sorted(selected)),
        node_evidence_policy="selected edges only; entity-wide evidence is a separate SQL lookup",
    )
    for node, data in sorted(binary.nodes(data=True)):
        if data.get("node_kind") != "entity" or data.get("entity_id") != node:
            raise ValueError(f"invalid canonical entity node: {node}")
        attributes = dict(data)
        attributes.update(entity_name=node, display_name=data.get("entity_name", node),
                          source_id="", passage_ids_json="[]", description="")
        graph.add_node(node, **attributes)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    skipped = []
    seen = set()
    for src, tgt, key, data in binary.edges(keys=True, data=True):
        assertion_id = data.get("assertion_id")
        if not assertion_id or assertion_id != str(key) or assertion_id in seen:
            raise ValueError("missing/duplicate/mismatched parallel assertion identity")
        seen.add(assertion_id)
        if selected is not None and data["predicate"] not in selected:
            skipped.append(assertion_id)
            continue
        pair = tuple(sorted((src, tgt)))
        groups.setdefault(pair, []).append(dict(data))
    for (src, tgt), assertions in sorted(groups.items()):
        assertions.sort(key=lambda a: a["assertion_id"])
        passages = sorted({pid for a in assertions for pid in _string_ids(a["passage_ids_json"])})
        names = sorted({a["predicate"] for a in assertions})
        graph.add_edge(src, tgt, src_id=src, tgt_id=tgt, weight=1.0,
                       relation_name=GRAPH_FIELD_SEP.join(names), keywords="", description="",
                       source_id=GRAPH_FIELD_SEP.join(passages), passage_ids_json=_json(passages),
                       assertion_ids_json=_json([a["assertion_id"] for a in assertions]),
                       assertions_json=_json(assertions), assertion_count=len(assertions))
    graph.graph["filtered_assertion_ids_json"] = _json(sorted(skipped))
    return graph


class FoundationPassageLookup:
    """Read-only adaptation to the existing doc_chunks.get_data_by_key protocol."""
    def __init__(self, ir):
        self.ir = ir

    async def get_data_by_key(self, key: str) -> dict[str, Any] | None:
        passage = self.ir.passages_by_id.get(key)
        if passage is None:
            return None
        return {"text": passage.text, "source_ref": passage.source_ref,
                "namespace_id": passage.namespace_id, "source_registry_id": passage.source_registry_id,
                "source_urls": list(passage.source_urls),
                "supporting_provenance_refs": list(passage.supporting_provenance_refs),
                "assertion_ids": sorted({aid for ref in passage.supporting_provenance_refs
                                         for aid in self.ir.assertions_by_provenance_ref.get(ref, ())})}


async def open_foundation_graph_context(
    project: FoundationProject, *, predicates: list[str] | None = None,
) -> tuple[OperatorContext, dict[str, Any]]:
    """Load checked artifacts, publish a reproducible view, bind real storage.

    The caller receives the real OperatorContext and can compose existing
    operators directly. No LLM, global GraphRAGContext, or MCP server is started.
    """
    from .Project import _inside
    artifacts = project.manifest["artifacts"]
    names = ["foundation", "binary_graph"] + (["passages"] if "passages" in artifacts else [])
    refs = [artifacts[name] for name in names]
    with project.log.operation("graph.runtime_adapter", inputs=refs,
                               parameters={"predicates": predicates, "version": GRAPH_RUNTIME_VERSION}) as run:
        paths = {}
        for name in names:
            ref = artifacts[name]
            path = _inside(project.root, ref["path"])
            if ref["input_digests"] != project.manifest["input_digests"] or file_sha256(path) != ref["sha256"]:
                raise ValueError(f"stale graph input artifact: {name}")
            paths[name] = path
        ir = load_foundation_ir(paths["foundation"], passage_path=paths.get("passages"))
        binary = nx.read_graphml(paths["binary_graph"], force_multigraph=True)
        view = simple_entity_view(binary, predicates=predicates)
        # Scope to the supported input, even when a caller holds a project open.
        if set(view) != set(ir.entities_by_id):
            raise ValueError("graph/IR entity identity mismatch")
        for src, tgt, data in view.edges(data=True):
            for assertion in json.loads(data["assertions_json"]):
                record = ir.assertions_by_id.get(assertion["assertion_id"])
                if record is None or record.predicate != assertion["predicate"]:
                    raise ValueError("graph/IR assertion mismatch")
                occurrences = [f.entity_id for fillers in record.roles.values()
                               for f in fillers if f.kind == "entity"]
                if sorted(occurrences) != sorted((src, tgt)):
                    raise ValueError("graph/IR role endpoint mismatch")
                expected = {p.passage_id for p in ir.passages_for_assertion(record.assertion_id)}
                if set(_string_ids(assertion["passage_ids_json"])) != expected:
                    raise ValueError("graph/IR passage identity mismatch")
        directory = project.root / "observations"
        directory.mkdir(exist_ok=True)
        path = directory / f"{run.execution_id}.graphml"
        nx.write_graphml(view, path)
        # Bind the exact persisted representation, not a separate demo graph.
        restored = nx.read_graphml(path)
        storage = NetworkXStorage()
        for node, data in restored.nodes(data=True):
            await storage.upsert_node(node, dict(data))
        for src, tgt, data in restored.edges(data=True):
            await storage.upsert_edge(src, tgt, dict(data))
        storage.graph.graph.update(restored.graph)
        ref = asdict(ArtifactRef.from_file(project.root, path, "runtime_graph", project.manifest["input_digests"]))
        ref["producing_execution"] = run.execution_id
        run.outputs = [ref]
        run.diagnostics = {"nodes": storage.get_node_num(), "pairs": storage.get_edge_num(),
                           "assertions": sum(d["assertion_count"] for _, _, d in view.edges(data=True)),
                           "skipped_nonbinary": project.manifest["graph_omissions"],
                           "filtered_assertions": json.loads(view.graph["filtered_assertion_ids_json"]),
                           "consumer_implementations": _implementation_files()}
        return OperatorContext(graph=storage, doc_chunks=FoundationPassageLookup(ir)), ref


async def retrieve_foundation_subgraph(
    project: FoundationProject, entity_ids: list[str], *, k: int = 2,
    predicates: list[str] | None = None,
) -> dict[str, SlotValue]:
    """Real khop -> materialize -> typed working set, with exact saved lineage."""
    from .Project import _write_json
    if not isinstance(entity_ids, list) or not entity_ids or not all(isinstance(x, str) and x for x in entity_ids):
        raise ValueError("entity_ids must be a nonempty list of canonical IDs")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    ctx, graph_ref = await open_foundation_graph_context(project, predicates=predicates)
    with project.log.operation("subgraph.khop_paths", inputs=[graph_ref],
                               parameters={"entity_ids": entity_ids, "k": k, "mode": "neighbors"}) as run:
        unknown = sorted(set(entity_ids) - set(ctx.graph.graph))
        if unknown:
            run.diagnostics = {"unknown_entity_ids": unknown}
            raise KeyError(f"unknown canonical entity IDs: {unknown}")
        seeds = SlotValue(SlotKind.ENTITY_SET, [EntityRecord(entity_name=x) for x in sorted(set(entity_ids))])
        slots = await subgraph_khop_paths({"entities": seeds}, ctx, {"k": k, "mode": "neighbors"})
        subgraph = slots["subgraph"]
        if subgraph.metadata.get("error"):
            raise RuntimeError(subgraph.metadata["error"])
        bounded = (await ctx.graph.get_induced_subgraph(sorted(subgraph.data.nodes))).copy()
        if set(subgraph.data.nodes) != set(bounded) or set(subgraph.data.edges) != {tuple(sorted(e)) for e in bounded.edges()}:
            raise ValueError("operator returned nodes/edges outside its graph view")
        subgraph.data.nx_graph = bounded
        selected_assertions = sorted({aid for _, _, d in bounded.edges(data=True) for aid in _string_ids(d["assertion_ids_json"])})
        selected_passages = sorted({pid for _, _, d in bounded.edges(data=True) for pid in _string_ids(d["passage_ids_json"])})
        evidence_gaps = sorted(aid for aid in selected_assertions if not ctx.doc_chunks.ir.passages_for_assertion(aid))
        graph_path = project.root / "observations" / f"{run.execution_id}.graphml"
        nx.write_graphml(bounded, graph_path)
        subgraph_ref = asdict(ArtifactRef.from_file(project.root, graph_path, "subgraph", project.manifest["input_digests"]))
        subgraph_ref["producing_execution"] = run.execution_id
        subgraph.metadata.update(artifact=subgraph_ref, input_graph=graph_ref,
                                 assertion_ids=selected_assertions, passage_ids=selected_passages)
        run.outputs = [subgraph_ref]
        run.diagnostics = {"nodes": sorted(bounded), "edges": subgraph.data.edges,
                           "assertion_ids": selected_assertions, "consumer_implementations": _implementation_files()}
    with project.log.operation("subgraph.materialize", inputs=[subgraph_ref, *([project.manifest["artifacts"]["passages"]] if "passages" in project.manifest["artifacts"] else [])],
                               parameters={"evidence_scope": "selected_edges_only"}) as run:
        materialized = await subgraph_materialize({"subgraph": subgraph}, ctx)
        chunks = materialized["chunks"].data
        if {chunk.chunk_id for chunk in chunks} != set(selected_passages):
            raise ValueError("materializer failed to resolve the exact selected passage set")
        for chunk in chunks:
            original = ctx.doc_chunks.ir.passages_by_id[chunk.chunk_id]
            if chunk.text != original.text:
                raise ValueError("materializer changed original source text")
            chunk.extra["assertion_ids"] = sorted(set(chunk.extra.get("assertion_ids", ())) & set(selected_assertions))
        status = ("partial_evidence" if evidence_gaps else "ok") if chunks else "insufficient_evidence"
        report = {"status": status, "entity_ids": entity_ids,
                  "assertions_without_passages": evidence_gaps,
                  "nodes": sorted(bounded), "edges": sorted(tuple(sorted(e)) for e in bounded.edges()),
                  "assertion_ids": selected_assertions, "evidence": [asdict(chunk) for chunk in chunks],
                  "graph_scope": {key: value for key, value in bounded.graph.items() if key not in ("node_default", "edge_default")},
                  "skipped_nonbinary_assertions": project.manifest["graph_omissions"],
                  "input_digests": project.manifest["input_digests"], "subgraph_artifact": subgraph_ref,
                  "runtime_graph_artifact": graph_ref, "execution_id": run.execution_id}
        output = project.root / "observations" / f"{run.execution_id}.json"
        _write_json(output, report)
        ref = asdict(ArtifactRef.from_file(project.root, output, "graph_evidence", project.manifest["input_digests"]))
        ref["producing_execution"] = run.execution_id
        run.outputs = [ref]
        run.diagnostics = {"passage_count": len(chunks), "status": report["status"]}
        for value in materialized.values():
            value.metadata.update(artifact=ref, report=report, subgraph_artifact=subgraph_ref)
        slots.update(materialized)
        return slots
