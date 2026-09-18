"""Typed analytical transforms over retrieved Foundation working sets.

This module promotes a small, executable analytics surface without pretending
that graph-derived metrics are source evidence. The first slice supports
centrality over the exact retrieved subgraph and relational aggregation. Leiden
remains unavailable unless the repository's real community dependency path is
present and integrated; it is never substituted with a differently named
algorithm.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from Core.Schema.SlotTypes import SlotKind, SlotValue

from .Execution import ArtifactRef

if TYPE_CHECKING:
    from .Project import FoundationProject

ANALYTICS_VERSION = "1.0"
SUPPORTED_CENTRALITY = ("degree", "betweenness", "pagerank", "closeness", "eigenvector")


def _ordered_nodes(graph: nx.Graph) -> list[str]:
    return sorted(str(node) for node in graph.nodes())


def centrality_from_subgraph(
    subgraph: SlotValue,
    *,
    method: str = "betweenness",
) -> SlotValue:
    """SUBGRAPH -> SCORE_VECTOR using declared NetworkX algorithms.

    Scores align to metadata['node_ids']; they are derived analytical state,
    not evidence and not a claim of causal/social importance.
    """
    if subgraph.kind != SlotKind.SUBGRAPH:
        raise TypeError("centrality requires a SUBGRAPH slot")
    if method not in SUPPORTED_CENTRALITY:
        raise ValueError(f"unsupported centrality method: {method}")
    record = subgraph.data
    graph = getattr(record, "nx_graph", None)
    if graph is None or not isinstance(graph, nx.Graph):
        raise ValueError("subgraph does not carry its attributed NetworkX working set")
    nodes = _ordered_nodes(graph)
    if not nodes:
        raise ValueError("centrality requires a nonempty working set")

    if method == "degree":
        scores = nx.degree_centrality(graph)
    elif method == "betweenness":
        scores = nx.betweenness_centrality(graph, normalized=True, weight=None)
    elif method == "pagerank":
        scores = nx.pagerank(graph, alpha=0.85, weight="weight")
    elif method == "closeness":
        scores = nx.closeness_centrality(graph)
    else:
        try:
            scores = nx.eigenvector_centrality(graph, max_iter=1000, tol=1.0e-9, weight="weight")
        except nx.PowerIterationFailedConvergence as exc:
            raise ValueError("eigenvector centrality did not converge") from exc
    values = np.asarray([float(scores[node]) for node in nodes], dtype=float)
    if values.shape != (len(nodes),) or not np.isfinite(values).all():
        raise ValueError("centrality returned a non-finite or misaligned score vector")
    return SlotValue(
        kind=SlotKind.SCORE_VECTOR,
        data=values,
        producer=f"analytics.centrality.{method}",
        metadata={
            "node_ids": nodes,
            "method": method,
            "analytics_version": ANALYTICS_VERSION,
            "derived_state": True,
            "interpretation_warning": "centrality is structural derived state, not source evidence or causal influence",
        },
    )


def _top_scores(scores: SlotValue, top_n: int) -> list[dict[str, Any]]:
    if scores.kind != SlotKind.SCORE_VECTOR:
        raise TypeError("expected SCORE_VECTOR")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    nodes = scores.metadata.get("node_ids")
    if not isinstance(nodes, list) or len(nodes) != len(scores.data):
        raise ValueError("score-vector node alignment is missing or invalid")
    ranked = sorted(
        ((str(node), float(score)) for node, score in zip(nodes, scores.data)),
        key=lambda item: (-item[1], item[0]),
    )
    return [{"entity_id": node, "score": score} for node, score in ranked[:top_n]]


def _selected_evidence_for_entities(
    graph: nx.Graph,
    top_entities: list[dict[str, Any]],
    chunks: list[Any],
) -> list[dict[str, Any]]:
    wanted = {item["entity_id"] for item in top_entities}
    assertion_ids: set[str] = set()
    for src, tgt, data in graph.edges(data=True):
        if src not in wanted and tgt not in wanted:
            continue
        encoded = data.get("assertion_ids_json", "[]")
        ids = json.loads(encoded)
        if isinstance(ids, list):
            assertion_ids.update(str(item) for item in ids)
    evidence = []
    seen = set()
    for chunk in chunks:
        chunk_assertions = set(chunk.extra.get("assertion_ids", ()))
        matched = sorted(chunk_assertions & assertion_ids)
        if not matched or chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        evidence.append({
            "passage_id": chunk.chunk_id,
            "assertion_ids": matched,
            "source_ref": chunk.extra.get("source_ref"),
            "namespace_id": chunk.extra.get("namespace_id"),
            "source_registry_id": chunk.extra.get("source_registry_id"),
            "text": chunk.text,
        })
    return evidence


async def analyze_foundation_subgraph(
    project: "FoundationProject",
    entity_ids: list[str],
    *,
    k: int = 2,
    method: str = "betweenness",
    top_n: int = 5,
    predicates: list[str] | None = None,
) -> dict[str, Any]:
    """Retrieve -> centrality -> selected-entity evidence with lineage."""
    from .GraphRuntime import retrieve_foundation_subgraph
    from .Project import _write_json

    slots = await retrieve_foundation_subgraph(
        project, entity_ids, k=k, predicates=predicates
    )
    subgraph = slots["subgraph"]
    subgraph_ref = subgraph.metadata["artifact"]
    with project.log.operation(
        f"analytics.centrality.{method}",
        inputs=[subgraph_ref],
        parameters={
            "entity_ids": entity_ids,
            "k": k,
            "method": method,
            "top_n": top_n,
            "predicates": predicates,
            "graph_scope": "retrieved_subgraph",
        },
    ) as run:
        score_slot = centrality_from_subgraph(subgraph, method=method)
        top_entities = _top_scores(score_slot, top_n)
        graph = subgraph.data.nx_graph
        evidence = _selected_evidence_for_entities(
            graph, top_entities, slots["chunks"].data
        )
        report = {
            "status": "ok" if evidence else "insufficient_evidence",
            "execution_id": run.execution_id,
            "method": method,
            "parameters": {"top_n": top_n, "k": k, "predicates": predicates},
            "graph_scope": {
                "seed_entity_ids": list(entity_ids),
                "nodes": sorted(str(node) for node in graph.nodes()),
                "edges": sorted(tuple(sorted((str(src), str(tgt)))) for src, tgt in graph.edges()),
                "subgraph_artifact": subgraph_ref,
            },
            "scores": [
                {"entity_id": node, "score": float(score)}
                for node, score in zip(score_slot.metadata["node_ids"], score_slot.data)
            ],
            "top_entities": top_entities,
            "evidence": evidence,
            "interpretation_warning": score_slot.metadata["interpretation_warning"],
            "input_digests": project.manifest["input_digests"],
        }
        output = project.root / "observations" / f"{run.execution_id}.json"
        _write_json(output, report)
        ref = asdict(ArtifactRef.from_file(
            project.root, output, "analytic_result", project.manifest["input_digests"]
        ))
        ref["producing_execution"] = run.execution_id
        run.outputs = [ref]
        run.diagnostics = {
            "node_count": len(graph),
            "edge_count": graph.number_of_edges(),
            "top_entity_ids": [item["entity_id"] for item in top_entities],
            "evidence_count": len(evidence),
        }
        return {**report, "artifact": ref}



def structural_summary_from_subgraph(subgraph: SlotValue) -> dict[str, Any]:
    """Return inspectable SNA structure metrics for the exact SUBGRAPH working set.

    Self-loops are retained in topology counts but removed for algorithms whose
    mathematical contract rejects them (coreness/bridges/articulation). This
    policy is returned explicitly rather than silently changing the graph.
    """
    if subgraph.kind != SlotKind.SUBGRAPH:
        raise TypeError("structural summary requires a SUBGRAPH slot")
    graph = getattr(subgraph.data, "nx_graph", None)
    if graph is None or not isinstance(graph, nx.Graph):
        raise ValueError("subgraph does not carry its attributed NetworkX working set")
    if graph.number_of_nodes() == 0:
        raise ValueError("structural summary requires a nonempty working set")

    simple = nx.Graph(graph)
    self_loops = sorted((str(u), str(v)) for u, v in nx.selfloop_edges(simple))
    loopless = simple.copy()
    loopless.remove_edges_from(nx.selfloop_edges(loopless))
    components = [sorted(str(node) for node in component) for component in nx.connected_components(loopless)]
    components.sort(key=lambda nodes: (-len(nodes), nodes))
    isolates = sorted(str(node) for node in nx.isolates(loopless))
    coreness = nx.core_number(loopless) if loopless.number_of_nodes() else {}
    bridges = sorted(tuple(sorted((str(u), str(v)))) for u, v in nx.bridges(loopless))
    articulation = sorted(str(node) for node in nx.articulation_points(loopless))

    assortativity: float | None = None
    assortativity_reason: str | None = None
    if loopless.number_of_edges() == 0:
        assortativity_reason = "undefined for a graph with no edges"
    else:
        with np.errstate(all="ignore"):
            raw_assortativity = float(nx.degree_assortativity_coefficient(loopless))
        if np.isfinite(raw_assortativity):
            assortativity = raw_assortativity
        else:
            assortativity_reason = "undefined for this degree distribution"

    return {
        "analytics_version": ANALYTICS_VERSION,
        "derived_state": True,
        "node_count": simple.number_of_nodes(),
        "edge_count": simple.number_of_edges(),
        "self_loops": self_loops,
        "component_count": len(components),
        "components": components,
        "isolates": isolates,
        "density": float(nx.density(simple)),
        "transitivity": float(nx.transitivity(loopless)),
        "average_clustering": float(nx.average_clustering(loopless)),
        "coreness": {str(node): int(value) for node, value in sorted(coreness.items(), key=lambda item: str(item[0]))},
        "bridges": bridges,
        "articulation_points": articulation,
        "degree_assortativity": assortativity,
        "degree_assortativity_note": assortativity_reason,
        "algorithm_policy": {
            "direction": "undirected",
            "parallel_edges": "runtime association view already aggregated by entity pair",
            "self_loops": "retained for counts/density; excluded from coreness, bridge, articulation and clustering calculations",
            "weights": "topology metrics unweighted; centrality methods declare weight use separately",
        },
        "interpretation_warning": "structural metrics are derived from the retrieved association view; they are not source evidence, causal influence, or population-wide claims",
    }


async def analyze_foundation_structure(
    project: "FoundationProject",
    entity_ids: list[str],
    *,
    k: int = 2,
    predicates: list[str] | None = None,
) -> dict[str, Any]:
    """Retrieve -> structural SNA summary -> exact selected-subgraph evidence."""
    from .GraphRuntime import retrieve_foundation_subgraph
    from .Project import _write_json

    slots = await retrieve_foundation_subgraph(project, entity_ids, k=k, predicates=predicates)
    subgraph = slots["subgraph"]
    subgraph_ref = subgraph.metadata["artifact"]
    with project.log.operation(
        "analytics.structural_summary",
        inputs=[subgraph_ref],
        parameters={
            "entity_ids": entity_ids,
            "k": k,
            "predicates": predicates,
            "graph_scope": "retrieved_subgraph",
        },
    ) as run:
        summary = structural_summary_from_subgraph(subgraph)
        evidence = [
            {
                "passage_id": chunk.chunk_id,
                "assertion_ids": sorted(chunk.extra.get("assertion_ids", ())),
                "source_ref": chunk.extra.get("source_ref"),
                "namespace_id": chunk.extra.get("namespace_id"),
                "source_registry_id": chunk.extra.get("source_registry_id"),
                "text": chunk.text,
            }
            for chunk in slots["chunks"].data
        ]
        graph = subgraph.data.nx_graph
        report = {
            "status": "ok" if evidence else "insufficient_evidence",
            "execution_id": run.execution_id,
            "analysis": "structural_summary",
            "parameters": {"k": k, "predicates": predicates},
            "graph_scope": {
                "seed_entity_ids": list(entity_ids),
                "nodes": sorted(str(node) for node in graph.nodes()),
                "edges": sorted(tuple(sorted((str(src), str(tgt)))) for src, tgt in graph.edges()),
                "subgraph_artifact": subgraph_ref,
            },
            "metrics": summary,
            "evidence": evidence,
            "input_digests": project.manifest["input_digests"],
        }
        output = project.root / "observations" / f"{run.execution_id}.json"
        _write_json(output, report)
        ref = asdict(ArtifactRef.from_file(
            project.root, output, "structural_analysis", project.manifest["input_digests"]
        ))
        ref["producing_execution"] = run.execution_id
        run.outputs = [ref]
        run.diagnostics = {
            "node_count": summary["node_count"],
            "edge_count": summary["edge_count"],
            "component_count": summary["component_count"],
            "evidence_count": len(evidence),
        }
        return {**report, "artifact": ref}

def aggregate_foundation_predicates(project: "FoundationProject") -> dict[str, Any]:
    """Exact SQL aggregation over the relational projection, with lineage."""
    from .Project import _write_json

    artifact = project.manifest["artifacts"]["relational"]
    with project.log.operation(
        "analytics.sql.predicate_counts",
        inputs=[artifact],
        parameters={"group_by": "predicate", "measure": "assertion_count"},
    ) as run:
        conn = project._connection()
        try:
            rows = conn.execute(
                "SELECT predicate, COUNT(*) AS assertion_count FROM assertions GROUP BY predicate ORDER BY predicate"
            ).fetchall()
        finally:
            conn.close()
        result = {
            "status": "ok",
            "execution_id": run.execution_id,
            "aggregation": "assertion_count_by_predicate",
            "rows": [dict(row) for row in rows],
            "input_digests": project.manifest["input_digests"],
        }
        directory = project.root / "observations"
        directory.mkdir(exist_ok=True)
        output = directory / f"{run.execution_id}.json"
        _write_json(output, result)
        ref = asdict(ArtifactRef.from_file(
            project.root, output, "table_aggregate", project.manifest["input_digests"]
        ))
        ref["producing_execution"] = run.execution_id
        run.outputs = [ref]
        run.diagnostics = {"row_count": len(rows)}
        return {**result, "artifact": ref}


def leiden_runtime_status() -> dict[str, Any]:
    """Report the real Leiden dependency state without substituting algorithms."""
    try:
        import graspologic  # noqa: F401
    except ImportError as exc:
        return {
            "available": False,
            "reason": "graspologic is not installed in this execution environment",
            "error": str(exc),
        }
    return {
        "available": True,
        "reason": "dependency present; saved-project Leiden adapter still requires consumer integration",
    }
