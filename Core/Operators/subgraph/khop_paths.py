"""Subgraph k-hop neighborhood/path operator."""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue, SubgraphRecord


def _edge_record_endpoints(edge):
    if isinstance(edge, dict):
        src = edge.get("src_id") or edge.get("source")
        tgt = edge.get("tgt_id") or edge.get("target")
        if src is not None and tgt is not None:
            return str(src), str(tgt)
    elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
        return str(edge[0]), str(edge[1])
    return None


def _split_connected_edge_sequence(raw_edges) -> list[list[str]]:
    """Turn an edge sequence into real contiguous node paths.

    NetworkXStorage may concatenate several paths found from the same seed into
    one list of edge records. A discontinuity therefore starts a new path; it
    must never be represented as an artificial edge between the two segments.
    """
    paths: list[list[str]] = []
    current: list[str] = []

    for raw_edge in raw_edges or []:
        endpoints = _edge_record_endpoints(raw_edge)
        if endpoints is None:
            continue
        src, tgt = endpoints

        if not current:
            current = [src, tgt]
        elif current[-1] == src:
            current.append(tgt)
        elif current[-1] == tgt:
            current.append(src)
        else:
            paths.append(current)
            current = [src, tgt]

    if current:
        paths.append(current)
    return paths


async def subgraph_khop_paths(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": ENTITY_SET}
    Outputs: {"subgraph": SUBGRAPH}
    Params:  {"k": int, "cutoff": int, "mode": "neighbors"|"paths"}
    """
    entities = inputs["entities"].data
    if not entities:
        return {
            "subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=SubgraphRecord(nodes=set(), edges=[]),
                producer="subgraph.khop_paths",
            )
        }

    p = params or {}
    k = max(1, int(p.get("k", 2)))
    cutoff = max(1, int(p.get("cutoff", k)))
    mode = p.get("mode", "neighbors")
    names = [record.entity_name for record in entities]

    try:
        if mode == "paths":
            raw_paths = await ctx.graph.get_paths_from_sources(
                start_nodes=names,
                cutoff=cutoff,
            )
            all_nodes = set(names)
            all_edges = []
            normalized_paths = []

            for raw_path in raw_paths or []:
                for path_nodes in _split_connected_edge_sequence(raw_path):
                    normalized_paths.append(path_nodes)
                    all_nodes.update(path_nodes)
                    all_edges.extend(
                        (path_nodes[index], path_nodes[index + 1])
                        for index in range(len(path_nodes) - 1)
                    )

            record = SubgraphRecord(
                nodes=all_nodes,
                edges=list(dict.fromkeys(all_edges)),
                paths=normalized_paths,
            )
        else:
            all_nodes = set(names)
            for hop in range(1, k + 1):
                neighbors = await ctx.graph.find_k_hop_neighbors_batch(
                    start_nodes=names,
                    k=hop,
                )
                all_nodes.update(neighbors or set())

            edge_set = set()
            for node in all_nodes:
                for edge in await ctx.graph.get_node_edges(node) or []:
                    if len(edge) < 2:
                        continue
                    src, tgt = str(edge[0]), str(edge[1])
                    if src in all_nodes and tgt in all_nodes:
                        edge_set.add(tuple(sorted((src, tgt))))

            record = SubgraphRecord(
                nodes=all_nodes,
                edges=sorted(edge_set),
            )

        return {
            "subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=record,
                producer="subgraph.khop_paths",
            )
        }
    except Exception as exc:
        logger.exception(f"subgraph_khop_paths failed: {exc}")
        return {
            "subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=SubgraphRecord(nodes=set(), edges=[]),
                producer="subgraph.khop_paths",
                metadata={"error": str(exc)},
            )
        }
