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
                path_nodes = []
                for raw_edge in raw_path:
                    endpoints = _edge_record_endpoints(raw_edge)
                    if endpoints is None:
                        continue
                    src, tgt = endpoints
                    all_nodes.update((src, tgt))
                    all_edges.append((src, tgt))
                    if not path_nodes:
                        path_nodes.extend((src, tgt))
                    elif path_nodes[-1] == src:
                        path_nodes.append(tgt)
                    elif path_nodes[-1] == tgt:
                        path_nodes.append(src)
                    else:
                        # The storage can return concatenated path segments.
                        # Preserve both endpoints rather than fabricating adjacency.
                        path_nodes.extend((src, tgt))
                if path_nodes:
                    normalized_paths.append(path_nodes)

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

            # Preserve actual graph edges inside the discovered neighborhood.
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
            )
        }
