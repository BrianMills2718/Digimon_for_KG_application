"""Subgraph k-hop paths operator.

Find k-hop neighborhoods and paths from seed entities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue, SubgraphRecord


async def subgraph_khop_paths(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": SlotValue(ENTITY_SET)}
    Outputs: {"subgraph": SlotValue(SUBGRAPH)}
    Params:  {"k": int, "cutoff": int, "mode": "neighbors"|"paths"}
    """
    entities = inputs["entities"].data
    if not entities:
        return {"subgraph": SlotValue(kind=SlotKind.SUBGRAPH, data=SubgraphRecord(nodes=set(), edges=[]), producer="subgraph.khop_paths")}

    p = params or {}
    k = p.get("k", 2)
    cutoff = p.get("cutoff", 5)
    mode = p.get("mode", "neighbors")
    names = [e.entity_name for e in entities]

    try:
        if mode == "paths":
            paths = await ctx.graph.get_paths_from_sources(start_nodes=names, cutoff=cutoff)
            all_nodes = set()
            all_edges = []
            for path in (paths or []):
                for node in path:
                    all_nodes.add(node)
                for i in range(len(path) - 1):
                    all_edges.append((path[i], path[i + 1]))
            record = SubgraphRecord(nodes=all_nodes, edges=all_edges, paths=paths)
        else:
            neighbor_set = await ctx.graph.find_k_hop_neighbors_batch(start_nodes=names, k=k)
            record = SubgraphRecord(nodes=neighbor_set or set(), edges=[])

        return {"subgraph": SlotValue(kind=SlotKind.SUBGRAPH, data=record, producer="subgraph.khop_paths")}

    except Exception as e:
        logger.exception(f"subgraph_khop_paths failed: {e}")
        return {"subgraph": SlotValue(kind=SlotKind.SUBGRAPH, data=SubgraphRecord(nodes=set(), edges=[]), producer="subgraph.khop_paths")}
