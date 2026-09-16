"""Subgraph Steiner-tree operator."""

from __future__ import annotations

from typing import Any, Dict, Optional

import networkx as nx
from networkx.algorithms.approximation.steinertree import steiner_tree

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue, SubgraphRecord


def _networkx_graph(graph):
    storage = getattr(graph, "_graph", graph)
    candidate = getattr(storage, "graph", storage)
    return candidate if isinstance(candidate, nx.Graph) else None


async def subgraph_steiner_tree(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": ENTITY_SET}
    Outputs: {"subgraph": SUBGRAPH}
    Params:  {"weight_attribute": str | None}
    """
    entities = inputs["entities"].data
    if not entities:
        return {
            "subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=SubgraphRecord(nodes=set(), edges=[]),
                producer="subgraph.steiner_tree",
            )
        }

    names = [record.entity_name for record in entities]

    try:
        graph = _networkx_graph(ctx.graph)
        if graph is None:
            raise TypeError("Steiner-tree operator requires a NetworkX-backed graph")

        terminals = [name for name in names if name in graph]
        if not terminals:
            result_graph = graph.subgraph([]).copy()
        elif len(terminals) == 1:
            result_graph = graph.subgraph(terminals).copy()
        else:
            # DIGIMON edge weights are relevance-like rather than guaranteed
            # path costs. Default to minimum-hop Steiner structure unless a
            # caller explicitly supplies a cost attribute.
            weight_attribute = (params or {}).get("weight_attribute") or "__unit_cost__"
            result_graph = steiner_tree(
                graph,
                terminal_nodes=terminals,
                weight=weight_attribute,
            )

        record = SubgraphRecord(
            nodes=set(result_graph.nodes()),
            edges=[(str(src), str(tgt)) for src, tgt in result_graph.edges()],
            nx_graph=result_graph,
        )
        return {
            "subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=record,
                producer="subgraph.steiner_tree",
            )
        }
    except Exception as exc:
        logger.exception(f"subgraph_steiner_tree failed: {exc}")
        return {
            "subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=SubgraphRecord(nodes=set(), edges=[]),
                producer="subgraph.steiner_tree",
            )
        }
