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


def _best_connected_terminal_group(graph: nx.Graph, entities) -> tuple[list[str], list[str]]:
    """Choose the connected terminal group with the strongest query evidence.

    A Steiner tree cannot connect terminals from different connected components.
    Instead of failing the whole method, choose the component containing the
    most selected terminals, breaking ties by summed entity relevance score.
    Returns ``(kept, dropped)`` terminal names.
    """
    score_by_name = {
        record.entity_name: float(record.score or 0.0)
        for record in entities
    }
    terminals = [record.entity_name for record in entities if record.entity_name in graph]
    if len(terminals) <= 1:
        return terminals, []

    undirected = graph.to_undirected() if graph.is_directed() else graph
    component_by_node = {}
    for component_index, component in enumerate(nx.connected_components(undirected)):
        for node in component:
            component_by_node[node] = component_index

    groups: dict[int, list[str]] = {}
    for terminal in terminals:
        groups.setdefault(component_by_node[terminal], []).append(terminal)

    if len(groups) <= 1:
        return terminals, []

    kept = max(
        groups.values(),
        key=lambda group: (
            len(group),
            sum(score_by_name.get(name, 0.0) for name in group),
        ),
    )
    kept_set = set(kept)
    dropped = [terminal for terminal in terminals if terminal not in kept_set]
    return kept, dropped


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

    try:
        graph = _networkx_graph(ctx.graph)
        if graph is None:
            raise TypeError("Steiner-tree operator requires a NetworkX-backed graph")

        terminals, dropped_terminals = _best_connected_terminal_group(graph, entities)
        if not terminals:
            result_graph = graph.subgraph([]).copy()
        elif len(terminals) == 1:
            result_graph = graph.subgraph(terminals).copy()
        else:
            # DIGIMON edge weights are relevance-like rather than guaranteed
            # path costs. Default to minimum-hop Steiner structure unless a
            # caller explicitly supplies a true cost attribute.
            weight_attribute = (params or {}).get("weight_attribute") or "__unit_cost__"
            working_graph = graph.to_undirected() if graph.is_directed() else graph
            component_nodes = nx.node_connected_component(working_graph, terminals[0])
            connected_graph = working_graph.subgraph(component_nodes).copy()
            result_graph = steiner_tree(
                connected_graph,
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
                metadata={
                    "requested_terminals": [record.entity_name for record in entities],
                    "used_terminals": terminals,
                    "dropped_disconnected_terminals": dropped_terminals,
                },
            )
        }
    except Exception as exc:
        logger.exception(f"subgraph_steiner_tree failed: {exc}")
        return {
            "subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=SubgraphRecord(nodes=set(), edges=[]),
                producer="subgraph.steiner_tree",
                metadata={"error": str(exc)},
            )
        }
