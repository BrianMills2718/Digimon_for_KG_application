"""Subgraph Steiner tree operator.

Compute minimum Steiner tree connecting seed entities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue, SubgraphRecord


async def subgraph_steiner_tree(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": SlotValue(ENTITY_SET)}
    Outputs: {"subgraph": SlotValue(SUBGRAPH)}
    """
    entities = inputs["entities"].data
    if not entities:
        return {"subgraph": SlotValue(kind=SlotKind.SUBGRAPH, data=SubgraphRecord(nodes=set(), edges=[]), producer="subgraph.steiner_tree")}

    names = [e.entity_name for e in entities]

    try:
        nx_subgraph = ctx.graph.get_induced_subgraph(nodes=names)
        if nx_subgraph is None:
            return {"subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=SubgraphRecord(nodes=set(names), edges=[]),
                producer="subgraph.steiner_tree",
            )}

        nodes = set(nx_subgraph.nodes())
        edges = list(nx_subgraph.edges())
        return {"subgraph": SlotValue(
            kind=SlotKind.SUBGRAPH,
            data=SubgraphRecord(nodes=nodes, edges=edges, nx_graph=nx_subgraph),
            producer="subgraph.steiner_tree",
        )}

    except Exception as e:
        logger.exception(f"subgraph_steiner_tree failed: {e}")
        return {"subgraph": SlotValue(
            kind=SlotKind.SUBGRAPH,
            data=SubgraphRecord(nodes=set(), edges=[]),
            producer="subgraph.steiner_tree",
        )}
