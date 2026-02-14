"""Meta: PCST (Prize-Collecting Steiner Tree) optimization operator.

Optimize a subgraph by selecting the most informative nodes and edges
based on entity/relationship scores using PCST.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue, SubgraphRecord


async def meta_pcst_optimize(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": SlotValue(ENTITY_SET), "relationships": SlotValue(RELATIONSHIP_SET)}
    Outputs: {"subgraph": SlotValue(SUBGRAPH)}
    Params:  {"prize_weight": float}

    Falls back to induced subgraph if pcst_fast is not available.
    """
    entities = inputs["entities"].data
    rels = inputs["relationships"].data
    p = params or {}

    if not entities:
        return {"subgraph": SlotValue(
            kind=SlotKind.SUBGRAPH,
            data=SubgraphRecord(nodes=set(), edges=[]),
            producer="meta.pcst_optimize",
        )}

    names = set(e.entity_name for e in entities)
    edges = [(r.src_id, r.tgt_id) for r in rels]

    try:
        import networkx as nx

        G = nx.Graph()
        for e in entities:
            G.add_node(e.entity_name, prize=e.score or 1.0)
        for r in rels:
            G.add_edge(r.src_id, r.tgt_id, weight=r.weight or 1.0)

        # If pcst_fast available, use it; otherwise just return induced subgraph
        try:
            from pcst_fast import pcst_fast
            import numpy as np

            node_list = list(G.nodes())
            node_idx = {n: i for i, n in enumerate(node_list)}
            prizes = np.array([G.nodes[n].get("prize", 1.0) for n in node_list])
            edge_list = list(G.edges())
            costs = np.array([1.0 / max(G.edges[e].get("weight", 1.0), 0.01) for e in edge_list])
            edge_array = np.array([[node_idx[e[0]], node_idx[e[1]]] for e in edge_list])

            selected_nodes, selected_edges = pcst_fast(
                edge_array, prizes, costs,
                -1, 1, "strong", 0,
            )

            opt_nodes = set(node_list[i] for i in selected_nodes)
            opt_edges = [edge_list[i] for i in selected_edges]

            return {"subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=SubgraphRecord(nodes=opt_nodes, edges=opt_edges),
                producer="meta.pcst_optimize",
            )}

        except ImportError:
            logger.info("pcst_fast not available, using induced subgraph fallback")
            # Take connected component containing highest-scored entity
            if nx.is_connected(G):
                subG = G
            else:
                components = list(nx.connected_components(G))
                best = max(components, key=len)
                subG = G.subgraph(best)

            return {"subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=SubgraphRecord(
                    nodes=set(subG.nodes()),
                    edges=list(subG.edges()),
                    nx_graph=subG,
                ),
                producer="meta.pcst_optimize",
            )}

    except Exception as e:
        logger.exception(f"meta_pcst_optimize failed: {e}")
        return {"subgraph": SlotValue(
            kind=SlotKind.SUBGRAPH,
            data=SubgraphRecord(nodes=names, edges=edges),
            producer="meta.pcst_optimize",
        )}
