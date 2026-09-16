"""Meta: PCST (Prize-Collecting Steiner Tree) optimization operator.

Optimize a subgraph by selecting informative nodes and edges from query-relevant
entity/relationship retrieval results.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue, SubgraphRecord


def _entity_prize(entity, prize_weight: float) -> float:
    """Return a non-negative prize without treating a real zero score as missing."""
    score = entity.score
    if score is None:
        score = 1.0
    return max(0.0, float(score)) * prize_weight


def _relationship_relevance(relationship) -> float:
    """Return higher-is-better relationship relevance for PCST edge costing."""
    if relationship.score is not None:
        return max(0.0, float(relationship.score))
    if relationship.weight is not None and relationship.weight > 0:
        return float(relationship.weight)
    return 1.0


async def meta_pcst_optimize(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": ENTITY_SET, "relationships": RELATIONSHIP_SET}
    Outputs: {"subgraph": SUBGRAPH}
    Params:  {"prize_weight": float}

    Entity VDB relevance becomes node prize. Relationship VDB relevance becomes
    inverse edge cost. Relationship-only endpoint nodes receive zero prize rather
    than an accidental default prize. If ``pcst_fast`` is unavailable, the
    fallback chooses the connected component with the greatest total prize.
    """
    entities = inputs["entities"].data
    rels = inputs["relationships"].data
    p = params or {}
    prize_weight = max(0.0, float(p.get("prize_weight", 1.0)))

    if not entities:
        return {
            "subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=SubgraphRecord(nodes=set(), edges=[]),
                producer="meta.pcst_optimize",
            )
        }

    names = {e.entity_name for e in entities}
    edges = [(r.src_id, r.tgt_id) for r in rels]

    try:
        import networkx as nx

        graph = nx.Graph()
        for entity in entities:
            graph.add_node(
                entity.entity_name,
                prize=_entity_prize(entity, prize_weight),
            )

        for relationship in rels:
            # NetworkX would otherwise auto-create unseen endpoints with no
            # attributes; make their zero-prize semantics explicit.
            for endpoint in (relationship.src_id, relationship.tgt_id):
                if endpoint not in graph:
                    graph.add_node(endpoint, prize=0.0)

            relevance = _relationship_relevance(relationship)
            graph.add_edge(
                relationship.src_id,
                relationship.tgt_id,
                relevance=relevance,
                graph_weight=float(relationship.weight or 0.0),
            )

        try:
            from pcst_fast import pcst_fast
            import numpy as np

            node_list = list(graph.nodes())
            node_idx = {node: i for i, node in enumerate(node_list)}
            prizes = np.array(
                [float(graph.nodes[node].get("prize", 0.0)) for node in node_list],
                dtype=float,
            )
            edge_list = list(graph.edges())
            edge_array = np.array(
                [[node_idx[src], node_idx[tgt]] for src, tgt in edge_list],
                dtype=int,
            )
            costs = np.array(
                [
                    1.0 / max(float(graph.edges[edge].get("relevance", 0.0)), 0.01)
                    for edge in edge_list
                ],
                dtype=float,
            )

            # pcst_fast handles a graph with no edges poorly; in that case the
            # highest-prize retrieved entity is the meaningful compact result.
            if not edge_list:
                best_node = max(node_list, key=lambda node: graph.nodes[node]["prize"])
                return {
                    "subgraph": SlotValue(
                        kind=SlotKind.SUBGRAPH,
                        data=SubgraphRecord(nodes={best_node}, edges=[]),
                        producer="meta.pcst_optimize",
                    )
                }

            selected_nodes, selected_edges = pcst_fast(
                edge_array,
                prizes,
                costs,
                -1,
                1,
                "strong",
                0,
            )

            opt_nodes = {node_list[i] for i in selected_nodes}
            opt_edges = [edge_list[i] for i in selected_edges]
            return {
                "subgraph": SlotValue(
                    kind=SlotKind.SUBGRAPH,
                    data=SubgraphRecord(nodes=opt_nodes, edges=opt_edges),
                    producer="meta.pcst_optimize",
                )
            }

        except ImportError:
            logger.info("pcst_fast not available; using prize-aware component fallback")
            if graph.number_of_nodes() == 1:
                subgraph = graph
            elif nx.is_connected(graph):
                subgraph = graph
            else:
                components = list(nx.connected_components(graph))
                best = max(
                    components,
                    key=lambda component: sum(
                        float(graph.nodes[node].get("prize", 0.0))
                        for node in component
                    ),
                )
                subgraph = graph.subgraph(best).copy()

            return {
                "subgraph": SlotValue(
                    kind=SlotKind.SUBGRAPH,
                    data=SubgraphRecord(
                        nodes=set(subgraph.nodes()),
                        edges=list(subgraph.edges()),
                        nx_graph=subgraph,
                    ),
                    producer="meta.pcst_optimize",
                )
            }

    except Exception as exc:
        logger.exception(f"meta_pcst_optimize failed: {exc}")
        return {
            "subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=SubgraphRecord(nodes=names, edges=edges),
                producer="meta.pcst_optimize",
                metadata={"error": str(exc)},
            )
        }
