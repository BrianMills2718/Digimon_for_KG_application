import networkx as nx
import pytest

from Core.AgentSchema.tool_contracts import SubgraphSteinerTreeInputs
from Core.AgentTools.subgraph_tools import subgraph_steiner_tree_tool


class GraphWrapper:
    def __init__(self, graph):
        self._graph = type("Storage", (), {"graph": graph})()


class Context:
    def __init__(self, graph):
        self.graph = GraphWrapper(graph)

    def get_graph_instance(self, graph_id):
        return self.graph


@pytest.mark.asyncio
async def test_direct_steiner_does_not_anchor_on_disconnected_first_terminal():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("isolated", "helper"),
            ("alpha", "middle"),
            ("middle", "beta"),
        ]
    )

    result = await subgraph_steiner_tree_tool(
        SubgraphSteinerTreeInputs(
            graph_reference_id="Demo_ERGraph",
            terminal_node_ids=["isolated", "alpha", "beta"],
        ),
        Context(graph),
    )

    edges = {
        frozenset((edge["source"], edge["target"]))
        for edge in result.steiner_tree_edges
    }
    assert edges == {
        frozenset(("alpha", "middle")),
        frozenset(("middle", "beta")),
    }


@pytest.mark.asyncio
async def test_direct_steiner_does_not_treat_relevance_weight_as_default_cost():
    graph = nx.Graph()
    graph.add_edge("alpha", "beta", weight=10.0)
    graph.add_edge("alpha", "detour", weight=1.0)
    graph.add_edge("detour", "beta", weight=1.0)

    result = await subgraph_steiner_tree_tool(
        SubgraphSteinerTreeInputs(
            graph_reference_id="Demo_ERGraph",
            terminal_node_ids=["alpha", "beta"],
            edge_weight_attribute=None,
        ),
        Context(graph),
    )

    edges = {
        frozenset((edge["source"], edge["target"]))
        for edge in result.steiner_tree_edges
    }
    assert edges == {frozenset(("alpha", "beta"))}
