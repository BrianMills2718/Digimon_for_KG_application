from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from Core.AgentSchema.tool_contracts import EntityPPRInputs
from Core.AgentTools.entity_tools import (
    entity_ppr_tool,
    ppr_damping_from_teleport_alpha,
)


class FakeGraph:
    def __init__(self):
        self.node_num = 2
        self._graph = SimpleNamespace(graph=nx.Graph([("alpha", "beta")]))
        self.last_damping = None

    async def get_node_index(self, entity_id):
        return {"alpha": 0, "beta": 1}.get(entity_id)

    async def personalized_pagerank(self, reset_prob_chunk, damping):
        self.last_damping = damping
        assert np.isclose(reset_prob_chunk[0].sum(), 1.0)
        return np.array([0.8, 0.2])


class FakeContext:
    def __init__(self, graph):
        self.graph = graph

    def get_graph_instance(self, graph_id):
        return self.graph


def test_teleport_alpha_converts_to_igraph_damping():
    assert ppr_damping_from_teleport_alpha(0.15) == pytest.approx(0.85)
    assert ppr_damping_from_teleport_alpha(None) == pytest.approx(0.85)
    with pytest.raises(ValueError):
        ppr_damping_from_teleport_alpha(1.1)


@pytest.mark.asyncio
async def test_entity_ppr_passes_link_following_damping_to_graph():
    graph = FakeGraph()
    result = await entity_ppr_tool(
        EntityPPRInputs(
            graph_reference_id="Demo_ERGraph",
            seed_entity_ids=["alpha"],
            personalization_weight_alpha=0.15,
            top_k_results=2,
        ),
        FakeContext(graph),
    )

    assert graph.last_damping == pytest.approx(0.85)
    assert result.ranked_entities[0][0] == "alpha"
