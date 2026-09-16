from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from Core.AgentSchema.tool_contracts import EntityPPRInputs
from Core.AgentTools.entity_tools import (
    entity_ppr_tool,
    ppr_damping_from_teleport_alpha,
)
from Core.Operators.entity.ppr import entity_ppr
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


class FakeGraph:
    entity_metakey = "entity_name"

    def __init__(self, scores=None):
        self.node_num = 2
        self._graph = SimpleNamespace(graph=nx.Graph([("alpha", "beta")]))
        self.last_damping = None
        self.scores = np.array(scores if scores is not None else [0.8, 0.2])

    async def get_node_index(self, entity_id):
        return {"alpha": 0, "beta": 1}.get(entity_id)

    async def get_node_by_indices(self, indices):
        nodes = {
            0: {"entity_name": "alpha", "source_id": "chunk-a"},
            1: {"entity_name": "beta", "source_id": "chunk-b"},
        }
        return [nodes[int(index)] for index in indices]

    async def personalized_pagerank(self, reset_prob_chunk, damping):
        self.last_damping = damping
        assert np.asarray(reset_prob_chunk[0]).sum() > 0
        return self.scores.copy()


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


@pytest.mark.asyncio
async def test_typed_ppr_uses_standard_damping_and_returns_highest_score_first():
    graph = FakeGraph(scores=[0.2, 0.8])
    ctx = SimpleNamespace(
        graph=graph,
        entities_vdb=None,
        sparse_matrices={},
        config=SimpleNamespace(
            use_entity_similarity_for_ppr=False,
            node_specificity=False,
            top_k=2,
        ),
    )
    inputs = {
        "query": SlotValue(
            kind=SlotKind.QUERY_TEXT,
            data="How are alpha and beta connected?",
            producer="test",
        ),
        "entities": SlotValue(
            kind=SlotKind.ENTITY_SET,
            data=[EntityRecord(entity_name="alpha", source_id="chunk-a")],
            producer="test",
        ),
    }

    result = await entity_ppr(inputs=inputs, ctx=ctx, params={})

    assert graph.last_damping == pytest.approx(0.85)
    ranked = result["entities"].data
    assert [entity.entity_name for entity in ranked] == ["beta", "alpha"]
