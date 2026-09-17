from types import SimpleNamespace

import pytest

from Core.AgentSchema.tool_contracts import (
    CommunityDetectFromEntitiesInputs,
    CommunityGetLayerInputs,
)
from Core.AgentTools.community_tools import (
    community_detect_from_entities_tool,
    community_get_layer_tool,
)
from Core.Schema.CommunitySchema import LeidenInfo


class FakeGraph:
    async def community_schema(self):
        return {
            "10": LeidenInfo(
                community_id="10",
                level=10,
                nodes={"alpha"},
                occurrence=0.1,
            ),
            "2": LeidenInfo(
                community_id="2",
                level=2,
                nodes={"alpha", "beta"},
                occurrence=0.9,
            ),
            "1": LeidenInfo(
                community_id="1",
                level=1,
                nodes={"beta"},
                occurrence=0.5,
            ),
        }


class FakeContext:
    def get_graph_instance(self, graph_id):
        return FakeGraph()


@pytest.mark.asyncio
async def test_direct_community_detection_preserves_cluster_id():
    result = await community_detect_from_entities_tool(
        CommunityDetectFromEntitiesInputs(
            graph_reference_id="Demo_ERGraph",
            seed_entity_ids=["alpha"],
            max_communities_to_return=5,
        ),
        FakeContext(),
    )

    ids = [community.community_id for community in result.relevant_communities]
    assert ids == ["2", "10"]


@pytest.mark.asyncio
async def test_direct_community_layer_sort_is_numeric_not_lexical():
    result = await community_get_layer_tool(
        CommunityGetLayerInputs(
            community_hierarchy_reference_id="Demo_ERGraph",
            max_layer_depth=10,
        ),
        FakeContext(),
    )

    assert [community.community_id for community in result.communities_in_layers] == [
        "1",
        "2",
        "10",
    ]
