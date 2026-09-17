import networkx as nx
import pytest

from Core.Community.LeidenCommunity import LeidenCommunity


class FakeCommunityMap:
    def __init__(self):
        self.data = None

    async def upsert(self, data):
        self.data = data


@pytest.mark.asyncio
async def test_singleton_graph_materializes_one_level_zero_community_without_leiden(monkeypatch):
    community = object.__new__(LeidenCommunity)
    community._community_node_map = FakeCommunityMap()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("hierarchical_leiden should not run for a singleton graph")

    monkeypatch.setattr(
        "Core.Community.LeidenCommunity.hierarchical_leiden",
        fail_if_called,
    )

    graph = nx.Graph()
    graph.add_node("Only Entity")

    result = await community._clustering(
        graph,
        max_cluster_size=10,
        random_seed=123,
    )

    expected = {
        "only entity": [{"level": 0, "cluster": "0"}],
    }
    assert result == expected
    assert community._community_node_map.data == expected
