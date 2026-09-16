import pytest

from Core.Community.BaseCommunity import BaseCommunity


class FakeCommunity(BaseCommunity):
    def __init__(self):
        super().__init__(llm=None, enforce_sub_communities=False, namespace=None)
        self.clustered = 0
        self.cluster_persisted = 0
        self.report_generated = 0
        self.report_persisted = 0

    async def clustering(self, **kwargs):
        assert kwargs["marker"] == "value"
        self.clustered += 1

    async def _load_cluster_map(self, force):
        return False

    async def _persist_cluster_map(self):
        self.cluster_persisted += 1

    async def _load_community_report(self, graph, force):
        assert graph == "graph"
        return False

    async def _generate_community_report(self, graph):
        assert graph == "graph"
        self.report_generated += 1

    async def _persist_community(self):
        self.report_persisted += 1


@pytest.mark.asyncio
async def test_community_cluster_and_report_lifecycle_runs_without_custom_logger_methods():
    community = FakeCommunity()

    await community.cluster(force=False, marker="value")
    await community.generate_community_report("graph", force=False)

    assert community.clustered == 1
    assert community.cluster_persisted == 1
    assert community.report_generated == 1
    assert community.report_persisted == 1
