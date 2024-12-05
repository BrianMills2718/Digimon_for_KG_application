from abc import ABC, abstractmethod


class BaseCommunity(ABC):
    """Base community class definition."""

    def __init__(self, llm, enforce_sub_communities, namespace):
        self.llm = llm
        self.enforce_sub_communities = enforce_sub_communities
        self.namespace = namespace

    async def generate_community_report(self, graph, cluster_node_map, force):
        # Try to load the graph
        is_exist = await self._load_community_report()
        if force or not is_exist:
            # Build the graph based on the input chunks
            await self._generate_community_report(graph, cluster_node_map)
            # Persist the graph into file
            await self._persist_community()

    @abstractmethod
    async def cluster(self, **kwargs):
        pass

    @abstractmethod
    async def _generate_community_report(self, graph, cluster_node_map):
        pass

    @abstractmethod
    async def _load_community_report(self):
        pass

    @abstractmethod
    async def _persist_community(self):
        pass
