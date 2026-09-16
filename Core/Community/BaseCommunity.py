from abc import ABC, abstractmethod

from Core.Common.Logger import logger


class BaseCommunity(ABC):
    """Base community class definition."""

    def __init__(self, llm, enforce_sub_communities, namespace):
        self.llm = llm
        self.enforce_sub_communities = enforce_sub_communities
        self.namespace = namespace

    async def generate_community_report(self, graph, force=False):
        """Load or generate community reports for ``graph``."""
        logger.info("Generating community report...")
        is_exist = await self._load_community_report(graph, force)
        if force or not is_exist:
            await self._generate_community_report(graph)
            await self._persist_community()
        logger.info("✅ [Community Report] Finished")

    async def cluster(self, **kwargs):
        """Load or generate the community-to-node clustering map."""
        logger.info("Starting community clustering")
        force = kwargs.pop("force", False)
        is_exist = await self._load_cluster_map(force)
        if force or not is_exist:
            await self.clustering(**kwargs)
            await self._persist_cluster_map()
        logger.info("✅ Community clustering finished")

    @abstractmethod
    async def _generate_community_report(self, graph):
        pass

    @abstractmethod
    async def clustering(self, **kwargs):
        pass

    @abstractmethod
    async def _load_community_report(self, graph, force):
        pass

    @abstractmethod
    async def _persist_community(self):
        pass

    @abstractmethod
    async def _load_cluster_map(self, force):
        pass

    @abstractmethod
    async def _persist_cluster_map(self):
        pass
