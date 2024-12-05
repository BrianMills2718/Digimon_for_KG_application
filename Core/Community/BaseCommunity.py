from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from Core.Provider.BaseLLM import BaseLLM
from typing import Optional


class BaseCommunity(ABC):
    """Base community class definition."""

    def __init__(self, llm, enforce_sub_communities):
        self.llm = llm
        self.enforce_sub_communities = enforce_sub_communities

    @abstractmethod
    async def generate_community_report(self, graph, cluster_node_map):
        pass


    @abstractmethod
    async def cluster(self, **kwargs):
        pass
