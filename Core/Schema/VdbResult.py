from abc import ABC, abstractmethod
import asyncio
class EntityResult(ABC):
    @abstractmethod
    def get_node_data(self):
        pass


class ColbertNodeResult(EntityResult):
    def __init__(self, node_idxs, ranks, scores):
        self.node_idxs = node_idxs
        self.ranks = ranks
        self.scores = scores

    async def get_node_data(self, graph):
        return await asyncio.gather(
            *[graph.get_node_by_index(node_idx) for node_idx in self.node_idxs]
        )
    
class VectorIndexNodeResult(EntityResult):
    def __init__(self, results):
        self.results = results

    async def get_node_data(self, graph):
        return await asyncio.gather(
            *[graph.get_node(r.metadata["entity_name"]) for r in self.results]
        )