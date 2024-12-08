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
            *[(graph.get_node_by_index(node_idx), self.scores[idx]) for idx, node_idx in enumerate(self.node_idxs)]
        )
    
class VectorIndexNodeResult(EntityResult):
    def __init__(self, results):
        self.results = results

    async def get_node_data(self, graph, score = False):

        nodes = await asyncio.gather( *[ graph.get_node(r.metadata["entity_name"]) for r in self.results])
        if score:

            return nodes, [r.score for r in self.results]
        else:
            return nodes