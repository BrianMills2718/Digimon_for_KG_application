from abc import ABC, abstractmethod
import asyncio

from Core.Common.Logger import logger


class EntityResult(ABC):
    @abstractmethod
    def get_node_data(self):
        pass


class ColbertNodeResult(EntityResult):
    def __init__(self, node_idxs, ranks, scores):
        self.node_idxs = node_idxs
        self.ranks = ranks
        self.scores = scores

    async def get_node_data(self, graph, score=False):
        nodes = await asyncio.gather(
            *[graph.get_node_by_index(node_idx) for node_idx in self.node_idxs]
        )
        if score:
            return nodes, [r for r in self.scores]
        return nodes

    async def get_tree_node_data(self, graph, score=False):
        nodes = await asyncio.gather(
            *[graph.get_node(node_idx) for node_idx in self.node_idxs]
        )
        if score:
            return nodes, [r for r in self.scores]
        return nodes


class VectorIndexNodeResult(EntityResult):
    def __init__(self, results):
        self.results = results

    async def get_node_data(self, graph, score=False):
        metakey = getattr(graph, "entity_metakey", "entity_name")

        def _get_name(result):
            metadata = result.metadata
            return (
                metadata.get(metakey)
                or metadata.get("entity_name")
                or metadata.get("name")
                or metadata.get("id", "")
            )

        nodes = await asyncio.gather(
            *[graph.get_node(_get_name(result)) for result in self.results]
        )
        if score:
            return nodes, [result.score for result in self.results]
        return nodes

    async def get_tree_node_data(self, graph, score=False):
        processed_nodes = []
        processed_scores = []
        for result in self.results:
            node_id = result.metadata.get(graph.entity_metakey)
            layer = result.metadata.get("layer", -1)
            if node_id is None:
                logger.warning(
                    f"Node ID via metakey '{graph.entity_metakey}' missing from VDB metadata: "
                    f"{result.metadata}"
                )
                continue
            node_obj = await graph.get_node(node_id)
            if node_obj and hasattr(node_obj, "text"):
                node_data = {"id": node_id, "text": node_obj.text, "layer": layer}
                if score:
                    node_data["vdb_score"] = result.score
                    processed_scores.append(result.score)
                processed_nodes.append(node_data)
            else:
                logger.warning(f"Could not retrieve tree node '{node_id}'")

        if score:
            return processed_nodes, processed_scores
        return processed_nodes


class RelationResult(ABC):
    @abstractmethod
    def get_edge_data(self):
        pass


class VectorIndexEdgeResult(RelationResult):
    def __init__(self, results):
        self.results = results

    @staticmethod
    def _edge_endpoints(result):
        metadata = result.metadata
        src = metadata.get("src_id") or metadata.get("source")
        tgt = metadata.get("tgt_id") or metadata.get("target")
        if src is None or tgt is None:
            raise KeyError(
                "Relationship VDB result is missing src/tgt metadata "
                f"(available keys: {sorted(metadata.keys())})"
            )
        return src, tgt

    async def get_edge_data(self, graph, score=False):
        endpoints = [self._edge_endpoints(result) for result in self.results]
        edges = await asyncio.gather(
            *[graph.get_edge(src, tgt) for src, tgt in endpoints]
        )
        if score:
            return edges, [result.score for result in self.results]
        return edges


class SubgraphResult(ABC):
    @abstractmethod
    def get_subgraph_data(self):
        pass


class VectorIndexSubgraphResult(SubgraphResult):
    def __init__(self, results):
        self.results = results

    async def get_subgraph_data(self, score=False):
        subgraphs_data = [
            {
                "source_id": result.metadata["source_id"],
                "subgraph_content": result.text,
            }
            for result in self.results
        ]
        if score:
            return subgraphs_data, [result.score for result in self.results]
        return subgraphs_data


class ColbertEdgeResult(RelationResult):
    def __init__(self, edge_idxs, ranks, scores):
        self.edge_idxs = edge_idxs
        self.ranks = ranks
        self.scores = scores

    async def get_edge_data(self, graph, score=False):
        edges = await asyncio.gather(
            *[graph.get_edge_by_index(edge_idx) for edge_idx in self.edge_idxs]
        )
        if score:
            return edges, list(self.scores)
        return edges
