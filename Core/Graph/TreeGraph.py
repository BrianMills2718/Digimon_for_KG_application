from Core.Graph.BaseGraph import BaseGraph
from Core.Schema.ChunkSchema import TextChunk
from Core.Common.Logger import logger
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Prompt.RaptorPrompt import SUMMARIZE
from Core.Community.ClusterFactory import get_community
from Core.Storage.TreeGraphStorage import TreeGraphStorage
from Core.Schema.TreeSchema import TreeNode

from typing import List, Set, Any

Embedding = List[float]


class TreeGraph(BaseGraph):

    def __init__(self, config, llm, encoder):
        super().__init__(config, llm, encoder)
        self._graph: TreeGraphStorage = TreeGraphStorage()  # Tree index
        self.embedding_model = get_rag_embedding(config.embedding.api_type, config)  # Embedding model
        self.graph_cluster_algorithm = get_community("raptor")  # Clustering algorithm

    def _embed_text(self, text: str):
        return self.embedding_model._get_text_embedding(text)

    async def _create_node(self, layer: int, text: str, children_indices: Set[int] = None):
        embedding = self._embed_text(text)
        node_id = self._graph.num_nodes  # Give it an index
        logger.info(
            "Create node_id = {node_id}, children = {children}".format(node_id=node_id, children=children_indices))
        return self._graph.upsert_node(node_id=node_id,
                                       node_data={"layer": layer, "text": text, "children": children_indices,
                                                  "embedding": embedding})

    async def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]) -> TreeNode:
        # Build a leaf node from a text chunk
        chunk_key, chunk_info = chunk_key_pair
        leaf_node = await self._create_node(0, chunk_info.content)
        return leaf_node

    async def _extract_cluster_relationship(self, layer: int, cluster: List[TreeNode]) -> TreeNode:
        # Build a non-leaf node from a cluster of nodes
        summarized_text = await self._summarize_from_cluster(cluster, self.config.summarization_length)
        parent_node = await self._create_node(layer, summarized_text, {node.index for node in cluster})
        return parent_node

    async def _summarize_from_cluster(self, node_list: List[TreeNode], summarization_length=150) -> str:
        # Give a summarization from a cluster of nodes
        node_texts = f"\n\n".join([' '.join(node.text.splitlines()) for node in node_list])
        content = SUMMARIZE.format(context=node_texts)
        return await self.llm.aask(content, max_tokens=summarization_length)

    async def _build_tree_from_leaves(self):
        for layer in range(self.config.num_layers):  # build a new layer
            if len(self._graph.get_layer(layer)) <= self.config.reduction_dimension + 1:
                break

            self._graph.add_layer()

            clusters = self.graph_cluster_algorithm.clustering(
                nodes=self._graph.get_layer(layer),
                reduction_dimension=self.config.reduction_dimension,
                **self.config.graph_cluster_params,
            )

            for cluster in clusters:  # for each cluster, create a new node
                await self._extract_cluster_relationship(layer + 1, cluster)

            logger.info("Layer: {layer}".format(layer=layer))
            logger.info(self._graph.get_layer(layer + 1))

        logger.info(self._graph.num_layers)
        return

    async def _build_graph(self, chunks: List[Any]):
        self._graph = TreeGraphStorage()  # clear the storage before rebuilding

        self._graph.add_layer()

        for index, chunk in enumerate(chunks):  # for each chunk, create a leaf node
            await self._extract_entity_relationship(chunk_key_pair=chunk)

        logger.info(f"Created {len(self._graph.leaf_nodes)} Leaf Embeddings")
        await self._build_tree_from_leaves()
        return
