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

import numpy as np
import umap
import random
from sklearn.mixture import GaussianMixture

class TreeGraph(BaseGraph):

    def __init__(self, config, llm, encoder):
        super().__init__(config, llm, encoder)
        self._graph: TreeGraphStorage = TreeGraphStorage()  # Tree index
        self.embedding_model = get_rag_embedding(config.embedding.api_type, config)  # Embedding model
        random.seed(self.config.random_seed)

    def _GMM_cluster(self, embeddings: np.ndarray, threshold: float, random_state: int = 0):
        max_clusters = min(50, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        optimal_clusters = n_clusters[np.argmin(bics)]

        gm = GaussianMixture(n_components=optimal_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, optimal_clusters

    def _perform_clustering(
        self, embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
    ) -> List[np.ndarray]:
        reduced_embeddings_global = umap.UMAP(
            n_neighbors=int((len(embeddings) - 1) ** 0.5), n_components=min(dim, len(embeddings) -2), metric=self.config.cluster_metric
        ).fit_transform(embeddings)
        global_clusters, n_global_clusters = self._GMM_cluster(
            reduced_embeddings_global, threshold
        )

        if verbose:
            logger.info(f"Global Clusters: {n_global_clusters}")

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        for i in range(n_global_clusters):
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]
            if verbose:
                logger.info(
                    f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
                )
            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= dim + 1:
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                reduced_embeddings_local = umap.UMAP(
                    n_neighbors=10, n_components=dim, metric=self.config.cluster_metric
                ).fit_transform(embeddings)
                local_clusters, n_local_clusters = self.GMM_cluster(
                    reduced_embeddings_local, threshold
                )

            if verbose:
                logger.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        if verbose:
            logger.info(f"Total Clusters: {total_clusters}")
        return all_local_clusters


    def _clustering(self, nodes: List[TreeNode], max_length_in_cluster, tokenizer, reduction_dimension, threshold, verbose) -> List[List[TreeNode]]:
        # Get the embeddings from the nodes
        embeddings = np.array([node.embedding for node in nodes])

        # Perform the clustering
        clusters = self._perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate the total length of the text in the nodes
            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster:
                if verbose:
                    logger.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                node_clusters.extend(
                    self._clustering(
                        cluster_nodes, max_length_in_cluster, tokenizer, reduction_dimension, threshold, verbose
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters

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

            clusters = self._clustering(
                nodes = self._graph.get_layer(layer),
                max_length_in_cluster =  self.config.max_length_in_cluster,
                tokenizer = self.ENCODER,
                reduction_dimension = self.config.reduction_dimension,
                threshold = self.config.threshold,
                verbose = self.config.verbose,
            )

            for cluster in clusters:  # for each cluster, create a new node
                await self._extract_cluster_relationship(layer + 1, cluster)

            logger.info("Layer: {layer}".format(layer=layer))
            logger.info(self._graph.get_layer(layer + 1))

        logger.info(self._graph.num_layers)
        return

    async def _build_graph(self, chunks: List[Any]):
        self._graph.clear()  # clear the storage before rebuilding

        self._graph.add_layer()

        for index, chunk in enumerate(chunks):  # for each chunk, create a leaf node
            await self._extract_entity_relationship(chunk_key_pair=chunk)

        logger.info(f"Created {len(self._graph.leaf_nodes)} Leaf Embeddings")
        await self._build_tree_from_leaves()
        return
