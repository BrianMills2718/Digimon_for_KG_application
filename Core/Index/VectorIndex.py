from Core.Common.Utils import mdhash_id
from Core.Common.Logger import logger
import os
from typing import Any

from llama_index.core.schema import Document
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings
from Core.Index.BaseIndex import BaseIndex, VectorIndexNodeResult, VectorIndexEdgeResult
import asyncio
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import QueryBundle
import numpy as np


class VectorIndex(BaseIndex):
    """Simple LlamaIndex-backed vector index."""

    def __init__(self, config):
        super().__init__(config)

    async def retrieval(self, query, top_k):
        if top_k is None:
            top_k = self._get_retrieve_top_k()
        retriever = self._index.as_retriever(
            similarity_top_k=top_k,
            embed_model=self.config.embed_model,
        )
        query_bundle = QueryBundle(query_str=query)
        return await retriever.aretrieve(query_bundle)

    async def retrieval_nodes(self, query, top_k, graph, need_score=False, tree_node=False):
        results = await self.retrieval(query, top_k)
        result = VectorIndexNodeResult(results)
        if tree_node:
            return await result.get_tree_node_data(graph, need_score)
        return await result.get_node_data(graph, need_score)

    async def retrieval_edges(self, query, top_k, graph, need_score=False):
        results = await self.retrieval(query, top_k)
        result = VectorIndexEdgeResult(results)
        return await result.get_edge_data(graph, need_score)

    async def retrieval_batch(self, queries, top_k):
        pass

    async def _update_index(self, datas: list[dict[str, Any]], meta_data: list):
        # BaseIndex no longer calls _get_index() before updates. Keep provider
        # initialization here so this backend is self-contained.
        Settings.embed_model = self.config.embed_model

        async def process_document(data):
            return Document(
                doc_id=mdhash_id(data["content"]),
                text=data["content"],
                metadata={key: data[key] for key in meta_data if key in data},
                excluded_embed_metadata_keys=meta_data,
            )

        documents = await asyncio.gather(*[process_document(data) for data in datas])
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        self._index = VectorStoreIndex(nodes, embed_model=self.config.embed_model)
        logger.info("refresh index size is {}".format(len(nodes)))

    async def _load_index(self) -> bool:
        try:
            Settings.embed_model = self.config.embed_model
            storage_context = StorageContext.from_defaults(persist_dir=self.config.persist_path)
            self._index = load_index_from_storage(
                storage_context,
                embed_model=self.config.embed_model,
            )
            return True
        except Exception as e:
            logger.error("Loading index error: {}".format(e))
            return False

    async def upsert(self, data: dict[str, Any]):
        pass

    def exist_index(self):
        return os.path.exists(self.config.persist_path)

    def _get_retrieve_top_k(self):
        return self.config.retrieve_top_k

    def _storage_index(self):
        self._index.storage_context.persist(persist_dir=self.config.persist_path)

    async def _update_index_from_documents(self, docs: list[Document]):
        refreshed_docs = self._index.refresh_ref_docs(docs)
        logger.info("refresh index size is {}".format(len([True for doc in refreshed_docs if doc])))

    def _get_index(self):
        # Compatibility hook for older callers; BaseIndex.build_index no longer
        # relies on it because index shape/provider setup belongs to _update_index.
        Settings.embed_model = self.config.embed_model
        return VectorStoreIndex([], embed_model=self.config.embed_model)

    async def _similarity_score(self, object_q, object_d):
        pass

    async def retrieval_nodes_with_score_matrix(self, query_list, top_k, graph):
        if isinstance(query_list, str):
            query_list = [query_list]
        results = await asyncio.gather(
            *[
                self.retrieval_nodes(query, top_k, graph, need_score=True)
                for query in query_list
            ]
        )
        reset_prob_matrix = np.zeros((len(query_list), graph.node_num))
        entity_indices = []
        scores = []

        async def set_idx_score(_idx, res):
            for entity, score in zip(res[0], res[1]):
                entity_indices.append(await graph.get_node_index(entity["entity_name"]))
                scores.append(score)

        await asyncio.gather(*[set_idx_score(idx, res) for idx, res in enumerate(results)])
        reset_prob_matrix[np.arange(len(query_list)).reshape(-1, 1), entity_indices] = scores
        all_entity_weights = reset_prob_matrix.max(axis=0)
        if all_entity_weights.sum() > 0:
            all_entity_weights /= all_entity_weights.sum()
        return all_entity_weights
