"""FAISS-backed LlamaIndex adapter used by DIGIMON entity/relationship VDBs."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import faiss
import numpy as np
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.vector_stores.faiss import FaissVectorStore

from Core.Common.Logger import logger
from Core.Common.Utils import mdhash_id
from Core.Index.BaseIndex import BaseIndex
from Core.Schema.VdbResult import VectorIndexEdgeResult, VectorIndexNodeResult


class FaissIndex(BaseIndex):
    """FAISS adapter with normalized higher-is-better retrieval scores."""

    def __init__(self, config):
        super().__init__(config)
        self.embedding_model = config.embed_model

    def _metric_type(self):
        if self._index is None:
            return faiss.METRIC_L2
        storage_context = getattr(self._index, "storage_context", None)
        vector_store = getattr(storage_context, "vector_store", None)
        client = getattr(vector_store, "client", None)
        return getattr(client, "metric_type", faiss.METRIC_L2)

    @staticmethod
    def normalize_backend_score(raw_score, metric_type):
        """Convert backend scores to DIGIMON's higher-is-better convention."""
        if raw_score is None:
            return None
        score = float(raw_score)
        if metric_type == faiss.METRIC_L2:
            return 1.0 / (1.0 + max(score, 0.0))
        return score

    async def retrieval(self, query, top_k):
        if self._index is None:
            raise RuntimeError("FAISS index is not loaded or built")
        if top_k is None:
            top_k = self._get_retrieve_top_k()

        query_text = str(query)
        retriever = self._index.as_retriever(
            similarity_top_k=top_k,
            embed_model=self.config.embed_model,
        )
        query_embedding = await self._embed_text(query_text)
        query_bundle = QueryBundle(query_str=query_text, embedding=query_embedding)
        results = await retriever.aretrieve(query_bundle)
        metric_type = self._metric_type()
        return [
            NodeWithScore(
                node=result.node,
                score=self.normalize_backend_score(result.score, metric_type),
            )
            for result in results
        ]

    async def retrieval_nodes(
        self, query, top_k, graph, need_score=False, tree_node=False
    ):
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
        return await asyncio.gather(*[self.retrieval(query, top_k) for query in queries])

    @staticmethod
    def _query_text_from_seed(item: Any) -> str:
        """Resolve typed/dict/string entity seeds to the text that should be searched."""
        if item is None:
            return ""
        entity_name = getattr(item, "entity_name", None)
        if entity_name:
            return str(entity_name)
        if isinstance(item, dict):
            return str(item.get("entity_name") or item.get("name") or item.get("id") or "")
        return str(item)

    async def retrieval_nodes_with_score_matrix(self, query_list, top_k, graph):
        if graph is None:
            raise ValueError("graph is required for score-matrix retrieval")

        queries = [query_list] if isinstance(query_list, str) else list(query_list)
        score_vector = np.zeros(graph.node_num)
        aggregated_scores: dict[int, float] = {}

        for item in queries:
            target_query = self._query_text_from_seed(item)
            if not target_query:
                continue

            nodes, scores = await self.retrieval_nodes(
                query=target_query,
                top_k=top_k,
                graph=graph,
                need_score=True,
            )
            for node_data, score in zip(nodes or [], scores or []):
                if not node_data:
                    continue
                entity_name = node_data.get(graph.entity_metakey)
                if not entity_name:
                    continue
                node_idx = await graph.get_node_index(entity_name)
                if node_idx is None or not 0 <= node_idx < len(score_vector):
                    continue
                aggregated_scores[node_idx] = aggregated_scores.get(node_idx, 0.0) + float(
                    score or 0.0
                )

        for node_idx, score in aggregated_scores.items():
            score_vector[node_idx] = score

        total = float(np.sum(score_vector))
        if total > 0:
            score_vector /= total
        return score_vector

    def _get_retrieve_top_k(self):
        return getattr(self.config, "retrieve_top_k", 5) or 5

    async def _embed_text(self, text: str):
        model = self.embedding_model
        if hasattr(model, "aget_text_embedding"):
            return await model.aget_text_embedding(text)
        if hasattr(model, "get_text_embedding"):
            return model.get_text_embedding(text)
        if hasattr(model, "_get_text_embedding"):
            return model._get_text_embedding(text)
        raise TypeError(
            f"Embedding provider {type(model).__name__} does not expose a supported text embedding API"
        )

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self.config.embed_model
        if hasattr(model, "aget_text_embedding_batch"):
            return await model.aget_text_embedding_batch(texts, show_progress=False)
        if hasattr(model, "get_text_embedding_batch"):
            return model.get_text_embedding_batch(texts)
        if hasattr(model, "_get_text_embeddings"):
            batch_size = getattr(model, "embed_batch_size", 32) or 32
            embeddings = []
            for start in range(0, len(texts), batch_size):
                embeddings.extend(model._get_text_embeddings(texts[start : start + batch_size]))
            return embeddings
        return [await self._embed_text(text) for text in texts]

    @staticmethod
    def _configured_dimension(model) -> int | None:
        for attr in ("dimensions", "embed_dim"):
            value = getattr(model, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    @staticmethod
    def _node_from_data(data_item: dict[str, Any], embedding, meta_data_keys: list):
        metadata = {
            key: data_item[key]
            for key in meta_data_keys
            if key in data_item
        }
        node_id = data_item.get("index")
        if node_id is None:
            node_id = data_item.get("id")
        if node_id is None:
            node_id = mdhash_id(data_item["content"])
        return TextNode(
            id_=str(node_id),
            text=data_item["content"],
            embedding=embedding,
            metadata=metadata,
            excluded_embed_metadata_keys=list(metadata.keys()),
            excluded_llm_metadata_keys=list(metadata.keys()),
        )

    async def _update_index(self, datas: list[dict[str, Any]], meta_data_keys: list):
        if self.config.embed_model is None:
            logger.error("FAISS index cannot build without an embedding provider")
            return
        if not datas:
            logger.error("FAISS index cannot build from an empty data list")
            return

        Settings.embed_model = self.config.embed_model
        texts = [data["content"] for data in datas]
        try:
            text_embeddings = await self._embed_batch(texts)
        except Exception as exc:
            logger.error(f"FAISS embedding failed: {exc}", exc_info=True)
            return

        if len(text_embeddings) != len(datas) or not text_embeddings:
            logger.error(
                f"Embedding count mismatch: {len(text_embeddings)} embeddings for {len(datas)} items"
            )
            return

        inferred_dimension = len(text_embeddings[0]) if text_embeddings[0] else 0
        configured_dimension = self._configured_dimension(self.config.embed_model)
        embed_dims = configured_dimension or inferred_dimension
        if not embed_dims:
            logger.error("Could not infer FAISS embedding dimension from provider or returned vectors")
            return
        if configured_dimension and inferred_dimension and configured_dimension != inferred_dimension:
            logger.warning(
                f"Configured embedding dimension {configured_dimension} differs from returned vector "
                f"dimension {inferred_dimension}; using returned dimension"
            )
            embed_dims = inferred_dimension

        nodes = []
        for data_item, embedding in zip(datas, text_embeddings):
            if len(embedding) != embed_dims:
                logger.error(
                    f"Inconsistent embedding dimension: expected {embed_dims}, got {len(embedding)}"
                )
                self._index = None
                return
            nodes.append(self._node_from_data(data_item, embedding, meta_data_keys))

        try:
            faiss_index = faiss.IndexHNSWFlat(int(embed_dims), 32)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=self.config.embed_model,
            )
            self._index.insert_nodes(nodes)
            logger.info(
                f"Built FAISS index with {len(nodes)} vectors at dimension {embed_dims}"
            )
        except Exception as exc:
            logger.error(f"FAISS index construction failed: {exc}", exc_info=True)
            self._index = None

    async def _load_index(self) -> bool:
        try:
            if not os.path.exists(str(self.config.persist_path)):
                return False
            Settings.embed_model = self.config.embed_model
            vector_store = FaissVectorStore.from_persist_dir(str(self.config.persist_path))
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(self.config.persist_path),
            )
            self._index = load_index_from_storage(
                storage_context=storage_context,
                embed_model=self.config.embed_model,
            )
            return self._index is not None
        except Exception as exc:
            logger.error(f"FAISS index load failed: {exc}", exc_info=True)
            self._index = None
            return False

    async def upsert(self, data: dict[str, Any]):
        """Insert one item into the existing index without replacing prior vectors."""
        if self._index is None:
            raise RuntimeError("FAISS index is not loaded or built")
        if "content" not in data:
            raise ValueError("FAISS upsert requires a 'content' field")

        embeddings = await self._embed_batch([data["content"]])
        if len(embeddings) != 1 or not embeddings[0]:
            raise RuntimeError("FAISS upsert failed to generate an embedding")

        node = self._node_from_data(data, embeddings[0], list(data.keys()))
        self._index.insert_nodes([node])
        self._storage_index()
        if self._index is None:
            raise RuntimeError("FAISS upsert persistence failed")

    def exist_index(self):
        return os.path.exists(self.config.persist_path)

    def _storage_index(self):
        if self._index is None or not self.config.persist_path:
            logger.error("Cannot persist an unavailable FAISS index")
            return
        try:
            persist_dir = str(self.config.persist_path)
            os.makedirs(persist_dir, exist_ok=True)
            self._index.storage_context.persist(persist_dir=persist_dir)
        except Exception as exc:
            logger.error(f"FAISS index persistence failed: {exc}", exc_info=True)
            self._index = None

    async def _update_index_from_documents(self, docs):
        if self._index is None:
            raise RuntimeError("FAISS index is not loaded or built")
        refreshed = self._index.refresh_ref_docs(docs)
        logger.info(f"Refreshed {sum(bool(value) for value in refreshed)} FAISS documents")

    def _get_index(self):
        """Compatibility hook; BaseIndex.build_index no longer calls this method."""
        dimension = self._configured_dimension(self.config.embed_model)
        if not dimension:
            raise RuntimeError(
                "FAISS dimension is unknown before embeddings are generated; use build_index()"
            )
        Settings.embed_model = self.config.embed_model
        vector_store = FaissVectorStore(
            faiss_index=faiss.IndexHNSWFlat(int(dimension), 32)
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self.config.embed_model,
        )

    async def _similarity_score(self, object_q, object_d):
        return None
