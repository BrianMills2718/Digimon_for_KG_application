"""
RAG Index Factory.
@Reference: https://github.com/geekan/MetaGPT/blob/main/metagpt/rag/factories/index.py
@Provide: BM25, FaissVectorStore, and MilvusVectorStore
"""
import faiss
import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
from Core.Index.ColBertStore import ColbertIndex
from pathlib import Path

from Core.Common.BaseFactory import ConfigBasedFactory
from Core.Index.Schema import (
    BaseIndexConfig,
    VectorIndexConfig,
    ColBertIndexConfig
)


class RAGIndexFactory(ConfigBasedFactory):
    def __init__(self):
        creators = {
            VectorIndexConfig: self._create_vector_index,
            ColBertIndexConfig: self._create_colbert
        }
        super().__init__(creators)

    def get_index(self, config: BaseIndexConfig, **kwargs) -> BaseIndex:
        """Key is IndexType."""
        return super().get_instance(config, **kwargs)

    def _create_vector_index(self, config: VectorIndexConfig, **kwargs) -> VectorStoreIndex:
        return VectorStoreIndex(
            nodes=[],
        )

    def _create_colbert(self, config: ColBertIndexConfig, **kwargs):
        index_path = (Path(config.persist_path) / config.index_name)
        if os.path.exists(index_path):
            return ColbertIndex.load_from_disk(config.persist_path, config.index_name)
        else:

            return ColbertIndex(**config.model_dump())

    def _index_from_storage(
            self, storage_context: StorageContext, config: BaseIndexConfig, **kwargs
    ) -> VectorStoreIndex:
        embed_model = self._extract_embed_model(config, **kwargs)

        return load_index_from_storage(storage_context=storage_context, embed_model=embed_model)

    def _index_from_vector_store(
            self, vector_store: BasePydanticVectorStore, config: BaseIndexConfig, **kwargs
    ) -> VectorStoreIndex:
        embed_model = self._extract_embed_model(config, **kwargs)

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

    def _extract_embed_model(self, config, **kwargs) -> BaseEmbedding:
        return self._val_from_config_or_kwargs("embed_model", config, **kwargs)


get_index = RAGIndexFactory().get_index
