"""
Here is the colbert index for our GraphRAG
"""
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig
from pathlib import Path

from Core.Common.Logger import logger
import os
from typing import Any, List
from colbert.data import Queries
from Core.Index.BaseIndex import BaseIndex


class ColBertIndex(BaseIndex):
    """VectorIndex is designed to be simple and straightforward.

    It is a lightweight and easy-to-use vector database for ANN search.
    """

    async def _update_index(self, elements, meta_data):

        with Run().context(
                RunConfig(index_root=self.config.index_path, nranks=self.config.ranks)
        ):
            indexer = Indexer(checkpoint=self.config.model_name, config=self.config)
            # Store the index
            indexer.index(name=self.config.index_name, collection=elements, overwrite=True)
            self._index = Searcher(
                index=self.config.index_name, collection=elements, checkpoint=self.config.model_name
            )

    async def _load_index(self):
        colbert_config = ColBERTConfig.load_from_index(Path(self.config.persist_path) / self.config.index_name)
        searcher = Searcher(
            index=self.config.index_name, index_root=self.config.persist_path, config=colbert_config
        )
        return searcher

    async def upsert(self, data: list[Any]):
        pass

    def exist_index(self):

        return os.path.exists(self.config.persist_path)

    async def retrieval(self, query, top_k=None):
        if top_k is None:
            top_k = self._get_retrieve_top_k()
        return await self._index.search(query, k=top_k)

    async def retrieval_batch(self, queries, top_k=None):
        if top_k is None:
            top_k = self._get_retrieve_top_k()
            try:
                if not isinstance(queries, Queries):
                    queries = Queries(data=queries)

                return self._index.search_all(queries, k=top_k)
            except Exception as e:
                logger.exception(f"fail to search {queries} for {e}")
                return []

    def _get_retrieve_top_k(self):
        return self.config.retrieve_top_k

    def _storage_index(self):
        # Stores the index for Colbert-index upon its creation.
        pass
