import os
from abc import ABC, abstractmethod

from llama_index.core import StorageContext, load_index_from_storage, Settings, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from Core.Common.Logger import logger
from Core.Index import get_index


class BaseIndex(ABC):
    def __init__(self, config):
        self.config = config
        self._index = None

    async def build_index(self, elements, meta_data, force):
        Settings.embed_model = self.config.embed_model
        from_load = False
        if self.exist_index() or not force:
            logger.info("Loading index from the file {}".format(self.config.persist_path))
            from_load = await self._load_index()
        else:
            self._index = get_index(self.config)
        if not from_load:
            # Note: When you successfully load the index from a file, you don't need to rebuild it.
            logger.info("Building index for input elements")
            await self._update_index(elements, meta_data)
            self._storage_index()
            logger.info("Index successfully built and stored.")

    def exist_index(self):
        return os.path.exists(self.config.persist_path)

    @abstractmethod
    async def retrieval(self, query, top_k):
        retriever: BaseRetriever = self.index.as_retriever(similarity_top_k=self._get_retrieve_top_k(),
                                                           embed_model=self.config.embed_model)
        nodes = await retriever.aretrieve(query)
        return nodes

    @abstractmethod
    async def retrieval_batch(self, queries, top_k):
        pass

    @abstractmethod
    async def _update_index(self, elements, meta_data):
        pass

    @abstractmethod
    def _get_retrieve_top_k(self):
        return 10

    @abstractmethod
    def _storage_index(self):
        pass

    @abstractmethod
    async def _load_index(self) -> bool:
        pass
