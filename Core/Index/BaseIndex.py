import os
from abc import ABC, abstractmethod

from Core.Common.Logger import logger
from Core.Common.Utils import clean_storage
from Core.Schema.VdbResult import *


class BaseIndex(ABC):
    async def load(self) -> bool:
        """Load an existing index through the subclass implementation."""
        if not self.exist_index():
            logger.warning(
                f"Attempted to load index from {self.config.persist_path}, but it does not exist."
            )
            return False
        logger.info(f"Attempting to load existing index from: {self.config.persist_path}")
        try:
            loaded_successfully = await self._load_index()
            if loaded_successfully:
                logger.info(f"Successfully loaded existing index from: {self.config.persist_path}")
            else:
                logger.warning(f"Failed to load existing index from: {self.config.persist_path}")
            return bool(loaded_successfully)
        except Exception as e:
            logger.error(f"Exception during index load from {self.config.persist_path}: {e}")
            return False

    def __init__(self, config):
        self.config = config
        self._index = None

    async def build_index(self, elements, meta_data, force=False) -> bool:
        """Load or build the index and return whether it is actually usable.

        Subclasses own index initialization inside ``_update_index``. This is
        important for backends such as FAISS where the correct structure
        depends on the embedding dimension discovered at runtime.
        """
        logger.info(
            f"Starting build_index for VDB at {self.config.persist_path}. Force flag is: {force}"
        )

        should_load_existing = self.exist_index() and not force
        if should_load_existing:
            logger.info(f"Attempting to load existing index from: {self.config.persist_path}")
            if await self._load_index():
                logger.info(f"Successfully loaded existing index from: {self.config.persist_path}")
                return self._index is not None
            logger.warning(
                f"Failed to load existing index from: {self.config.persist_path}. "
                "Will proceed to build a new one."
            )

        if self.exist_index():
            logger.info(
                f"Deleting existing index at {self.config.persist_path} before rebuilding "
                f"(force={force}, load_failed={should_load_existing})."
            )
            await self.clean_index()

        # Do not call _get_index() here. Vector/FAISS/ColBERT implementations
        # initialize the concrete structure while processing the actual data.
        self._index = None
        logger.info(
            f"Building and persisting new index with {len(elements)} elements "
            f"using metadata keys: {meta_data}."
        )
        await self._update_index(elements, meta_data)
        if self._index is None:
            logger.error("Index update failed and left the index unavailable.")
            return False

        self._storage_index()
        if not self.exist_index():
            logger.error(
                f"Index persistence did not create {self.config.persist_path}; treating build as failed."
            )
            return False

        logger.info("✅ Finished VDB index setup process successfully.")
        return True

    def exist_index(self):
        return os.path.exists(self.config.persist_path)

    @abstractmethod
    async def retrieval(self, query, top_k):
        pass

    @abstractmethod
    def _get_index(self):
        """Legacy factory hook retained for compatibility; build_index no longer calls it."""
        pass

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

    async def similarity_score(self, object_q, object_d):
        return await self._similarity_score(object_q, object_d)

    async def _similarity_score(self, object_q, object_d):
        pass

    async def get_max_score(self, query):
        pass

    async def clean_index(self):
        clean_storage(self.config.persist_path)

    @abstractmethod
    async def retrieval_nodes(self, query, top_k, graph):
        pass

    async def retrieval_nodes_with_score_matrix(self, query_list, top_k, graph):
        pass
