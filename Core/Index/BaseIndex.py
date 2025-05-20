import os
from abc import ABC, abstractmethod
from Core.Common.Utils import clean_storage
from Core.Common.Logger import logger
from Core.Schema.VdbResult import * 

class BaseIndex(ABC):
    def __init__(self, config):
        self.config = config
        self._index = None

    async def build_index(self, elements, meta_data, force=False):
        logger.info(f"Starting build_index for VDB at {self.config.persist_path}. Force flag is: {force}")

        should_load_existing = self.exist_index() and not force
        index_loaded_successfully = False

        if should_load_existing:
            logger.info(f"Attempting to load existing index from: {self.config.persist_path}")
            index_loaded_successfully = await self._load_index()
            if index_loaded_successfully:
                logger.info(f"Successfully loaded existing index from: {self.config.persist_path}")
            else:
                logger.warning(f"Failed to load existing index from: {self.config.persist_path}. Will proceed to build a new one.")
        
        if not index_loaded_successfully: # This covers: (not exist_index) OR (force=True) OR (load_failed)
            if self.exist_index(): 
                logger.info(f"Deleting existing index at {self.config.persist_path} before rebuilding (force={force}, load_failed={not index_loaded_successfully and should_load_existing}).")
                await self.clean_index() 

            logger.info(f"Initializing new index structure for VDB at {self.config.persist_path}.")
            self._index = self._get_index() 
            
            logger.info(f"Building and persisting new index with {len(elements)} elements using metadata keys: {meta_data}.") # Added metadata keys logging
            await self._update_index(elements, meta_data) 
            self._storage_index() 
            logger.info("New index successfully built and stored.")
        
        logger.info("✅ Finished VDB index setup process.")

    def exist_index(self):
        return os.path.exists(self.config.persist_path)

    @abstractmethod
    async def retrieval(self, query, top_k):
        pass

    @abstractmethod
    def _get_index(self):
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