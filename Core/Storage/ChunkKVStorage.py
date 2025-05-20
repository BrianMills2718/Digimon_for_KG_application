import pickle
import os
from typing import Dict, List, Optional, Any, Union, Set # Changed Union to Any for simplicity with Pydantic fields
from pydantic import BaseModel, Field, ConfigDict # Import Pydantic components

from Core.Common.Utils import split_string_by_multi_markers 
from Core.Common.Constants import GRAPH_FIELD_SEP
import numpy as np
import numpy.typing as npt
from Core.Common.Logger import logger
from Core.Storage.BaseKVStorage import BaseKVStorage # This should be the corrected Pydantic version
from Core.Schema.ChunkSchema import TextChunk
# Namespace will be inherited from BaseKVStorage -> BaseStorage

class ChunkKVStorage(BaseKVStorage): # type: ignore # No longer a dataclass
    """
    Key-Value storage for TextChunks, allowing access by index or by a string key (chunk_id).
    Persists data to pickle files. Now a Pydantic BaseModel.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class variables for filenames
    DATA_IDX_PKL_NAME_CONST: str = "chunk_data_idx.pkl"
    CHUNK_KEY_PKL_NAME_CONST: str = "chunk_data_key.pkl"

    # Pydantic fields for instance data, initialized via default_factory or None
    internal_data_idx: Dict[int, TextChunk] = Field(default_factory=dict)
    internal_chunk_key: Dict[str, TextChunk] = Field(default_factory=dict)
    internal_key_to_index: Dict[str, int] = Field(default_factory=dict)
    # internal_np_keys: Optional[npt.NDArray[np.object_]] = None # If not actively used, can be omitted or kept as Optional

    def __init__(self, **data: Any):
        """
        Initializes ChunkKVStorage.
        'namespace' and 'config' are handled by the parent Pydantic BaseModel's __init__
        if they are defined as fields in BaseStorage.
        """
        super().__init__(**data)
        # Pydantic handles field initialization based on defaults or passed data.
        # If specific post-init logic is needed for these dicts, a @model_validator(mode='after') could be used.

    async def size(self) -> int:
        return len(self.internal_data_idx)

    async def get_by_key(self, key: str) -> Optional[TextChunk]:
        index = self.internal_key_to_index.get(key)
        if index is not None:
            return self.internal_data_idx.get(index)
        return None

    async def get_data_by_index(self, index: int) -> Optional[TextChunk]:
        return self.internal_data_idx.get(index)
        
    async def get_index_by_merge_key(self, merge_chunk_id: str) -> List[Optional[int]]:
        key_list = split_string_by_multi_markers(merge_chunk_id, [GRAPH_FIELD_SEP])
        index_list = [self.internal_key_to_index.get(chunk_id) for chunk_id in key_list]
        return index_list
    
    async def get_index_by_key(self, key: str) -> Optional[int]:
        return self.internal_key_to_index.get(key)

    async def upsert_batch(self, keys: List[str], values: List[TextChunk]) -> None:
        for key, value in zip(keys, values):
            await self.upsert(key, value)

    async def upsert(self, key: str, value: TextChunk) -> None:
        self.internal_chunk_key[key] = value
        index = self.internal_key_to_index.get(key)
        if index is None:
            if not hasattr(value, 'index') or not isinstance(value.index, int):
                logger.error(f"TextChunk for key '{key}' is missing a valid integer 'index' attribute. Skipping upsert.")
                # Remove potentially inconsistent entry from internal_chunk_key if index is bad
                if key in self.internal_chunk_key:
                    del self.internal_chunk_key[key]
                return 
            index = value.index 
            self.internal_key_to_index[key] = index
        self.internal_data_idx[index] = value

    async def delete_by_key(self, key: str) -> None:
        index = self.internal_key_to_index.pop(key, None)
        if index is not None:
            self.internal_data_idx.pop(index, None)
            self.internal_chunk_key.pop(key, None) 
        else:
            logger.warning(f"Key '{key}' not found in indexed key-value storage for deletion.")
    
    async def get_all_chunks_items(self) -> List[tuple[str, TextChunk]]:
        return list(self.internal_chunk_key.items())

    @property
    def data_idx_pkl_file_path(self) -> Optional[str]:
        if self.namespace:
            return self.namespace.get_save_path(self.DATA_IDX_PKL_NAME_CONST)
        return None

    @property
    def chunk_key_pkl_file_path(self) -> Optional[str]:
        if self.namespace:
            return self.namespace.get_save_path(self.CHUNK_KEY_PKL_NAME_CONST)
        return None
    
    async def load_chunk(self, force: bool = False) -> bool:
        if not force and (self.internal_data_idx or self.internal_chunk_key) :
            logger.info("Chunk data already in memory. Skipping load unless forced.")
            return True

        if force:
            logger.info("Forcing reload of chunk data.")
            self.internal_data_idx.clear()
            self.internal_chunk_key.clear()
            self.internal_key_to_index.clear()

        idx_file_path = self.data_idx_pkl_file_path
        key_file_path = self.chunk_key_pkl_file_path

        if not idx_file_path or not key_file_path:
            logger.warning("Namespace not set. Cannot load chunk data from files.")
            return False

        logger.info(f"Attempting to load chunk data from: {idx_file_path} and {key_file_path}")
        loaded_successfully = False
        if os.path.exists(idx_file_path) and os.path.exists(key_file_path):
            try:
                with open(idx_file_path, "rb") as file:
                    self.internal_data_idx = pickle.load(file)
                with open(key_file_path, "rb") as file:
                    self.internal_chunk_key = pickle.load(file)
                self.internal_key_to_index = {
                    key: value.index 
                    for key, value in self.internal_chunk_key.items() 
                    if hasattr(value, 'index') and isinstance(value.index, int)
                }
                logger.info(
                    f"Successfully loaded chunk data (idx and key) from: {idx_file_path} and {key_file_path}")
                loaded_successfully = True
            except Exception as e:
                logger.error(
                    f"Failed to load chunk data from: {idx_file_path} and {key_file_path} with {e}! Need to re-chunk the documents if this was not intended.")
                self.internal_data_idx.clear() 
                self.internal_chunk_key.clear()
                self.internal_key_to_index.clear()
        else:
            logger.info("Pickle file(s) do not exist! Need to chunk the documents from scratch.")
        return loaded_successfully

    async def _persist_internal(self):
        idx_file_path = self.data_idx_pkl_file_path
        key_file_path = self.chunk_key_pkl_file_path

        if not idx_file_path or not key_file_path:
            logger.warning("Namespace not set. Cannot persist chunk data to files.")
            return

        logger.info(f"Writing data into {idx_file_path} and {key_file_path}")
        try:
            # Ensure directory exists
            if idx_file_path: os.makedirs(os.path.dirname(idx_file_path), exist_ok=True)
            if key_file_path: os.makedirs(os.path.dirname(key_file_path), exist_ok=True)
            
            if idx_file_path: self._write_chunk_data_to_file(self.internal_data_idx, idx_file_path)
            if key_file_path: self._write_chunk_data_to_file(self.internal_chunk_key, key_file_path)
        except Exception as e:
            logger.error(f"Error persisting chunk data: {e}")

    @staticmethod
    def _write_chunk_data_to_file(data_to_write: Dict, pkl_file_path: str):
        with open(pkl_file_path, "wb") as file:
            pickle.dump(data_to_write, file)

    async def persist(self):
        await self._persist_internal()

    # This method might be redundant with get_all_chunks_items, consider consolidating
    async def get_chunks_legacy(self): 
        return list(self.internal_chunk_key.items())

    # --- Methods from BaseKVStorage that need to be implemented if not covered ---
    async def all_keys(self) -> list[str]:
        return list(self.internal_chunk_key.keys())

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[TextChunk, None]]:
        # This is a simplified implementation. 'fields' argument is not used here
        # as TextChunk is returned whole.
        if fields:
            logger.warning("'fields' argument is not utilized in this implementation of get_by_ids for ChunkKVStorage.")
        return [await self.get_by_key(id_val) for id_val in ids]

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        return {s for s in data if s not in self.internal_chunk_key}

    async def drop(self):
        self.internal_data_idx.clear()
        self.internal_chunk_key.clear()
        self.internal_key_to_index.clear()
        logger.info("ChunkKVStorage data dropped.")

