import asyncio
from Core.Chunk.ChunkFactory import create_chunk_method
from Core.Common.Utils import mdhash_id
from Core.Common.Logger import logger
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.ChunkKVStorage import ChunkKVStorage
from typing import List, Union, Any # Added Any for docs type hint clarity

# Assuming self.config.chunk is ChunkConfig from Config.ChunkConfig
# Assuming self.ENCODER is a tiktoken encoder instance
# Assuming self.workspace.make_for("chunk_storage") returns a Namespace instance

class DocChunk:
    def __init__(self, config_chunk, token_model_encoder, namespace_obj):
        self.config = config_chunk # e.g., instance of ChunkConfig
        self.chunk_method = create_chunk_method(self.config.chunk_method)
        # Pass namespace to ChunkKVStorage constructor
        self._chunk = ChunkKVStorage(namespace=namespace_obj) 
        self.token_model = token_model_encoder

    # namespace property was here, but it's better if ChunkKVStorage handles its own namespace directly
    # if it's passed during its __init__

    async def build_chunks(self, docs: Union[str, List[Any]], force=True): # Added Any to List
        logger.info("Starting chunk the given documents")
  
        is_exist = await self._load_chunk(force)
        if not is_exist or force:
            processed_docs_dict = {}
            if isinstance(docs, str):
                processed_docs_dict = {mdhash_id(docs.strip(), prefix="doc-"): {"content": docs.strip(), "title": ""}}
            elif isinstance(docs, list):
                if all(isinstance(doc, dict) and "content" in doc for doc in docs):
                    processed_docs_dict = {
                        # Use doc_id from input if available, else generate
                        doc.get("doc_id", mdhash_id(doc["content"].strip(), prefix="doc-")): {
                            "content": doc["content"].strip(),
                            "title": doc.get("title", ""),
                        }
                        for doc in docs
                    }
                elif all(isinstance(doc, str) for doc in docs): # List of strings
                    processed_docs_dict = {
                        mdhash_id(doc.strip(), prefix="doc-"): {
                            "content": doc.strip(),
                            "title": "",
                        }
                        for doc in docs
                    }
                else:
                    logger.error("Unsupported format for 'docs' list. Expected list of dicts with 'content' or list of strings.")
                    return


            flatten_list = list(processed_docs_dict.items())
            doc_contents = [doc_item[1]["content"] for doc_item in flatten_list]
            doc_keys = [doc_item[0] for doc_item in flatten_list]
            title_list = [doc_item[1]["title"] for doc_item in flatten_list]
            
            # Ensure token_model is not None and has encode_batch
            if not hasattr(self.token_model, 'encode_batch'):
                logger.error("Token model does not have 'encode_batch' method.")
                return
                
            tokens = self.token_model.encode_batch(doc_contents, num_threads=16) # type: ignore

            # Ensure chunk_method is callable
            if not callable(self.chunk_method):
                logger.error(f"Chunk method '{self.config.chunk_method}' is not callable.")
                return

            chunks_data = await self.chunk_method(
                tokens,
                doc_keys=doc_keys,
                tiktoken_model=self.token_model,
                title_list=title_list,
                overlap_token_size=self.config.chunk_overlap_token_size,
                max_token_size=self.config.chunk_token_size,
            )

            for chunk_dict in chunks_data:
                # Ensure 'content' key exists for mdhash_id
                if "content" not in chunk_dict:
                    logger.warning(f"Chunk data missing 'content' key: {chunk_dict}. Skipping.")
                    continue
                chunk_dict["chunk_id"] = mdhash_id(chunk_dict["content"], prefix="chunk-")
                # Ensure all required fields for TextChunk are present in chunk_dict
                # TextChunk(tokens, chunk_id, content, doc_id, index, title)
                required_fields = ["tokens", "chunk_id", "content", "doc_id", "index"]
                if not all(field in chunk_dict for field in required_fields):
                    logger.warning(f"Chunk data missing required fields for TextChunk: {chunk_dict}. Required: {required_fields}. Skipping.")
                    continue
                
                # Add title if not present, defaulting to None or empty string as TextChunk expects
                chunk_dict.setdefault("title", None) 

                await self._chunk.upsert(chunk_dict["chunk_id"], TextChunk(**chunk_dict))

            await self._chunk.persist()
        logger.info("✅ Finished the chunking stage")

    async def _load_chunk(self, force=False):
        if force:
            return False
        return await self._chunk.load_chunk(force=force) # Pass force here too

    async def get_chunks(self):
        """Returns all chunks as a list of (key, TextChunk) items."""
        # Corrected to call the appropriate method on ChunkKVStorage
        return await self._chunk.get_all_chunks_items()

    @property
    async def size(self):
        return await self._chunk.size()

    async def get_index_by_merge_key(self, chunk_id):
        return await self._chunk.get_index_by_merge_key(chunk_id)

    async def get_index_by_key(self, key):
        return await self._chunk.get_index_by_key(key)

    async def get_data_by_key(self, chunk_id):
        chunk = await self._chunk.get_by_key(chunk_id)
        return chunk.content if chunk else None

    async def get_data_by_index(self, index):
        chunk = await self._chunk.get_data_by_index(index)
        return chunk.content if chunk else None

    # get_key_by_index might not be directly available in ChunkKVStorage,
    # it depends on how ChunkKVStorage is implemented.
    # If needed, it would require iterating or a reverse map in ChunkKVStorage.
    # For now, commenting out as it's not used in the current flow and might be complex.
    # async def get_key_by_index(self, index):
    #     return await self._chunk.get_key_by_index(index) 

    async def get_data_by_indices(self, indices: List[int]): # Added type hint
        return await asyncio.gather(
            *[self.get_data_by_index(index) for index in indices]
        )
