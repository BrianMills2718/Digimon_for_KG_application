from Core.Common.Logger import logger
import os
from typing import Any, List
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    Document
)
from llama_index.core import StorageContext, load_index_from_storage
from Core.Common.Utils import mdhash_id
from Core.Index.BaseIndex import BaseIndex


class VectorIndex(BaseIndex):
    """VectorIndex is designed to be simple and straightforward.

    It is a lightweight and easy-to-use vector database for ANN search.
    """

    async def retrieval_batch(self, queries, top_k):
        pass

    async def _update_index(self, datas, meta_data):

        pass

    async def _load_index(self) -> bool:
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.config.persist_path)
            self._index = load_index_from_storage(storage_context)
            return True
        except Exception as e:
            logger.error("Loading index error: {}".format(e))
            return False

    async def upsert(self, data: dict[str: Any]):

        documents = [Document(text=data["content"], key=mdhash_id(t)) for t in data]
        await self._update_index_from_documents(documents)

    else:
    logger.warning("The type of data is not supported")


async def upsert_with_embedding(self, text: str, embedding: List[float], metadata: dict):
    await self._update_index_from_documents([Document(text=text, embedding=embedding, metadata=metadata)])


def exist_index(self):
    return os.path.exists(self.config.persist_path)


async def retrieval(self, query, top_k=None):
    retriever: BaseRetriever = self._index.as_retriever(similarity_top_k=top_k)
    nodes = await retriever.aretrieve(query)
    return nodes


def _get_retrieve_top_k(self):
    return self.config.retrieve_top_k


def _storage_index(self):
    self._index.storage_context.persist(persist_dir=self.config.persist_path)
