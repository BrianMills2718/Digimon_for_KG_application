from Core.Common.Utils import mdhash_id
from Core.Common.Logger import logger
import os
from typing import Any
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    Document
)
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings
from Core.Index.BaseIndex import BaseIndex


class VectorIndex(BaseIndex):
    """VectorIndex is designed to be simple and straightforward.

    It is a lightweight and easy-to-use vector database for ANN search.
    """

    def __init__(self, config):
        super().__init__(config)

    async def retrieval(self, query, top_k):
        if top_k is None:
            top_k = self._get_retrieve_top_k()
        retriever: BaseRetriever = self._index.as_retriever(similarity_top_k=top_k, embed_model=self.config.embed_model)
        nodes = await retriever.aretrieve(query)
        return nodes

    async def retrieval_batch(self, queries, top_k):
        pass

    async def _update_index(self, datas: list[dict[str:Any]], meta_data: list):
        documents = [
            Document(
                doc_id=mdhash_id(data["content"]),
                text=data["content"],
                metadata={key: data[key] for key in meta_data},
                excluded_embed_metadata_keys=meta_data,
            )
            for data in datas
        ]
        await self._update_index_from_documents(documents)

    async def _load_index(self) -> bool:
        try:
            Settings.embed_model = self.config.embed_model

            storage_context = StorageContext.from_defaults(persist_dir=self.config.persist_path)
            self._index = load_index_from_storage(storage_context)
            return True
        except Exception as e:
            logger.error("Loading index error: {}".format(e))
            return False

    async def upsert(self, data: dict[str: Any]):
        pass

    def exist_index(self):
        return os.path.exists(self.config.persist_path)

    def _get_retrieve_top_k(self):
        return self.config.retrieve_top_k

    def _storage_index(self):
        self._index.storage_context.persist(persist_dir=self.config.persist_path)

    async def _update_index_from_documents(self, docs: list[Document]):
        refreshed_docs = self._index.refresh_ref_docs(docs)

        # the number of docs that are refreshed. if True in refreshed_docs, it means the doc is refreshed.
        logger.info("refresh index size is {}".format(len([True for doc in refreshed_docs if doc])))

    def _get_index(self):
        Settings.embed_model = self.config.embed_model
        return VectorStoreIndex([])
