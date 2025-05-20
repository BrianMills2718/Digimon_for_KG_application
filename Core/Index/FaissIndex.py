from Core.Common.Utils import mdhash_id
from Core.Common.Logger import logger
import os
import faiss
from typing import Any
from llama_index.core.schema import (
    Document,
    TextNode
)
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings
from Core.Index.BaseIndex import BaseIndex, VectorIndexNodeResult, VectorIndexEdgeResult
import asyncio
from llama_index.core.schema import QueryBundle
import numpy as np
from llama_index.vector_stores.faiss import FaissVectorStore
from concurrent.futures import ProcessPoolExecutor
from llama_index.embeddings.openai import OpenAIEmbedding

class FaissIndex(BaseIndex):
    """FaissIndex is designed to be simple and straightforward.

    It is a lightweight and easy-to-use vector database for ANN search.
    """

    def __init__(self, config):
        super().__init__(config)
        self.embedding_model =config.embed_model
    async def retrieval(self, query, top_k):
        if top_k is None:
            top_k = self._get_retrieve_top_k()
        retriever = self._index.as_retriever(similarity_top_k=top_k, embed_model=self.config.embed_model)
        query_emb = self._embed_text(query)
        query_bundle = QueryBundle(query_str=query, embedding=query_emb)
    
        return await retriever.aretrieve(query_bundle)

    async def retrieval_nodes(self, query, top_k, graph, need_score=False, tree_node=False):
        results = await self.retrieval(query, top_k)
        result = VectorIndexNodeResult(results)
        if tree_node:
            return await result.get_tree_node_data(graph, need_score)
        else:
            return await result.get_node_data(graph, need_score)

    async def retrieval_edges(self, query, top_k, graph, need_score=False):

        results = await self.retrieval(query, top_k)
        result = VectorIndexEdgeResult(results)

        return await result.get_edge_data(graph, need_score)

    async def retrieval_batch(self, queries, top_k):
        pass

    def _embed_text(self, text: str):
        return self.embedding_model._get_text_embedding(text)
    
    async def _update_index(self, datas: list[dict[str, Any]], meta_data_keys: list):
        # datas is a list of dicts, where each dict is like:
        # {"content": node.text, "index": node.index, "layer": node_id_to_layer.get(node.index, -1)}
        # meta_data_keys is ["index", "layer"]

        Settings.embed_model = self.config.embed_model # Ensure embed_model is set for LlamaIndex
        
        nodes_to_insert = []
        all_texts_to_embed = [data["content"] for data in datas]

        # Batch embed texts
        text_embeddings = []
        if hasattr(self, "embedding_model") and hasattr(self.embedding_model, "_get_text_embeddings"):
            # Support OpenAIEmbedding or similar interface
            batch_size = getattr(self.embedding_model, "embed_batch_size", 32)
            for i in range(0, len(all_texts_to_embed), batch_size):
                batch = all_texts_to_embed[i:i + batch_size]
                batch_embeddings = self.embedding_model._get_text_embeddings(batch)
                text_embeddings.extend(batch_embeddings)
        else:
            # Fallback for other embedding models, assuming a similar batch interface
            logger.warning("embedding_model does not support batch embedding. Falling back to single embedding loop.")
            for text in all_texts_to_embed:
                text_embeddings.append(self.embedding_model._get_text_embeddings([text])[0])

        if len(text_embeddings) != len(datas):
            logger.error(f"Mismatch in number of embeddings ({len(text_embeddings)}) and data elements ({len(datas)}). Aborting VDB update.")
            return

        for i, data_item in enumerate(datas):
            # Construct metadata dictionary for the TextNode
            node_metadata_dict = {}
            for key in meta_data_keys: # meta_data_keys is ["index", "layer"]
                if key in data_item:
                    node_metadata_dict[key] = data_item[key]
                else:
                    logger.warning(f"Metadata key '{key}' not found in data_item: {data_item}. It will be missing from VDB metadata.")
            
            # Create TextNode with text, embedding, and the constructed metadata
            # Using data_item["index"] (node_id) as LlamaIndex node ID for better traceability
            node_id_for_llama_index = str(data_item.get("index", mdhash_id(data_item["content"])))

            node = TextNode(
                id_=node_id_for_llama_index, # Use original node ID as LlamaIndex node ID
                text=data_item["content"],
                embedding=text_embeddings[i],
                metadata=node_metadata_dict,
                excluded_embed_metadata_keys=list(node_metadata_dict.keys()), # Exclude all metadata from embedding
                excluded_llm_metadata_keys=list(node_metadata_dict.keys()) # Also exclude from LLM prompt
            )
            nodes_to_insert.append(node)

        if not self._index: # If index wasn't loaded or is being rebuilt
            logger.info("Creating new FaissVectorStore and VectorStoreIndex.")
            vector_store = FaissVectorStore(faiss_index=faiss.IndexHNSWFlat(self.embedding_model.dimensions, 32)) # Ensure dimensions are correct
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex(nodes=[], storage_context=storage_context, embed_model=self.config.embed_model)
        
        if nodes_to_insert:
            logger.info(f"Inserting {len(nodes_to_insert)} nodes into Faiss index.")
            self._index.insert_nodes(nodes_to_insert) # insert_nodes can take a list of TextNode
        else:
            logger.warning("No nodes to insert into Faiss index.")
            
        logger.info(f"FaissIndex: refresh index size is {len(nodes_to_insert)}")

    async def _load_index(self) -> bool:
        try:
            Settings.embed_model = self.config.embed_model

            vector_store = FaissVectorStore.from_persist_dir(str(self.config.persist_path))
  
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.config.persist_path)
     
            self._index  =load_index_from_storage(storage_context=storage_context, embed_model=self.config.embed_model)

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
        #TODO: config the faiss index config
        vector_store = FaissVectorStore(faiss_index=faiss.IndexHNSWFlat(1024, 32))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return  VectorStoreIndex(
            nodes = [],
            storage_context=storage_context,
            embed_model= self.config.embed_model,
        )   
        # self.config.embed_model
        # return VectorStoreIndex([])

    async def _similarity_score(self, object_q, object_d):
        # For llama_index based vector database, we do not need it now!
        pass

    async def retrieval_nodes_with_score_matrix(self, query_list, top_k, graph):
        if isinstance(query_list, str):
            query_list = [query_list]
        results = await asyncio.gather(
            *[self.retrieval_nodes(query, top_k, graph, need_score=True) for query in query_list])
        reset_prob_matrix = np.zeros((len(query_list), graph.node_num))
        entity_indices = []
        scores = []

        async def set_idx_score(idx, res):
            for entity, score in zip(res[0], res[1]):
                entity_indices.append(await graph.get_node_index(entity["entity_name"]))
                scores.append(score)

        await asyncio.gather(*[set_idx_score(idx, res) for idx, res in enumerate(results)])
        reset_prob_matrix[np.arange(len(query_list)).reshape(-1, 1), entity_indices] = scores
        all_entity_weights = reset_prob_matrix.max(axis=0)  # (1, #all_entities)

        # Normalize the scores
        all_entity_weights /= all_entity_weights.sum()
        return all_entity_weights
