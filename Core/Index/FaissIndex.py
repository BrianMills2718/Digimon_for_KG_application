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

    def _get_retrieve_top_k(self):
        # Assuming self.config is an instance of FAISSIndexConfig (or a compatible config)
        # and has an attribute like 'retrieve_top_k' or similar.
        if hasattr(self.config, 'retrieve_top_k'):
            return self.config.retrieve_top_k
        else:
            logger.warning("FaissIndex config does not have 'retrieve_top_k', defaulting to 5.")
            return 5


    def _embed_text(self, text: str):
        return self.embedding_model._get_text_embedding(text)
    
    async def _update_index(self, datas: list[dict[str, Any]], meta_data_keys: list):
        logger.info(f"Starting FaissIndex._update_index with {len(datas)} data elements.")
        if not hasattr(self.config, 'embed_model') or self.config.embed_model is None:
            logger.error("FaissIndex config is missing 'embed_model'. Cannot proceed with VDB update.")
            return
        logger.info(f"Using embedding model: {type(self.config.embed_model)}")
        embed_dims = None
        if hasattr(self.config.embed_model, 'dimensions'):
            embed_dims = self.config.embed_model.dimensions
            logger.info(f"Embedding model dimensions: {embed_dims}")
        elif hasattr(self.config.embed_model, 'embed_dim'):
            embed_dims = self.config.embed_model.embed_dim
            logger.info(f"Attempting to use embed_dim: {embed_dims}")
        else:
            logger.error("Cannot determine embedding dimensions. Faiss index creation will likely fail.")
            return
        Settings.embed_model = self.config.embed_model
        nodes_to_insert = []
        all_texts_to_embed = [data["content"] for data in datas]
        text_embeddings = []
        logger.info(f"Attempting to embed {len(all_texts_to_embed)} texts for FaissIndex.")
        try:
            if hasattr(self.config.embed_model, 'aget_text_embedding_batch'):
                text_embeddings = await self.config.embed_model.aget_text_embedding_batch(all_texts_to_embed, show_progress=True)
            elif hasattr(self.config.embed_model, '_get_text_embeddings'):
                batch_size = getattr(self.config.embed_model, "embed_batch_size", 32)
                for i in range(0, len(all_texts_to_embed), batch_size):
                    batch = all_texts_to_embed[i:i + batch_size]
                    batch_embeddings = self.config.embed_model._get_text_embeddings(batch)
                    text_embeddings.extend(batch_embeddings)
            else:
                logger.warning("Embedding model does not support a known batch embedding method. Embedding one by one (this might be slow or fail).")
                for text_to_embed in all_texts_to_embed:
                    text_embeddings.append(await self.config.embed_model.aget_text_embedding(text_to_embed) if hasattr(self.config.embed_model, 'aget_text_embedding') else self.config.embed_model.get_text_embedding(text_to_embed))
        except Exception as e:
            logger.error(f"Error during text embedding batch: {e}", exc_info=True)
            return
        if len(text_embeddings) != len(datas):
            logger.error(f"Mismatch in number of embeddings ({len(text_embeddings)}) and data elements ({len(datas)}). Aborting VDB update.")
            return
        logger.info(f"Successfully generated {len(text_embeddings)} embeddings.")
        for i, data_item in enumerate(datas):
            node_metadata_dict = {}
            for key in meta_data_keys:
                if key in data_item:
                    node_metadata_dict[key] = data_item[key]
                else:
                    logger.warning(f"Metadata key '{key}' not found in data_item: {data_item}. It will be missing from VDB metadata.")
            node_id_for_llama_index = str(data_item.get("index", mdhash_id(data_item["content"])))
            node = TextNode(
                id_=node_id_for_llama_index,
                text=data_item["content"],
                embedding=text_embeddings[i],
                metadata=node_metadata_dict,
                excluded_embed_metadata_keys=list(node_metadata_dict.keys()),
                excluded_llm_metadata_keys=list(node_metadata_dict.keys())
            )
            nodes_to_insert.append(node)
        if not self._index:
            logger.info("FaissIndex: Creating new FaissVectorStore and VectorStoreIndex in _update_index.")
            if embed_dims is None:
                logger.error("FaissIndex: CRITICAL - Cannot determine embedding_model dimensions for Faiss IndexHNSWFlat creation. Aborting VDB update logic in _update_index.")
                return
            logger.info(f"FaissIndex: Initializing Faiss IndexHNSWFlat with dimensions: {embed_dims}.")
            faiss_index_instance = faiss.IndexHNSWFlat(embed_dims, 32)
            vector_store = FaissVectorStore(faiss_index=faiss_index_instance)
            logger.info(f"FaissIndex: FaissVectorStore created. faiss_index object: {vector_store.faiss_index}")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=self.config.embed_model
            )
            logger.info(f"FaissIndex: New VectorStoreIndex created. Index ID: {self._index.index_id}")
        if nodes_to_insert:
            logger.info(f"FaissIndex: Inserting {len(nodes_to_insert)} nodes into index.")
            self._index.insert_nodes(nodes_to_insert)
            logger.info(f"FaissIndex: Node insertion complete.")
        else:
            logger.warning("FaissIndex: No nodes to insert into index during _update_index.")

    async def _load_index(self) -> bool:
        logger.info(f"FaissIndex._load_index: Attempting to load index from persist_path: {self.config.persist_path}")
        try:
            Settings.embed_model = self.config.embed_model # Ensure embed_model is set for LlamaIndex
            logger.info(f"FaissIndex._load_index: Set Settings.embed_model to: {type(Settings.embed_model)}")

            if not os.path.exists(str(self.config.persist_path)):
                logger.error(f"FaissIndex._load_index: Persist path {self.config.persist_path} does NOT exist. Cannot load.")
                return False
            logger.info(f"FaissIndex._load_index: Persist path {self.config.persist_path} confirmed to exist.")
            logger.info(f"FaissIndex._load_index: Contents of persist_path: {os.listdir(str(self.config.persist_path))}")

            # Key step: Load the FaissVectorStore from the directory
            vector_store = FaissVectorStore.from_persist_dir(str(self.config.persist_path))
            logger.info(f"FaissIndex._load_index: Successfully called FaissVectorStore.from_persist_dir().")
            logger.info(f"FaissIndex._load_index: Type of loaded vector_store: {type(vector_store)}")

            if vector_store and hasattr(vector_store, '_faiss_index') and vector_store._faiss_index:
                logger.info(f"FaissIndex._load_index: Loaded vector_store has attribute _faiss_index.")
                logger.info(f"FaissIndex._load_index: vector_store._faiss_index object: {vector_store._faiss_index}")
                logger.info(f"FaissIndex._load_index: Loaded vector_store._faiss_index.ntotal: {vector_store._faiss_index.ntotal}")
            else:
                logger.warning("FaissIndex._load_index: Loaded vector_store OR its _faiss_index is None/empty/missing after from_persist_dir.")
                if vector_store:
                    logger.warning(f"FaissIndex._load_index: Attributes of loaded vector_store: {dir(vector_store)}")

            # Create StorageContext using the loaded vector_store and the same persist_dir
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(self.config.persist_path) # Pass persist_dir again as per LlamaIndex patterns
            )
            logger.info("FaissIndex._load_index: StorageContext created from defaults with loaded vector_store.")

            # Load the overall VectorStoreIndex
            self._index = load_index_from_storage(
                storage_context=storage_context,
                embed_model=self.config.embed_model # Pass embed_model here too
            )
            logger.info(f"FaissIndex._load_index: Successfully loaded index from storage. Index object: {self._index}")

            if self._index and self._index.index_struct: # Check if index structure is loaded
                 logger.info(f"FaissIndex._load_index: Index ID: {self._index.index_id}, Summary: {self._index.index_struct.summary}")
            else:
                logger.warning("FaissIndex._load_index: self._index or self._index.index_struct is None after loading.")

            return True
        except Exception as e:
            logger.error(f"FaissIndex._load_index: Loading index error: {e}", exc_info=True)
            # You might want to log self.config.persist_path and its contents here for debugging
            if hasattr(self.config, 'persist_path') and self.config.persist_path and os.path.exists(str(self.config.persist_path)):
                logger.error(f"FaissIndex._load_index: Contents of {self.config.persist_path} at time of error: {os.listdir(str(self.config.persist_path))}")
            elif hasattr(self.config, 'persist_path') and self.config.persist_path:
                logger.error(f"FaissIndex._load_index: Persist path {self.config.persist_path} does not exist at time of error.")
            else:
                logger.error(f"FaissIndex._load_index: self.config has no persist_path or it's None at time of error.")

            self._index = None # Ensure index is None if loading fails
            return False

    async def upsert(self, data: dict[str: Any]):
        pass

    def exist_index(self):
        return os.path.exists(self.config.persist_path)

    def _storage_index(self):
        if self._index and self._index.storage_context and hasattr(self.config, 'persist_path') and self.config.persist_path:
            persist_dir = str(self.config.persist_path)
            logger.info(f"FaissIndex._storage_index: Attempting to persist LlamaIndex StorageContext to directory: {persist_dir}")
            try:
                os.makedirs(persist_dir, exist_ok=True)
                logger.info(f"FaissIndex._storage_index: Ensured persist directory exists: {persist_dir}")
            except Exception as e:
                logger.error(f"FaissIndex._storage_index: Error creating persist directory {persist_dir}: {e}", exc_info=True)
                return
            try:
                # ADD THESE NEW LOGGING LINES:
                if self._index.storage_context.vector_store:
                    vector_store_instance = self._index.storage_context.vector_store
                    logger.info(f"FaissIndex._storage_index: Type of vector_store from storage_context: {type(vector_store_instance)}")
                    # import pdb; pdb.set_trace()  # Debugger removed
                    if isinstance(vector_store_instance, FaissVectorStore):
                        logger.info(f"FaissIndex._storage_index: vector_store IS a FaissVectorStore instance.")
                        logger.info(f"FaissIndex._storage_index: vector_store._faiss_index object: {vector_store_instance._faiss_index}")
                        if vector_store_instance._faiss_index:
                            logger.info(f"FaissIndex._storage_index: vector_store._faiss_index.ntotal (num vectors): {vector_store_instance._faiss_index.ntotal}")
                        else:
                            logger.warning("FaissIndex._storage_index: vector_store._faiss_index is None or empty!")
                    else:
                        logger.warning(f"FaissIndex._storage_index: vector_store is NOT a FaissVectorStore instance. Type is: {type(vector_store_instance)}")
                else:
                    logger.warning("FaissIndex._storage_index: self._index.storage_context.vector_store is None!")
                # END OF NEW LOGGING LINES

                self._index.storage_context.persist(persist_dir=persist_dir)
                logger.info(f"FaissIndex._storage_index: LlamaIndex storage_context.persist() called successfully for directory: {persist_dir}")
            except Exception as e:
                logger.error(f"FaissIndex._storage_index: Error during LlamaIndex storage_context.persist() for {persist_dir}: {e}", exc_info=True)
                return
            try:
                persisted_files = os.listdir(persist_dir)
                logger.info(f"FaissIndex._storage_index: Files found in persist directory '{persist_dir}' after persist call: {persisted_files}")
                expected_faiss_binary_filename = "default__vector_store.json"  # LlamaIndex's actual output name
                faiss_binary_path = os.path.join(persist_dir, expected_faiss_binary_filename)
                if not os.path.exists(faiss_binary_path):
                    logger.warning(
                        f"FaissIndex._storage_index: WARNING! Expected Faiss binary data file '",
                        f"{expected_faiss_binary_filename}' NOT FOUND in '{persist_dir}'. "
                        f"This will cause loading errors if this file is the intended binary Faiss index."
                    )
                else:
                    logger.info(
                        f"FaissIndex._storage_index: Found file '{expected_faiss_binary_filename}' "
                        f"in '{persist_dir}', which is expected to be the binary Faiss index."
                    )
            except Exception as e:
                logger.error(f"FaissIndex._storage_index: Error listing files in persist directory {persist_dir} after persist: {e}", exc_info=True)
        else:
            logger.warning("FaissIndex._storage_index: Skipped persistence. Conditions not met: self._index and self._index.storage_context and self.config.persist_path must be valid.")
            if not self._index:
                logger.warning("FaissIndex._storage_index: self._index is None.")
            elif not self._index.storage_context:
                logger.warning("FaissIndex._storage_index: self._index.storage_context is None.")
            elif not (hasattr(self.config, 'persist_path') and self.config.persist_path):
                logger.warning("FaissIndex._storage_index: self.config.persist_path is not set.")

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
