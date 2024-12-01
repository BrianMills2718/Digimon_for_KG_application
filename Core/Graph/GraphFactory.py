"""
Graph Factory.
"""
import os
from Core.Graph.BaseGraph import BaseGraph
from Core.Graph.ERGraph import ERGraph
from Core.Graph.PassageGraph import PassageGraph
from Core.Graph.TreeGraph import TreeGraph
from Core.Graph.RKGraph import RKGraph
from pathlib import Path

from Core.Common.BaseFactory import ConfigBasedFactory



class GraphFactory(ConfigBasedFactory):
    def __init__(self):
        creators = {
            "er_graph": self._create_er_graph,
            "rkg_graph": self._create_rkg_graph,
            "tree_graph": self._create_tree_graph,
            "passage_graph": self._crease_passage_graph
        }
        super().__init__(creators)

    def get_graph(self, config, **kwargs) -> BaseGraph:
        """Key is PersistType."""
        return super().get_instance(config, **kwargs)

    def _create_ergraph(self, config: FAISSIndexConfig, **kwargs) -> VectorStoreIndex:
        if os.path.exists(config.persist_path):

            vector_store = FaissVectorStore.from_persist_dir(str(config.persist_path))
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=config.persist_path)
            return self._index_from_storage(storage_context=storage_context, config=config, **kwargs)

        else:
            vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(config.embed_model.dimensions))
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            return VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=config.embed_model,
            )

    def _create_bm25(self, config: BM25IndexConfig, **kwargs) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(persist_dir=config.persist_path)

        return self._index_from_storage(storage_context=storage_context, config=config, **kwargs)

    def _create_milvus(self, config: MilvusIndexConfig, **kwargs) -> VectorStoreIndex:
        vector_store = MilvusVectorStore(collection_name=config.collection_name, uri=config.uri, token=config.token)

        return self._index_from_vector_store(vector_store=vector_store, config=config, **kwargs)

    def _crease_colbert(self, config: ColBertIndexConfig, **kwargs) -> VectorStoreIndex:
        #     import pdb
        #     # pdb.set_trace()
        index_path = (Path(config.persist_path) / config.index_name)
        if os.path.exists(index_path):
            return ColbertIndex.load_from_disk(config.persist_path, config.index_name)
        else:

            return ColbertIndex(**config.model_dump())

    def _index_from_storage(
            self, storage_context: StorageContext, config: BaseIndexConfig, **kwargs
    ) -> VectorStoreIndex:
        embed_model = self._extract_embed_model(config, **kwargs)

        return load_index_from_storage(storage_context=storage_context, embed_model=embed_model)

    def _index_from_vector_store(
            self, vector_store: BasePydanticVectorStore, config: BaseIndexConfig, **kwargs
    ) -> VectorStoreIndex:
        embed_model = self._extract_embed_model(config, **kwargs)

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

    def _extract_embed_model(self, config, **kwargs) -> BaseEmbedding:
        return self._val_from_config_or_kwargs("embed_model", config, **kwargs)


get_graph = GraphFactory().get_graph