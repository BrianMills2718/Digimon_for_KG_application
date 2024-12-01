"""
Index Config Factory.
"""
from Core.Index import get_rag_embedding
from Core.Index.Schema import (
    FAISSIndexConfig,
    ColBertIndexConfig,
    MilvusIndexConfig
)

from Core.Common.BaseFactory import ConfigBasedFactory


class IndexConfigFactory(ConfigBasedFactory):
    def __init__(self):
        creators = {
            "faiss": self._create_faiss_config,
            "milvus": self._create_milvus_config,
            "colbert": self._create_colbert_config,
        }
        super().__init__(creators)

    def get_config(self, config, namespace):
        """Key is PersistType."""
        return super().get_instance(config.vdb_type)

    @staticmethod
    def _create_faiss_config(config, namespace):
        return FAISSIndexConfig(
            persist_path=namespace,
            embed_model=get_rag_embedding(config)
        )

    @staticmethod
    def _create_milvus_config(config, namespace):
        return MilvusIndexConfig(persist_path=namespace, embed_model=get_rag_embedding(config))

    @staticmethod
    def _create_colbert_config(config, namespace):
        return ColBertIndexConfig(persist_path=namespace, index_name="nbits_2", model_name=config.colbert_checkpoint_path, nbits=2)


get_index_config = IndexConfigFactory().get_config
