from Core.Common.BaseFactory import ConfigBasedFactory
from Core.Index.ColBertIndex import ColBertIndex
from Core.Index.Schema import (
    BaseIndexConfig,
    VectorIndexConfig,
    ColBertIndexConfig
)
from Core.Index.VectorIndex import VectorIndex


class RAGIndexFactory(ConfigBasedFactory):
    def __init__(self):
        creators = {
            VectorIndexConfig: self._create_vector_index,
            ColBertIndexConfig: self._create_colbert
        }
        super().__init__(creators)

    def get_index(self, config: BaseIndexConfig):
        """Key is IndexType."""
        return super().get_instance(config)

    @classmethod
    def _create_vector_index(cls, config):
        return VectorIndex(config)

    @classmethod
    def _create_colbert(cls, config: ColBertIndexConfig):
        return ColBertIndex(config)


get_index = RAGIndexFactory().get_index
