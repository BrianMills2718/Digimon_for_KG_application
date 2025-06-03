import faiss
import os
from Core.Common.BaseFactory import ConfigBasedFactory
from Core.Common.Logger import logger

# Try to import ColBERT, but make it optional
COLBERT_AVAILABLE = False
try:
    from Core.Index.ColBertIndex import ColBertIndex
    COLBERT_AVAILABLE = True
    logger.info("ColBERT support is available")
except ImportError as e:
    logger.warning(f"ColBERT support is not available due to missing dependencies: {e}")
    logger.warning("System will use FAISS for all vector indexing needs")

from Core.Index.Schema import (
    BaseIndexConfig,
    VectorIndexConfig,
    ColBertIndexConfig,
    FAISSIndexConfig
)
from Core.Index.VectorIndex import VectorIndex
from Core.Index.FaissIndex import FaissIndex
from Core.Index.EnhancedFaissIndex import EnhancedFaissIndex
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage


class RAGIndexFactory(ConfigBasedFactory):
    def __init__(self):
        creators = {
            VectorIndexConfig: self._create_vector_index,
            ColBertIndexConfig: self._create_colbert,
            FAISSIndexConfig: self._create_faiss,

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
        if not COLBERT_AVAILABLE:
            error_msg = (
                "ColBERT index was requested but ColBERT is not available due to dependency issues.\n"
                "This is likely due to incompatible transformers/tokenizers versions.\n"
                "Options:\n"
                "1. Set 'vdb_type: faiss' in your config instead of 'colbert'\n"
                "2. Set 'disable_colbert: true' in your Config2.yaml\n"
                "3. Fix the dependency conflict by installing: pip install transformers==4.21.0 tokenizers==0.12.1\n"
                "Note: Option 3 may break other components that need newer transformers."
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        return ColBertIndex(config)

    
    def _create_faiss(self, config):
        # Use enhanced version if batch embedding optimization is enabled
        if getattr(config, 'use_enhanced_batch_embedding', False):
            logger.info("Using EnhancedFaissIndex with batch embedding optimization")
            return EnhancedFaissIndex(config)
        return FaissIndex(config)


get_index = RAGIndexFactory().get_index
