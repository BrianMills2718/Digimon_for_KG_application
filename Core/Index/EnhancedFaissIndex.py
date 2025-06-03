"""
Enhanced FaissIndex with optimized batch embedding processing.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from Core.Index.FaissIndex import FaissIndex
from Core.Common.BatchEmbeddingProcessor import BatchEmbeddingProcessor
from Core.Common.Logger import logger
from Core.Common.PerformanceMonitor import monitor_performance
from llama_index.core.schema import TextNode
from Core.Common.Utils import mdhash_id


class EnhancedFaissIndex(FaissIndex):
    """
    Enhanced FAISS index with optimized batch embedding processing.
    
    Features:
    - Dynamic batch sizing
    - Concurrent batch processing
    - Embedding deduplication
    - Performance monitoring
    - Progress tracking
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize batch processor
        self.batch_processor = BatchEmbeddingProcessor(
            embed_model=self.config.embed_model,
            initial_batch_size=getattr(config, 'embed_batch_size', 32),
            max_batch_size=getattr(config, 'max_embed_batch_size', 256),
            min_batch_size=getattr(config, 'min_embed_batch_size', 8),
            max_concurrent_batches=getattr(config, 'max_concurrent_embed_batches', 3),
            enable_deduplication=getattr(config, 'enable_embed_deduplication', True),
            cache_embeddings=getattr(config, 'cache_embeddings', True)
        )
        
        logger.info(
            f"Initialized EnhancedFaissIndex with batch processor "
            f"(initial batch size: {self.batch_processor.current_batch_size})"
        )
    
    @monitor_performance("enhanced_update_index")
    async def _update_index(self, datas: List[Dict[str, Any]], meta_data_keys: List[str]):
        """
        Enhanced index update with optimized batch embedding.
        
        Args:
            datas: List of data items to index
            meta_data_keys: Metadata keys to include
        """
        logger.info(f"Starting EnhancedFaissIndex._update_index with {len(datas)} data elements")
        
        # Validate embedding model
        if not hasattr(self.config, 'embed_model') or self.config.embed_model is None:
            logger.error("EnhancedFaissIndex config is missing 'embed_model'")
            return
        
        # Get embedding dimensions
        embed_dims = self._get_embedding_dimensions()
        if embed_dims is None:
            logger.error("Cannot determine embedding dimensions")
            return
        
        # Extract texts for embedding
        texts_to_embed = []
        text_to_data_idx = {}
        
        for idx, data in enumerate(datas):
            text = data.get("content", "")
            if text:
                texts_to_embed.append(text)
                text_to_data_idx[len(texts_to_embed) - 1] = idx
        
        if not texts_to_embed:
            logger.warning("No texts to embed found in data")
            return
        
        logger.info(f"Extracted {len(texts_to_embed)} texts for embedding")
        
        # Generate embeddings using batch processor
        embeddings, stats = await self.batch_processor.embed_texts(
            texts_to_embed,
            show_progress=True
        )
        
        # Log embedding statistics
        logger.info(
            f"Embedding stats: {stats['successful']}/{stats['total_texts']} successful, "
            f"{stats['cached']} cached, {stats['computed']} computed, "
            f"{stats['texts_per_second']:.1f} texts/sec"
        )
        
        # Create nodes for insertion
        nodes_to_insert = []
        successful_count = 0
        failed_count = 0
        
        for text_idx, embedding in enumerate(embeddings):
            if embedding is None:
                failed_count += 1
                logger.warning(f"Skipping text {text_idx} due to embedding failure")
                continue
            
            data_idx = text_to_data_idx[text_idx]
            data_item = datas[data_idx]
            
            # Create metadata
            node_metadata = {}
            for key in meta_data_keys:
                if key in data_item:
                    node_metadata[key] = data_item[key]
                else:
                    logger.debug(f"Metadata key '{key}' not found in data item {data_idx}")
            
            # Create node
            node_id = str(data_item.get("index", mdhash_id(data_item["content"])))
            node = TextNode(
                id_=node_id,
                text=data_item["content"],
                embedding=embedding,
                metadata=node_metadata,
                excluded_embed_metadata_keys=list(node_metadata.keys()),
                excluded_llm_metadata_keys=list(node_metadata.keys())
            )
            nodes_to_insert.append(node)
            successful_count += 1
        
        logger.info(
            f"Created {successful_count} nodes for insertion "
            f"({failed_count} failed embeddings)"
        )
        
        # Initialize or update index
        if not self._index:
            self._initialize_index(embed_dims)
        
        # Insert nodes
        if nodes_to_insert:
            logger.info(f"Inserting {len(nodes_to_insert)} nodes into index")
            self._index.insert_nodes(nodes_to_insert)
            logger.info("Node insertion complete")
        else:
            logger.warning("No nodes to insert into index")
        
        # Log performance summary
        perf_summary = self.batch_processor.get_performance_summary()
        logger.info(
            f"Batch processor performance: avg batch time {perf_summary.get('average_batch_time', 0):.2f}s, "
            f"throughput {perf_summary.get('throughput', {}).get('texts_per_second', 0):.1f} texts/sec"
        )
    
    def _get_embedding_dimensions(self) -> Optional[int]:
        """Get embedding dimensions from the model."""
        if hasattr(self.config.embed_model, 'dimensions'):
            return self.config.embed_model.dimensions
        elif hasattr(self.config.embed_model, 'embed_dim'):
            return self.config.embed_model.embed_dim
        elif hasattr(self.config.embed_model, '_model_kwargs') and 'dimensions' in self.config.embed_model._model_kwargs:
            return self.config.embed_model._model_kwargs['dimensions']
        else:
            # Try to get dimensions by embedding a sample text
            try:
                sample_embedding = self.config.embed_model.get_text_embedding("test")
                if isinstance(sample_embedding, (list, np.ndarray)):
                    return len(sample_embedding)
            except Exception as e:
                logger.error(f"Failed to determine embedding dimensions: {e}")
        
        return None
    
    def _initialize_index(self, embed_dims: int):
        """Initialize the FAISS index."""
        import faiss
        from llama_index.vector_stores.faiss import FaissVectorStore
        from llama_index.core import StorageContext, VectorStoreIndex, Settings
        
        logger.info(f"Initializing FAISS index with {embed_dims} dimensions")
        
        # Set embedding model in settings
        Settings.embed_model = self.config.embed_model
        
        # Create FAISS index
        faiss_index = faiss.IndexHNSWFlat(embed_dims, 32)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create vector store index
        self._index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self.config.embed_model
        )
        
        logger.info(f"Initialized VectorStoreIndex with ID: {self._index.index_id}")
    
    def get_batch_processor_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        return self.batch_processor.get_performance_summary()
    
    def clear_embedding_cache(self):
        """Clear the embedding cache."""
        self.batch_processor.clear_cache()
    
    def optimize_batch_settings(self):
        """Optimize batch processing settings based on performance."""
        self.batch_processor.optimize_settings()