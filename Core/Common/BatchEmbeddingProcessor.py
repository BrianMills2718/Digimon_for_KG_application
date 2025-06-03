"""
Enhanced batch embedding processor for improved performance.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import time

from .LoggerConfig import get_logger
from .PerformanceMonitor import monitor_performance

logger = get_logger(__name__)


class BatchEmbeddingProcessor:
    """
    Optimized batch processor for embeddings with:
    - Dynamic batch sizing based on performance
    - Concurrent batch processing
    - Deduplication to avoid redundant embeddings
    - Progress tracking
    - Error recovery
    """
    
    def __init__(
        self,
        embed_model: Any,
        initial_batch_size: int = 32,
        max_batch_size: int = 256,
        min_batch_size: int = 8,
        max_concurrent_batches: int = 3,
        enable_deduplication: bool = True,
        cache_embeddings: bool = True
    ):
        """
        Initialize batch embedding processor.
        
        Args:
            embed_model: The embedding model to use
            initial_batch_size: Starting batch size
            max_batch_size: Maximum batch size
            min_batch_size: Minimum batch size
            max_concurrent_batches: Maximum concurrent batches to process
            enable_deduplication: Whether to deduplicate texts before embedding
            cache_embeddings: Whether to cache embeddings for reuse
        """
        self.embed_model = embed_model
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_deduplication = enable_deduplication
        self.cache_embeddings = cache_embeddings
        
        # Performance tracking
        self.batch_times: List[float] = []
        self.batch_sizes: List[int] = []
        
        # Embedding cache
        self.embedding_cache: Dict[str, np.ndarray] = {} if cache_embeddings else None
        
        # Semaphore for concurrent batch processing
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    def _adjust_batch_size(self, processing_time: float, batch_size: int):
        """Dynamically adjust batch size based on performance."""
        # Target processing time per batch (in seconds)
        target_time = 2.0
        
        if processing_time < target_time * 0.5 and batch_size < self.max_batch_size:
            # Processing is fast, increase batch size
            self.current_batch_size = min(int(batch_size * 1.5), self.max_batch_size)
            logger.debug(f"Increased batch size to {self.current_batch_size}")
        elif processing_time > target_time * 2 and batch_size > self.min_batch_size:
            # Processing is slow, decrease batch size
            self.current_batch_size = max(int(batch_size * 0.7), self.min_batch_size)
            logger.debug(f"Decreased batch size to {self.current_batch_size}")
    
    async def _embed_batch(
        self,
        texts: List[str],
        batch_idx: int,
        total_batches: int,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """Embed a single batch of texts."""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                if show_progress:
                    logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(texts)} texts)")
                
                # Try different embedding methods based on what's available
                embeddings = None
                
                if hasattr(self.embed_model, 'aget_text_embedding_batch'):
                    # Async batch embedding (preferred)
                    embeddings = await self.embed_model.aget_text_embedding_batch(texts)
                elif hasattr(self.embed_model, 'get_text_embedding_batch'):
                    # Sync batch embedding
                    embeddings = await asyncio.to_thread(
                        self.embed_model.get_text_embedding_batch, texts
                    )
                elif hasattr(self.embed_model, '_get_text_embeddings'):
                    # Internal batch method
                    embeddings = await asyncio.to_thread(
                        self.embed_model._get_text_embeddings, texts
                    )
                else:
                    # Fallback to individual embeddings
                    logger.warning("No batch embedding method found, falling back to individual embeddings")
                    embeddings = []
                    for text in texts:
                        if hasattr(self.embed_model, 'aget_text_embedding'):
                            emb = await self.embed_model.aget_text_embedding(text)
                        else:
                            emb = await asyncio.to_thread(
                                self.embed_model.get_text_embedding, text
                            )
                        embeddings.append(emb)
                
                processing_time = time.time() - start_time
                self.batch_times.append(processing_time)
                self.batch_sizes.append(len(texts))
                
                # Adjust batch size based on performance
                self._adjust_batch_size(processing_time, len(texts))
                
                if show_progress:
                    logger.info(
                        f"Batch {batch_idx + 1}/{total_batches} completed in {processing_time:.2f}s "
                        f"({len(texts) / processing_time:.1f} texts/sec)"
                    )
                
                return embeddings
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx + 1}: {e}")
                # Return None embeddings for failed batch
                return [None] * len(texts)
    
    @monitor_performance("batch_embed_texts")
    async def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Embed a list of texts with optimized batch processing.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress
            metadata: Optional metadata for each text
            
        Returns:
            Tuple of (embeddings, stats)
        """
        if not texts:
            return [], {"total_texts": 0, "cached": 0, "computed": 0}
        
        start_time = time.time()
        total_texts = len(texts)
        
        # Deduplication if enabled
        if self.enable_deduplication:
            unique_texts = list(set(texts))
            text_to_idx = {text: i for i, text in enumerate(unique_texts)}
            indices = [text_to_idx[text] for text in texts]
            texts_to_embed = unique_texts
            logger.info(
                f"Deduplication: {total_texts} texts -> {len(unique_texts)} unique texts "
                f"({(1 - len(unique_texts)/total_texts) * 100:.1f}% duplicates)"
            )
        else:
            texts_to_embed = texts
            indices = list(range(len(texts)))
        
        # Check cache
        embeddings_dict = {}
        texts_to_compute = []
        texts_to_compute_indices = []
        cached_count = 0
        
        if self.cache_embeddings and self.embedding_cache:
            for i, text in enumerate(texts_to_embed):
                if text in self.embedding_cache:
                    embeddings_dict[i] = self.embedding_cache[text]
                    cached_count += 1
                else:
                    texts_to_compute.append(text)
                    texts_to_compute_indices.append(i)
            
            if cached_count > 0:
                logger.info(f"Using {cached_count} cached embeddings")
        else:
            texts_to_compute = texts_to_embed
            texts_to_compute_indices = list(range(len(texts_to_embed)))
        
        # Compute new embeddings
        if texts_to_compute:
            # Create batches
            batches = []
            batch_indices = []
            
            for i in range(0, len(texts_to_compute), self.current_batch_size):
                batch = texts_to_compute[i:i + self.current_batch_size]
                batch_idx = texts_to_compute_indices[i:i + self.current_batch_size]
                batches.append(batch)
                batch_indices.append(batch_idx)
            
            total_batches = len(batches)
            logger.info(
                f"Processing {len(texts_to_compute)} texts in {total_batches} batches "
                f"(batch size: {self.current_batch_size})"
            )
            
            # Process batches concurrently
            tasks = []
            for batch_idx, (batch, indices_batch) in enumerate(zip(batches, batch_indices)):
                task = self._embed_batch(batch, batch_idx, total_batches, show_progress)
                tasks.append((task, indices_batch, batch))
            
            # Wait for all batches to complete
            for task, indices_batch, batch in tasks:
                batch_embeddings = await task
                
                # Store results
                for idx, text, embedding in zip(indices_batch, batch, batch_embeddings):
                    if embedding is not None:
                        embeddings_dict[idx] = embedding
                        # Update cache
                        if self.cache_embeddings:
                            self.embedding_cache[text] = embedding
        
        # Reassemble embeddings in original order
        if self.enable_deduplication:
            # Map back from unique texts to original order
            final_embeddings = []
            for idx in indices:
                if idx in embeddings_dict:
                    final_embeddings.append(embeddings_dict[idx])
                else:
                    logger.warning(f"Missing embedding for index {idx}")
                    final_embeddings.append(None)
        else:
            # Direct mapping
            final_embeddings = [embeddings_dict.get(i) for i in range(len(texts))]
        
        # Calculate statistics
        total_time = time.time() - start_time
        computed_count = len(texts_to_compute)
        successful_count = sum(1 for emb in final_embeddings if emb is not None)
        
        stats = {
            "total_texts": total_texts,
            "unique_texts": len(texts_to_embed) if self.enable_deduplication else total_texts,
            "cached": cached_count,
            "computed": computed_count,
            "successful": successful_count,
            "failed": total_texts - successful_count,
            "total_time": total_time,
            "texts_per_second": total_texts / total_time if total_time > 0 else 0,
            "average_batch_size": np.mean(self.batch_sizes) if self.batch_sizes else self.current_batch_size,
            "current_batch_size": self.current_batch_size,
            "cache_hit_rate": cached_count / len(texts_to_embed) if texts_to_embed else 0
        }
        
        logger.info(
            f"Embedding complete: {successful_count}/{total_texts} texts in {total_time:.2f}s "
            f"({stats['texts_per_second']:.1f} texts/sec, "
            f"cache hit rate: {stats['cache_hit_rate']:.1%})"
        )
        
        return final_embeddings, stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of batch processing."""
        if not self.batch_times:
            return {"status": "no_data"}
        
        return {
            "total_batches": len(self.batch_times),
            "average_batch_time": np.mean(self.batch_times),
            "average_batch_size": np.mean(self.batch_sizes),
            "total_processing_time": sum(self.batch_times),
            "cache_size": len(self.embedding_cache) if self.embedding_cache else 0,
            "current_batch_size": self.current_batch_size,
            "throughput": {
                "texts_per_second": sum(self.batch_sizes) / sum(self.batch_times),
                "batches_per_minute": 60 / np.mean(self.batch_times)
            }
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self.embedding_cache:
            cache_size = len(self.embedding_cache)
            self.embedding_cache.clear()
            logger.info(f"Cleared {cache_size} cached embeddings")
    
    def optimize_settings(self):
        """Optimize batch settings based on historical performance."""
        if len(self.batch_times) < 10:
            logger.info("Not enough data to optimize settings (need at least 10 batches)")
            return
        
        # Calculate optimal batch size based on throughput
        throughputs = [
            size / time for size, time in zip(self.batch_sizes, self.batch_times)
        ]
        
        # Find batch size with best throughput
        best_idx = np.argmax(throughputs)
        optimal_batch_size = self.batch_sizes[best_idx]
        
        # Apply some smoothing
        self.current_batch_size = int(
            0.7 * self.current_batch_size + 0.3 * optimal_batch_size
        )
        self.current_batch_size = max(
            self.min_batch_size,
            min(self.max_batch_size, self.current_batch_size)
        )
        
        logger.info(
            f"Optimized batch size to {self.current_batch_size} "
            f"(based on {len(self.batch_times)} batches)"
        )