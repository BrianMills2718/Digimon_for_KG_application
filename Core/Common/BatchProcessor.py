"""
Batch processing utilities for efficient embedding and LLM operations.
"""

import asyncio
from typing import List, Dict, Any, Callable, TypeVar, Optional, Union
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .LoggerConfig import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 32
    max_concurrent_batches: int = 4
    timeout_seconds: float = 300
    retry_failed: bool = True
    show_progress: bool = True


class BatchProcessor:
    """Processes items in batches for improved efficiency."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        
    def _create_batches(self, items: List[T]) -> List[List[T]]:
        """Split items into batches."""
        batches = []
        for i in range(0, len(items), self.config.batch_size):
            batch = items[i:i + self.config.batch_size]
            batches.append(batch)
        return batches
    
    async def process_async(
        self,
        items: List[T],
        process_func: Callable[[List[T]], List[R]],
        desc: str = "Processing"
    ) -> List[R]:
        """
        Process items in batches asynchronously.
        
        Args:
            items: Items to process
            process_func: Async function to process a batch
            desc: Description for progress tracking
            
        Returns:
            Processed results in same order as input
        """
        if not items:
            return []
            
        batches = self._create_batches(items)
        results = [None] * len(batches)
        
        logger.info(f"{desc}: {len(items)} items in {len(batches)} batches")
        
        async def process_batch(idx: int, batch: List[T]):
            async with self._semaphore:
                start_time = time.time()
                try:
                    result = await asyncio.wait_for(
                        process_func(batch),
                        timeout=self.config.timeout_seconds
                    )
                    results[idx] = result
                    elapsed = time.time() - start_time
                    logger.debug(f"Batch {idx+1}/{len(batches)} completed in {elapsed:.2f}s")
                except asyncio.TimeoutError:
                    logger.error(f"Batch {idx+1} timed out after {self.config.timeout_seconds}s")
                    if self.config.retry_failed:
                        # Retry once with smaller batch
                        logger.info(f"Retrying batch {idx+1} with smaller size")
                        sub_batches = self._create_batches(batch)
                        sub_results = []
                        for sub_batch in sub_batches:
                            try:
                                sub_result = await process_func(sub_batch)
                                sub_results.extend(sub_result)
                            except Exception as e:
                                logger.error(f"Sub-batch failed: {e}")
                                sub_results.extend([None] * len(sub_batch))
                        results[idx] = sub_results
                    else:
                        results[idx] = [None] * len(batch)
                except Exception as e:
                    logger.error(f"Batch {idx+1} failed: {e}")
                    results[idx] = [None] * len(batch)
        
        # Process all batches
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        await asyncio.gather(*tasks)
        
        # Flatten results
        flat_results = []
        for batch_result in results:
            if batch_result is not None:
                flat_results.extend(batch_result)
            
        return flat_results
    
    def process_sync(
        self,
        items: List[T],
        process_func: Callable[[List[T]], List[R]],
        desc: str = "Processing",
        use_threads: bool = True
    ) -> List[R]:
        """
        Process items in batches synchronously.
        
        Args:
            items: Items to process
            process_func: Function to process a batch
            desc: Description for progress tracking
            use_threads: Whether to use thread pool
            
        Returns:
            Processed results in same order as input
        """
        if not items:
            return []
            
        batches = self._create_batches(items)
        logger.info(f"{desc}: {len(items)} items in {len(batches)} batches")
        
        if use_threads and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_concurrent_batches) as executor:
                results = list(executor.map(process_func, batches))
        else:
            results = [process_func(batch) for batch in batches]
        
        # Flatten results
        flat_results = []
        for batch_result in results:
            if batch_result is not None:
                flat_results.extend(batch_result)
                
        return flat_results


class EmbeddingBatchProcessor(BatchProcessor):
    """Specialized batch processor for embeddings."""
    
    def __init__(
        self,
        embedding_func: Callable[[List[str]], List[List[float]]],
        config: Optional[BatchConfig] = None
    ):
        """
        Initialize embedding batch processor.
        
        Args:
            embedding_func: Function to generate embeddings
            config: Batch configuration
        """
        super().__init__(config)
        self.embedding_func = embedding_func
        self._cache: Dict[str, List[float]] = {}
        
    async def embed_texts_async(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for texts in batches.
        
        Args:
            texts: Texts to embed
            use_cache: Whether to use caching
            
        Returns:
            Embeddings for each text
        """
        # Separate cached and uncached texts
        if use_cache:
            uncached_texts = []
            uncached_indices = []
            embeddings = [None] * len(texts)
            
            for i, text in enumerate(texts):
                if text in self._cache:
                    embeddings[i] = self._cache[text]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Process uncached texts
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} texts "
                       f"({len(texts) - len(uncached_texts)} cached)")
            
            new_embeddings = await self.process_async(
                uncached_texts,
                self.embedding_func,
                desc="Embedding generation"
            )
            
            # Update results and cache
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                if use_cache and embedding is not None:
                    self._cache[texts[idx]] = embedding
        
        return embeddings
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
        logger.info("Cleared embedding cache")


class ProgressTracker:
    """Simple progress tracker for batch operations."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.processed = 0
        self.start_time = time.time()
        self._last_update = 0
        
    def update(self, n: int = 1):
        """Update progress."""
        self.processed += n
        current_time = time.time()
        
        # Update every second
        if current_time - self._last_update >= 1.0:
            elapsed = current_time - self.start_time
            rate = self.processed / elapsed if elapsed > 0 else 0
            eta = (self.total - self.processed) / rate if rate > 0 else 0
            
            logger.info(
                f"{self.desc}: {self.processed}/{self.total} "
                f"({self.processed/self.total*100:.1f}%) "
                f"[{rate:.1f} items/s, ETA: {eta:.0f}s]"
            )
            self._last_update = current_time
    
    def finish(self):
        """Mark as finished."""
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0
        logger.info(
            f"{self.desc}: Completed {self.total} items "
            f"in {elapsed:.1f}s ({rate:.1f} items/s)"
        )