"""
Cache manager for LLM responses and expensive computations.
Supports both in-memory and disk-based caching.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from datetime import datetime, timedelta
import pickle
import asyncio
from functools import wraps

from .LoggerConfig import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Manages caching for expensive operations like LLM calls."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_seconds: int = 3600,
        max_memory_items: int = 1000,
        enable_disk_cache: bool = True,
        enable_memory_cache: bool = True
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for disk cache (default: .cache/digimon)
            ttl_seconds: Time to live for cache entries in seconds
            max_memory_items: Maximum items in memory cache
            enable_disk_cache: Whether to use disk caching
            enable_memory_cache: Whether to use memory caching
        """
        self.ttl_seconds = ttl_seconds
        self.max_memory_items = max_memory_items
        self.enable_disk_cache = enable_disk_cache
        self.enable_memory_cache = enable_memory_cache
        
        # Memory cache
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        
        # Disk cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "digimon"
        self.cache_dir = Path(cache_dir)
        
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"CacheManager initialized with cache_dir={self.cache_dir}, "
                   f"ttl={ttl_seconds}s, memory_cache={enable_memory_cache}, "
                   f"disk_cache={enable_disk_cache}")
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from prefix and data."""
        # Convert data to stable string representation
        if isinstance(data, dict):
            # Sort dict keys for consistent hashing
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, (list, tuple)):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
            
        # Create hash
        hash_obj = hashlib.sha256(f"{prefix}:{data_str}".encode())
        return hash_obj.hexdigest()[:16]
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.ttl_seconds
    
    def _evict_memory_cache(self):
        """Evict oldest entries if memory cache is full."""
        if len(self._memory_cache) >= self.max_memory_items:
            # Sort by access time and remove oldest
            sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[:len(sorted_keys)//4]]
            
            for key in keys_to_remove:
                del self._memory_cache[key]
                del self._access_times[key]
                
            logger.debug(f"Evicted {len(keys_to_remove)} items from memory cache")
    
    def get(self, prefix: str, key_data: Any) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            prefix: Cache key prefix (e.g., "llm_response", "embedding")
            key_data: Data to generate cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_key = self._generate_key(prefix, key_data)
        
        # Check memory cache first
        if self.enable_memory_cache and cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            if not self._is_expired(entry["timestamp"]):
                self._access_times[cache_key] = time.time()
                logger.debug(f"Cache hit (memory): {prefix}:{cache_key}")
                return entry["value"]
            else:
                # Remove expired entry
                del self._memory_cache[cache_key]
                del self._access_times[cache_key]
        
        # Check disk cache
        if self.enable_disk_cache:
            cache_file = self.cache_dir / f"{prefix}_{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        entry = pickle.load(f)
                    
                    if not self._is_expired(entry["timestamp"]):
                        # Optionally promote to memory cache
                        if self.enable_memory_cache:
                            self._evict_memory_cache()
                            self._memory_cache[cache_key] = entry
                            self._access_times[cache_key] = time.time()
                        
                        logger.debug(f"Cache hit (disk): {prefix}:{cache_key}")
                        return entry["value"]
                    else:
                        # Remove expired file
                        cache_file.unlink()
                except Exception as e:
                    logger.error(f"Error reading cache file {cache_file}: {e}")
        
        logger.debug(f"Cache miss: {prefix}:{cache_key}")
        return None
    
    def set(self, prefix: str, key_data: Any, value: Any):
        """
        Set value in cache.
        
        Args:
            prefix: Cache key prefix
            key_data: Data to generate cache key
            value: Value to cache
        """
        cache_key = self._generate_key(prefix, key_data)
        entry = {
            "value": value,
            "timestamp": time.time()
        }
        
        # Store in memory cache
        if self.enable_memory_cache:
            self._evict_memory_cache()
            self._memory_cache[cache_key] = entry
            self._access_times[cache_key] = time.time()
        
        # Store in disk cache
        if self.enable_disk_cache:
            cache_file = self.cache_dir / f"{prefix}_{cache_key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(entry, f)
                logger.debug(f"Cached to disk: {prefix}:{cache_key}")
            except Exception as e:
                logger.error(f"Error writing cache file {cache_file}: {e}")
    
    def clear(self, prefix: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            prefix: If provided, only clear entries with this prefix
        """
        if prefix:
            # Clear specific prefix
            keys_to_remove = [k for k in self._memory_cache.keys() 
                            if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._memory_cache[key]
                if key in self._access_times:
                    del self._access_times[key]
            
            if self.enable_disk_cache:
                for cache_file in self.cache_dir.glob(f"{prefix}_*.pkl"):
                    cache_file.unlink()
                    
            logger.info(f"Cleared cache for prefix: {prefix}")
        else:
            # Clear all
            self._memory_cache.clear()
            self._access_times.clear()
            
            if self.enable_disk_cache:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    
            logger.info("Cleared all cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_size = len(self._memory_cache)
        disk_files = len(list(self.cache_dir.glob("*.pkl"))) if self.enable_disk_cache else 0
        
        return {
            "memory_items": memory_size,
            "disk_files": disk_files,
            "memory_enabled": self.enable_memory_cache,
            "disk_enabled": self.enable_disk_cache,
            "ttl_seconds": self.ttl_seconds,
            "max_memory_items": self.max_memory_items
        }


def cached(
    prefix: str,
    ttl_seconds: Optional[int] = None,
    cache_manager: Optional[CacheManager] = None
):
    """
    Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix
        ttl_seconds: Override default TTL
        cache_manager: CacheManager instance (uses global if not provided)
    """
    def decorator(func: Callable) -> Callable:
        # Use provided cache manager or create a default one
        _cache_manager = cache_manager or _get_default_cache_manager()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key from function arguments
            key_data = {
                "args": args,
                "kwargs": kwargs
            }
            
            # Check cache
            cached_result = _cache_manager.get(prefix, key_data)
            if cached_result is not None:
                return cached_result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            _cache_manager.set(prefix, key_data, result)
            
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key from function arguments
            key_data = {
                "args": args,
                "kwargs": kwargs
            }
            
            # Check cache
            cached_result = _cache_manager.get(prefix, key_data)
            if cached_result is not None:
                return cached_result
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            _cache_manager.set(prefix, key_data, result)
            
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def _get_default_cache_manager() -> CacheManager:
    """Get or create default cache manager."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return _get_default_cache_manager()