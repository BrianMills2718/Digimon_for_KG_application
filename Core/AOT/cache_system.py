"""
Advanced Caching System for AOT
Multi-level cache with invalidation logic and precomputation
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Set, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import pickle
from collections import OrderedDict
from enum import Enum

from Core.Common.Logger import logger


class CacheLevel(Enum):
    """Cache levels in order of speed"""
    MEMORY = "memory"      # In-memory cache (fastest)
    DISK = "disk"          # Disk-based cache
    REMOTE = "remote"      # Remote cache (e.g., Redis)


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    tags: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def update_access(self):
        """Update access time and count"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


class LRUCache:
    """LRU cache implementation for memory level"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry and move to end (most recently used)"""
        if key not in self.cache:
            return None
        
        entry = self.cache.pop(key)
        if not entry.is_expired():
            entry.update_access()
            self.cache[key] = entry  # Move to end
            return entry
        return None
    
    def put(self, entry: CacheEntry):
        """Add entry, evicting LRU if needed"""
        key = entry.key
        
        # Remove if exists
        if key in self.cache:
            self.cache.pop(key)
        
        # Add to end
        self.cache[key] = entry
        
        # Evict LRU if over capacity
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove first (LRU)
    
    def invalidate(self, key: str) -> bool:
        """Remove entry from cache"""
        if key in self.cache:
            self.cache.pop(key)
            return True
        return False
    
    def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all entries with any of the given tags"""
        keys_to_remove = []
        for key, entry in self.cache.items():
            if entry.tags & tags:  # Intersection
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.cache.pop(key)
        
        return len(keys_to_remove)


class CacheStatistics:
    """Track cache performance metrics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
        self.precompute_hits = 0
        self.total_latency_ms = 0.0
        self.cache_latency_ms = 0.0
        
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        total = self.hits + self.misses
        return self.total_latency_ms / total if total > 0 else 0.0
    
    @property
    def cache_speedup(self) -> float:
        """How much faster cache hits are vs misses"""
        if self.misses == 0 or self.total_latency_ms == 0:
            return 1.0
        
        avg_miss_latency = (self.total_latency_ms - self.cache_latency_ms) / self.misses
        avg_hit_latency = self.cache_latency_ms / self.hits if self.hits > 0 else 0
        
        return avg_miss_latency / avg_hit_latency if avg_hit_latency > 0 else 1.0


class AdvancedCacheSystem:
    """
    Multi-level caching system with invalidation and precomputation
    """
    
    def __init__(self, 
                 memory_size: int = 1000,
                 enable_disk_cache: bool = True,
                 enable_precomputation: bool = True):
        self.memory_cache = LRUCache(max_size=memory_size)
        self.enable_disk_cache = enable_disk_cache
        self.enable_precomputation = enable_precomputation
        
        self.statistics = CacheStatistics()
        self.invalidation_rules: List[Callable] = []
        self.precompute_queue: asyncio.Queue = asyncio.Queue()
        self.precompute_running = False
        
        # Disk cache directory
        self.disk_cache_dir = "cache/aot"
        if self.enable_disk_cache:
            import os
            os.makedirs(self.disk_cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, 
                          operation: str, 
                          params: Dict[str, Any]) -> str:
        """Generate deterministic cache key"""
        # Sort params for consistency
        sorted_params = json.dumps(params, sort_keys=True)
        combined = f"{operation}:{sorted_params}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def get(self, 
                  operation: str, 
                  params: Dict[str, Any],
                  ttl_seconds: Optional[float] = None) -> Optional[Any]:
        """
        Get value from cache, checking multiple levels
        """
        key = self._generate_cache_key(operation, params)
        start_time = datetime.utcnow()
        
        # Check memory cache first
        entry = self.memory_cache.get(key)
        if entry:
            self.statistics.hits += 1
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.statistics.cache_latency_ms += latency
            self.statistics.total_latency_ms += latency
            logger.debug(f"Cache hit (memory): {operation}")
            return entry.value
        
        # Check disk cache if enabled
        if self.enable_disk_cache:
            disk_value = await self._get_from_disk(key)
            if disk_value is not None:
                # Promote to memory cache
                entry = CacheEntry(
                    key=key,
                    value=disk_value,
                    created_at=datetime.utcnow(),
                    accessed_at=datetime.utcnow(),
                    access_count=1,
                    ttl_seconds=ttl_seconds
                )
                self.memory_cache.put(entry)
                
                self.statistics.hits += 1
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.statistics.cache_latency_ms += latency
                self.statistics.total_latency_ms += latency
                logger.debug(f"Cache hit (disk): {operation}")
                return disk_value
        
        # Cache miss
        self.statistics.misses += 1
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.statistics.total_latency_ms += latency
        return None
    
    async def put(self,
                  operation: str,
                  params: Dict[str, Any],
                  value: Any,
                  ttl_seconds: Optional[float] = None,
                  tags: Optional[Set[str]] = None):
        """
        Store value in cache
        """
        key = self._generate_cache_key(operation, params)
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
            access_count=0,
            ttl_seconds=ttl_seconds,
            tags=tags or set()
        )
        
        # Store in memory
        self.memory_cache.put(entry)
        
        # Store on disk if enabled
        if self.enable_disk_cache:
            await self._put_to_disk(key, value, ttl_seconds)
    
    async def invalidate(self, operation: str, params: Dict[str, Any]) -> bool:
        """Invalidate specific cache entry"""
        key = self._generate_cache_key(operation, params)
        
        # Invalidate from memory
        memory_invalidated = self.memory_cache.invalidate(key)
        
        # Invalidate from disk
        disk_invalidated = False
        if self.enable_disk_cache:
            disk_invalidated = await self._invalidate_from_disk(key)
        
        if memory_invalidated or disk_invalidated:
            self.statistics.invalidations += 1
            
        return memory_invalidated or disk_invalidated
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all entries with given tags"""
        count = self.memory_cache.invalidate_by_tags(tags)
        
        # Note: Disk invalidation by tags would require maintaining an index
        # For simplicity, we only invalidate memory cache by tags
        
        self.statistics.invalidations += count
        return count
    
    def add_invalidation_rule(self, rule: Callable):
        """Add a rule for automatic invalidation"""
        self.invalidation_rules.append(rule)
    
    async def check_invalidation_rules(self):
        """Check and apply invalidation rules"""
        for rule in self.invalidation_rules:
            try:
                if asyncio.iscoroutinefunction(rule):
                    await rule(self)
                else:
                    rule(self)
            except Exception as e:
                logger.error(f"Error in invalidation rule: {e}")
    
    async def precompute(self, 
                        operation: str, 
                        params_list: List[Dict[str, Any]],
                        compute_func: Callable,
                        ttl_seconds: Optional[float] = None):
        """
        Schedule precomputation of values
        """
        for params in params_list:
            await self.precompute_queue.put({
                "operation": operation,
                "params": params,
                "compute_func": compute_func,
                "ttl_seconds": ttl_seconds
            })
        
        # Start precompute worker if not running
        if not self.precompute_running:
            asyncio.create_task(self._precompute_worker())
    
    async def _precompute_worker(self):
        """Background worker for precomputation"""
        self.precompute_running = True
        
        while True:
            try:
                # Get task with timeout
                task = await asyncio.wait_for(
                    self.precompute_queue.get(),
                    timeout=60.0  # Wait up to 60 seconds
                )
                
                # Check if already cached
                cached = await self.get(task["operation"], task["params"])
                if cached is not None:
                    self.statistics.precompute_hits += 1
                    continue
                
                # Compute value
                compute_func = task["compute_func"]
                if asyncio.iscoroutinefunction(compute_func):
                    value = await compute_func(task["params"])
                else:
                    value = compute_func(task["params"])
                
                # Cache result
                await self.put(
                    task["operation"],
                    task["params"],
                    value,
                    ttl_seconds=task["ttl_seconds"]
                )
                
                logger.debug(f"Precomputed: {task['operation']}")
                
            except asyncio.TimeoutError:
                # No more tasks, exit worker
                break
            except Exception as e:
                logger.error(f"Error in precompute worker: {e}")
        
        self.precompute_running = False
    
    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        import os
        file_path = os.path.join(self.disk_cache_dir, f"{key}.cache")
        metadata_file = os.path.join(self.disk_cache_dir, f"{key}.meta")
        
        if os.path.exists(file_path):
            try:
                # Check metadata for TTL
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        created_at = datetime.fromisoformat(metadata['created_at'])
                        ttl_seconds = metadata.get('ttl_seconds')
                        
                        if ttl_seconds is not None:
                            age = (datetime.utcnow() - created_at).total_seconds()
                            if age > ttl_seconds:
                                # Expired, remove from disk
                                os.remove(file_path)
                                os.remove(metadata_file)
                                return None
                
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error reading disk cache: {e}")
        
        return None
    
    async def _put_to_disk(self, key: str, value: Any, ttl_seconds: Optional[float] = None):
        """Store value in disk cache with metadata"""
        import os
        file_path = os.path.join(self.disk_cache_dir, f"{key}.cache")
        metadata_file = os.path.join(self.disk_cache_dir, f"{key}.meta")
        
        try:
            # Write value
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Write metadata
            metadata = {
                'created_at': datetime.utcnow().isoformat(),
                'ttl_seconds': ttl_seconds
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            logger.error(f"Error writing disk cache: {e}")
    
    async def _invalidate_from_disk(self, key: str) -> bool:
        """Remove value from disk cache"""
        import os
        file_path = os.path.join(self.disk_cache_dir, f"{key}.cache")
        metadata_file = os.path.join(self.disk_cache_dir, f"{key}.meta")
        
        removed = False
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed = True
            except Exception as e:
                logger.error(f"Error removing disk cache: {e}")
        
        if os.path.exists(metadata_file):
            try:
                os.remove(metadata_file)
            except Exception as e:
                logger.error(f"Error removing disk metadata: {e}")
                
        return removed
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get cache analytics and metrics"""
        return {
            "hit_rate": f"{self.statistics.hit_rate:.1%}",
            "total_requests": self.statistics.hits + self.statistics.misses,
            "hits": self.statistics.hits,
            "misses": self.statistics.misses,
            "evictions": self.statistics.evictions,
            "invalidations": self.statistics.invalidations,
            "precompute_hits": self.statistics.precompute_hits,
            "avg_latency_ms": f"{self.statistics.avg_latency_ms:.2f}",
            "cache_speedup": f"{self.statistics.cache_speedup:.1f}x",
            "memory_entries": len(self.memory_cache.cache),
            "memory_size_mb": sum(
                len(pickle.dumps(entry.value)) for entry in self.memory_cache.cache.values()
            ) / (1024 * 1024)
        }


# Example invalidation rule
def time_based_invalidation_rule(cache: AdvancedCacheSystem):
    """Invalidate entries older than 1 hour"""
    now = datetime.utcnow()
    keys_to_remove = []
    
    for key, entry in cache.memory_cache.cache.items():
        age = (now - entry.created_at).total_seconds()
        if age > 3600:  # 1 hour
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        cache.memory_cache.invalidate(key)


# Example usage
async def example_cache_usage():
    cache = AdvancedCacheSystem()
    
    # Add invalidation rule
    cache.add_invalidation_rule(time_based_invalidation_rule)
    
    # Example computation function
    async def expensive_computation(params):
        await asyncio.sleep(0.1)  # Simulate expensive operation
        return f"Result for {params}"
    
    # First call - cache miss
    result1 = await cache.get("compute", {"x": 1, "y": 2})
    if result1 is None:
        result1 = await expensive_computation({"x": 1, "y": 2})
        await cache.put("compute", {"x": 1, "y": 2}, result1, ttl_seconds=300)
    
    # Second call - cache hit
    result2 = await cache.get("compute", {"x": 1, "y": 2})
    
    # Precompute some values
    params_list = [{"x": i, "y": i+1} for i in range(5)]
    await cache.precompute("compute", params_list, expensive_computation, ttl_seconds=300)
    
    # Wait for precomputation
    await asyncio.sleep(1)
    
    # Get analytics
    print(json.dumps(cache.get_analytics(), indent=2))


if __name__ == "__main__":
    asyncio.run(example_cache_usage())