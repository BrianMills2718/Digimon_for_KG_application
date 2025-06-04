# Core/Memory/memory_system.py

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os
import pickle
import asyncio
from enum import Enum
from pathlib import Path

from Core.Common.Logger import logger
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep
from Core.AgentTools.tool_registry import ToolCategory


class MemoryType(Enum):
    """Types of memory in the system"""
    SESSION = "session"      # Current conversation context
    PATTERN = "pattern"      # Successful execution patterns
    USER = "user"           # User preferences and history
    SYSTEM = "system"       # System-wide learnings


@dataclass
class MemoryEntry:
    """Base memory entry"""
    id: str
    timestamp: datetime
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    quality_score: float = 0.0
    
    def access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


@dataclass
class QueryPattern:
    """Pattern for successful query execution"""
    query_type: str
    query_embedding: Optional[List[float]] = None
    strategy_id: str = ""
    tool_sequence: List[str] = field(default_factory=list)
    execution_time_ms: int = 0
    quality_score: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class UserPreference:
    """User-specific preferences"""
    user_id: str
    preference_type: str
    value: Any
    confidence: float = 1.0
    update_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class MemoryStore:
    """Base class for memory storage"""
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.store: Dict[str, MemoryEntry] = {}
        self._access_queue: deque = deque(maxlen=max_size)
        
    def add(self, key: str, entry: MemoryEntry) -> None:
        """Add entry to store with LRU eviction"""
        if len(self.store) >= self.max_size:
            # Evict least recently used
            if self._access_queue:
                oldest_key = self._access_queue.popleft()
                if oldest_key in self.store:
                    del self.store[oldest_key]
        
        self.store[key] = entry
        self._access_queue.append(key)
        
    def get(self, key: str) -> Optional[MemoryEntry]:
        """Get entry and update access stats"""
        entry = self.store.get(key)
        if entry:
            # Check TTL
            if datetime.utcnow() - entry.timestamp > self.ttl:
                del self.store[key]
                return None
            
            entry.access()
            # Update access order
            if key in self._access_queue:
                self._access_queue.remove(key)
            self._access_queue.append(key)
            
        return entry
    
    def search(self, criteria: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Search entries by criteria"""
        results = []
        for entry in self.store.values():
            if self._matches_criteria(entry, criteria):
                results.append(entry)
                
        # Sort by relevance (access count and recency)
        results.sort(
            key=lambda e: (e.quality_score, e.access_count, e.last_accessed or e.timestamp),
            reverse=True
        )
        
        return results[:limit]
    
    def _matches_criteria(self, entry: MemoryEntry, criteria: Dict[str, Any]) -> bool:
        """Check if entry matches search criteria"""
        for key, value in criteria.items():
            if key in entry.content:
                if entry.content[key] != value:
                    return False
            elif key in entry.metadata:
                if entry.metadata[key] != value:
                    return False
        return True
    
    def clear_expired(self) -> int:
        """Remove expired entries"""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, entry in self.store.items()
            if current_time - entry.timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self.store[key]
            
        return len(expired_keys)


class SessionMemory(MemoryStore):
    """Memory for current session/conversation"""
    
    def __init__(self):
        super().__init__(max_size=1000, ttl_hours=4)
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_context: Dict[str, Any] = {}
        
    def add_conversation_turn(self, query: str, response: Any, metadata: Dict[str, Any] = None):
        """Add a conversation turn"""
        turn = {
            "timestamp": datetime.utcnow(),
            "query": query,
            "response": response,
            "metadata": metadata or {}
        }
        self.conversation_history.append(turn)
        
        # Store as memory entry
        entry = MemoryEntry(
            id=f"turn_{len(self.conversation_history)}",
            timestamp=turn["timestamp"],
            content=turn
        )
        self.add(entry.id, entry)
        
    def get_recent_context(self, num_turns: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        return self.conversation_history[-num_turns:]
    
    def update_context(self, key: str, value: Any):
        """Update current context"""
        self.current_context[key] = value


class PatternMemory(MemoryStore):
    """Memory for successful execution patterns"""
    
    def __init__(self):
        super().__init__(max_size=5000, ttl_hours=168)  # 1 week
        self.patterns: Dict[str, QueryPattern] = {}
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)
        
    def learn_pattern(
        self, 
        query: str,
        query_type: str,
        strategy_id: str,
        tool_sequence: List[str],
        execution_time_ms: int,
        quality_score: float,
        metadata: Dict[str, Any] = None
    ):
        """Learn from a successful execution"""
        pattern_key = f"{query_type}_{strategy_id}"
        
        if pattern_key in self.patterns:
            # Update existing pattern
            pattern = self.patterns[pattern_key]
            pattern.success_count += 1
            # Exponential moving average for scores
            alpha = 0.3
            pattern.quality_score = alpha * quality_score + (1 - alpha) * pattern.quality_score
            pattern.execution_time_ms = int(alpha * execution_time_ms + (1 - alpha) * pattern.execution_time_ms)
        else:
            # Create new pattern
            pattern = QueryPattern(
                query_type=query_type,
                strategy_id=strategy_id,
                tool_sequence=tool_sequence,
                execution_time_ms=execution_time_ms,
                quality_score=quality_score,
                success_count=1,
                metadata=metadata or {}
            )
            self.patterns[pattern_key] = pattern
            
        # Update index
        self.pattern_index[query_type].add(pattern_key)
        
        # Store as memory entry
        entry = MemoryEntry(
            id=pattern_key,
            timestamp=datetime.utcnow(),
            content={
                "query": query,
                "pattern": pattern.__dict__
            },
            quality_score=quality_score
        )
        self.add(pattern_key, entry)
        
    def find_similar_patterns(
        self, 
        query_type: str,
        min_success_rate: float = 0.7,
        min_quality_score: float = 0.6
    ) -> List[QueryPattern]:
        """Find patterns for similar queries"""
        candidates = []
        
        for pattern_key in self.pattern_index.get(query_type, []):
            pattern = self.patterns.get(pattern_key)
            if pattern and pattern.success_rate >= min_success_rate and pattern.quality_score >= min_quality_score:
                candidates.append(pattern)
                
        # Sort by quality and success rate
        candidates.sort(
            key=lambda p: (p.quality_score * p.success_rate, p.success_count),
            reverse=True
        )
        
        return candidates
    
    def update_pattern_failure(self, query_type: str, strategy_id: str):
        """Record a pattern failure"""
        pattern_key = f"{query_type}_{strategy_id}"
        if pattern_key in self.patterns:
            self.patterns[pattern_key].failure_count += 1


class UserMemory(MemoryStore):
    """Memory for user preferences and history"""
    
    def __init__(self):
        super().__init__(max_size=10000, ttl_hours=720)  # 30 days
        self.preferences: Dict[str, Dict[str, UserPreference]] = defaultdict(dict)
        self.query_history: Dict[str, List[str]] = defaultdict(list)
        
    def add_preference(
        self,
        user_id: str,
        preference_type: str,
        value: Any,
        confidence: float = 1.0
    ):
        """Add or update user preference"""
        pref_key = f"{user_id}_{preference_type}"
        
        if pref_key in self.preferences[user_id]:
            # Update existing preference
            pref = self.preferences[user_id][pref_key]
            pref.value = value
            pref.confidence = confidence
            pref.update_count += 1
            pref.last_updated = datetime.utcnow()
        else:
            # Create new preference
            pref = UserPreference(
                user_id=user_id,
                preference_type=preference_type,
                value=value,
                confidence=confidence
            )
            self.preferences[user_id][pref_key] = pref
            
        # Store as memory entry
        entry = MemoryEntry(
            id=pref_key,
            timestamp=datetime.utcnow(),
            content={
                "user_id": user_id,
                "preference": pref.__dict__
            }
        )
        self.add(pref_key, entry)
        
    def get_user_preferences(self, user_id: str) -> Dict[str, UserPreference]:
        """Get all preferences for a user"""
        return self.preferences.get(user_id, {})
    
    def add_query_to_history(self, user_id: str, query: str):
        """Add query to user's history"""
        self.query_history[user_id].append(query)
        # Keep only recent queries
        if len(self.query_history[user_id]) > 100:
            self.query_history[user_id] = self.query_history[user_id][-100:]


class SystemMemory(MemoryStore):
    """System-wide memory for global learnings"""
    
    def __init__(self):
        super().__init__(max_size=10000, ttl_hours=2160)  # 90 days
        self.global_stats: Dict[str, Any] = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_response_time": 0,
            "popular_query_types": defaultdict(int),
            "tool_usage_stats": defaultdict(int)
        }
        
    def update_query_stats(
        self,
        query_type: str,
        success: bool,
        response_time_ms: int,
        tools_used: List[str]
    ):
        """Update global query statistics"""
        self.global_stats["total_queries"] += 1
        if success:
            self.global_stats["successful_queries"] += 1
            
        # Update average response time
        n = self.global_stats["total_queries"]
        avg = self.global_stats["average_response_time"]
        self.global_stats["average_response_time"] = (avg * (n - 1) + response_time_ms) / n
        
        # Update query type popularity
        self.global_stats["popular_query_types"][query_type] += 1
        
        # Update tool usage
        for tool in tools_used:
            self.global_stats["tool_usage_stats"][tool] += 1
            
    def get_popular_strategies(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most popular query types"""
        sorted_types = sorted(
            self.global_stats["popular_query_types"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_types[:limit]


class GraphRAGMemory:
    """
    Unified memory system for GraphRAG with multiple memory levels
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory stores
        self.session_memory = SessionMemory()
        self.pattern_memory = PatternMemory()
        self.user_memory = UserMemory()
        self.system_memory = SystemMemory()
        
        # Load persisted memories
        self._load_memories()
        
        logger.info("GraphRAGMemory: Initialized multi-level memory system")
        
    def learn_from_execution(
        self,
        query: str,
        user_id: Optional[str],
        plan: ExecutionPlan,
        execution_results: Dict[str, Any],
        quality_score: float,
        execution_time_ms: int
    ):
        """Learn from a query execution"""
        # Extract query type (simplified - could use LLM for better classification)
        query_type = self._classify_query(query)
        
        # Extract tool sequence from plan
        tool_sequence = []
        for step in plan.steps:
            if hasattr(step.action, 'tools'):
                for tool in step.action.tools:
                    tool_sequence.append(tool.tool_id)
                    
        # Learn pattern if quality is good
        if quality_score >= 0.7:
            self.pattern_memory.learn_pattern(
                query=query,
                query_type=query_type,
                strategy_id=plan.plan_id,
                tool_sequence=tool_sequence,
                execution_time_ms=execution_time_ms,
                quality_score=quality_score,
                metadata={
                    "plan_description": plan.plan_description,
                    "num_steps": len(plan.steps)
                }
            )
            
        # Update user history
        if user_id:
            self.user_memory.add_query_to_history(user_id, query)
            
        # Update system stats
        self.system_memory.update_query_stats(
            query_type=query_type,
            success=quality_score >= 0.6,
            response_time_ms=execution_time_ms,
            tools_used=tool_sequence
        )
        
        # Add to session
        self.session_memory.add_conversation_turn(
            query=query,
            response=execution_results,
            metadata={
                "quality_score": quality_score,
                "execution_time_ms": execution_time_ms
            }
        )
        
    def recommend_strategy(
        self,
        query: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Recommend execution strategy based on memory"""
        query_type = self._classify_query(query)
        
        # Find successful patterns
        patterns = self.pattern_memory.find_similar_patterns(query_type)
        
        if not patterns:
            return None
            
        # Get best pattern
        best_pattern = patterns[0]
        
        # Consider user preferences if available
        if user_id:
            user_prefs = self.user_memory.get_user_preferences(user_id)
            # Could adjust strategy based on preferences
            
        return {
            "strategy_id": best_pattern.strategy_id,
            "tool_sequence": best_pattern.tool_sequence,
            "expected_quality": best_pattern.quality_score,
            "expected_time_ms": best_pattern.execution_time_ms,
            "confidence": best_pattern.success_rate
        }
        
    def get_session_context(self, num_turns: int = 5) -> List[Dict[str, Any]]:
        """Get recent session context"""
        return self.session_memory.get_recent_context(num_turns)
        
    def update_user_preference(
        self,
        user_id: str,
        preference_type: str,
        value: Any,
        confidence: float = 1.0
    ):
        """Update user preference"""
        self.user_memory.add_preference(user_id, preference_type, value, confidence)
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        return {
            "stats": self.system_memory.global_stats,
            "popular_strategies": self.system_memory.get_popular_strategies()
        }
        
    def persist_memories(self):
        """Save memories to disk"""
        # Save pattern memory
        pattern_path = self.storage_path / "patterns.pkl"
        with open(pattern_path, 'wb') as f:
            pickle.dump({
                "patterns": self.pattern_memory.patterns,
                "pattern_index": dict(self.pattern_memory.pattern_index)
            }, f)
            
        # Save user memory
        user_path = self.storage_path / "users.pkl"
        with open(user_path, 'wb') as f:
            pickle.dump({
                "preferences": dict(self.user_memory.preferences),
                "query_history": dict(self.user_memory.query_history)
            }, f)
            
        # Save system memory
        system_path = self.storage_path / "system.pkl"
        with open(system_path, 'wb') as f:
            pickle.dump(self.system_memory.global_stats, f)
            
        logger.info("GraphRAGMemory: Persisted memories to disk")
        
    def _load_memories(self):
        """Load persisted memories from disk"""
        # Load pattern memory
        pattern_path = self.storage_path / "patterns.pkl"
        if pattern_path.exists():
            try:
                with open(pattern_path, 'rb') as f:
                    data = pickle.load(f)
                    self.pattern_memory.patterns = data.get("patterns", {})
                    self.pattern_memory.pattern_index = defaultdict(set, data.get("pattern_index", {}))
                logger.info(f"GraphRAGMemory: Loaded {len(self.pattern_memory.patterns)} patterns")
            except Exception as e:
                logger.error(f"Failed to load pattern memory: {e}")
                
        # Load user memory
        user_path = self.storage_path / "users.pkl"
        if user_path.exists():
            try:
                with open(user_path, 'rb') as f:
                    data = pickle.load(f)
                    self.user_memory.preferences = defaultdict(dict, data.get("preferences", {}))
                    self.user_memory.query_history = defaultdict(list, data.get("query_history", {}))
                logger.info(f"GraphRAGMemory: Loaded user data for {len(self.user_memory.preferences)} users")
            except Exception as e:
                logger.error(f"Failed to load user memory: {e}")
                
        # Load system memory
        system_path = self.storage_path / "system.pkl"
        if system_path.exists():
            try:
                with open(system_path, 'rb') as f:
                    self.system_memory.global_stats = pickle.load(f)
                logger.info("GraphRAGMemory: Loaded system statistics")
            except Exception as e:
                logger.error(f"Failed to load system memory: {e}")
                
    def _classify_query(self, query: str) -> str:
        """Simple query classification (could be enhanced with LLM)"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["who", "person", "people", "entity"]):
            return "entity_discovery"
        elif any(word in query_lower for word in ["relationship", "connection", "between"]):
            return "relationship_analysis"
        elif any(word in query_lower for word in ["summarize", "overview", "explain"]):
            return "summarization"
        elif any(word in query_lower for word in ["compare", "difference", "similar"]):
            return "comparison"
        else:
            return "general"
            
    async def cleanup_expired(self):
        """Clean up expired entries from all memory stores"""
        total_cleaned = 0
        total_cleaned += self.session_memory.clear_expired()
        total_cleaned += self.pattern_memory.clear_expired()
        total_cleaned += self.user_memory.clear_expired()
        total_cleaned += self.system_memory.clear_expired()
        
        if total_cleaned > 0:
            logger.info(f"GraphRAGMemory: Cleaned up {total_cleaned} expired entries")
            
        return total_cleaned