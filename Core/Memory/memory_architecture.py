"""
Memory Architecture for Cognitive System

This module implements a comprehensive memory system with episodic and semantic
memory stores, consolidation mechanisms, and intelligent retrieval.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory in the system"""
    EPISODIC = "episodic"      # Events and experiences
    SEMANTIC = "semantic"       # Facts and concepts
    PROCEDURAL = "procedural"   # Skills and procedures
    WORKING = "working"         # Short-term active memory


@dataclass
class MemoryEntry:
    """Base class for all memory entries"""
    id: str
    memory_type: MemoryType
    content: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance: float = 0.5
    decay_rate: float = 0.1
    associations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.last_accessed:
            data['last_accessed'] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        data['memory_type'] = MemoryType(data['memory_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_accessed'):
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


@dataclass
class EpisodicMemory(MemoryEntry):
    """Memory of specific events and experiences"""
    # Override with default
    memory_type: MemoryType = field(default=MemoryType.EPISODIC, init=False)
    event_sequence: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    emotional_valence: float = 0.0  # -1 to 1 (negative to positive)
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    duration: Optional[timedelta] = None


@dataclass 
class SemanticMemory(MemoryEntry):
    """Memory of facts, concepts, and general knowledge"""
    # Override with default
    memory_type: MemoryType = field(default=MemoryType.SEMANTIC, init=False)
    concept: str = ""
    category: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=dict)
    confidence: float = 0.8
    source: Optional[str] = None


class MemoryStore(ABC):
    """Abstract base class for memory stores"""
    
    @abstractmethod
    async def store(self, memory: MemoryEntry) -> str:
        """Store a memory entry"""
        pass
    
    @abstractmethod
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: Dict[str, Any],
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search for memories matching criteria"""
        pass
    
    @abstractmethod
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry"""
        pass
    
    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        pass


class InMemoryStore(MemoryStore):
    """In-memory implementation of memory store"""
    
    def __init__(self):
        self.memories: Dict[str, MemoryEntry] = {}
        self.index_by_type: Dict[MemoryType, Set[str]] = defaultdict(set)
        self.index_by_concept: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def store(self, memory: MemoryEntry) -> str:
        """Store a memory entry"""
        async with self._lock:
            self.memories[memory.id] = memory
            self.index_by_type[memory.memory_type].add(memory.id)
            
            # Index semantic memories by concept
            if isinstance(memory, SemanticMemory):
                self.index_by_concept[memory.concept.lower()].add(memory.id)
            
            return memory.id
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory"""
        memory = self.memories.get(memory_id)
        if memory:
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
        return memory
    
    async def search(
        self,
        query: Dict[str, Any],
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search for memories matching criteria"""
        results = []
        
        # Filter by memory type if specified
        if 'memory_type' in query:
            memory_ids = self.index_by_type.get(query['memory_type'], set())
            candidates = [self.memories[mid] for mid in memory_ids]
        else:
            candidates = list(self.memories.values())
        
        # Filter by concept for semantic memories
        if 'concept' in query:
            concept_ids = self.index_by_concept.get(query['concept'].lower(), set())
            candidates = [m for m in candidates if m.id in concept_ids]
        
        # Filter by time range
        if 'start_time' in query:
            start_time = query['start_time']
            candidates = [m for m in candidates if m.timestamp >= start_time]
        
        if 'end_time' in query:
            end_time = query['end_time']
            candidates = [m for m in candidates if m.timestamp <= end_time]
        
        # Sort by relevance (could be improved with better scoring)
        candidates.sort(key=lambda m: (m.importance, m.access_count), reverse=True)
        
        return candidates[:limit]
    
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry"""
        async with self._lock:
            if memory_id not in self.memories:
                return False
            
            memory = self.memories[memory_id]
            for key, value in updates.items():
                if hasattr(memory, key):
                    setattr(memory, key, value)
            
            return True
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        async with self._lock:
            if memory_id not in self.memories:
                return False
            
            memory = self.memories[memory_id]
            del self.memories[memory_id]
            
            # Remove from indices
            self.index_by_type[memory.memory_type].discard(memory_id)
            if isinstance(memory, SemanticMemory):
                self.index_by_concept[memory.concept.lower()].discard(memory_id)
            
            return True


class MemoryConsolidation:
    """Handles memory consolidation and organization"""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.consolidation_threshold = 0.7
        self.pattern_threshold = 3
    
    async def consolidate_episodic_to_semantic(
        self,
        time_window: timedelta = timedelta(hours=24)
    ) -> List[SemanticMemory]:
        """Extract semantic knowledge from episodic memories"""
        # Get recent episodic memories
        end_time = datetime.utcnow()
        start_time = end_time - time_window
        
        episodes = await self.memory_store.search({
            'memory_type': MemoryType.EPISODIC,
            'start_time': start_time,
            'end_time': end_time
        }, limit=100)
        
        # Extract patterns and create semantic memories
        patterns = self._extract_patterns(episodes)
        semantic_memories = []
        
        for pattern in patterns:
            if pattern['frequency'] >= self.pattern_threshold:
                semantic_mem = SemanticMemory(
                    id=f"semantic_{datetime.utcnow().timestamp()}",
                    content=pattern['content'],
                    timestamp=datetime.utcnow(),
                    concept=pattern['concept'],
                    category=pattern['category'],
                    properties=pattern['properties'],
                    confidence=pattern['frequency'] / len(episodes),
                    importance=0.7,
                    source="consolidation"
                )
                
                await self.memory_store.store(semantic_mem)
                semantic_memories.append(semantic_mem)
        
        return semantic_memories
    
    def _extract_patterns(
        self,
        episodes: List[EpisodicMemory]
    ) -> List[Dict[str, Any]]:
        """Extract recurring patterns from episodic memories"""
        patterns = []
        
        # Simple pattern extraction - could be enhanced with ML
        entity_frequency = defaultdict(int)
        relation_frequency = defaultdict(int)
        
        for episode in episodes:
            # Count entity occurrences
            for participant in episode.participants:
                entity_frequency[participant] += 1
            
            # Count relationship patterns in event sequences
            for event in episode.event_sequence:
                if 'relation' in event:
                    relation_frequency[event['relation']] += 1
        
        # Create patterns from frequent items
        for entity, freq in entity_frequency.items():
            if freq >= self.pattern_threshold:
                patterns.append({
                    'concept': entity,
                    'category': 'entity',
                    'frequency': freq,
                    'content': f"{entity} frequently appears in experiences",
                    'properties': {'frequency': freq}
                })
        
        return patterns
    
    async def strengthen_associations(self, memory_id: str, related_ids: List[str]):
        """Strengthen associations between memories"""
        memory = await self.memory_store.retrieve(memory_id)
        if not memory:
            return
        
        # Add bidirectional associations
        for related_id in related_ids:
            if related_id not in memory.associations:
                memory.associations.append(related_id)
            
            # Update related memory too
            related_memory = await self.memory_store.retrieve(related_id)
            if related_memory and memory_id not in related_memory.associations:
                related_memory.associations.append(memory_id)
                await self.memory_store.update(
                    related_id,
                    {'associations': related_memory.associations}
                )
        
        await self.memory_store.update(
            memory_id,
            {'associations': memory.associations}
        )
    
    async def decay_memories(self, decay_factor: float = 0.95):
        """Apply time-based decay to memory importance"""
        all_memories = await self.memory_store.search({}, limit=10000)
        
        for memory in all_memories:
            # Skip recently accessed memories
            if memory.last_accessed:
                time_since_access = datetime.utcnow() - memory.last_accessed
                if time_since_access < timedelta(hours=1):
                    continue
            
            # Apply decay
            new_importance = memory.importance * decay_factor
            await self.memory_store.update(
                memory.id,
                {'importance': new_importance}
            )


class MemoryRetrieval:
    """Intelligent memory retrieval with relevance scoring"""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
    
    async def retrieve_relevant_memories(
        self,
        query: str,
        context: Dict[str, Any],
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10
    ) -> List[Tuple[MemoryEntry, float]]:
        """Retrieve memories relevant to query with relevance scores"""
        candidates = []
        
        # Get candidate memories
        if memory_types:
            for mem_type in memory_types:
                memories = await self.memory_store.search(
                    {'memory_type': mem_type},
                    limit=limit * 2
                )
                candidates.extend(memories)
        else:
            candidates = await self.memory_store.search({}, limit=limit * 3)
        
        # Score memories by relevance
        scored_memories = []
        for memory in candidates:
            score = self._calculate_relevance_score(memory, query, context)
            scored_memories.append((memory, score))
        
        # Sort by score and return top results
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return scored_memories[:limit]
    
    def _calculate_relevance_score(
        self,
        memory: MemoryEntry,
        query: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for a memory"""
        score = 0.0
        
        # Text similarity (simplified - could use embeddings)
        query_lower = query.lower()
        if isinstance(memory.content, str):
            content_lower = memory.content.lower()
            # Simple keyword matching
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            score += overlap * 0.2
        
        # Importance weight
        score += memory.importance * 0.3
        
        # Recency weight
        age = datetime.utcnow() - memory.timestamp
        recency_score = 1.0 / (1.0 + age.total_seconds() / 86400)  # Day decay
        score += recency_score * 0.2
        
        # Access frequency weight
        access_score = min(1.0, memory.access_count / 10.0)
        score += access_score * 0.1
        
        # Context matching
        if context and memory.metadata:
            context_overlap = len(
                set(context.keys()) & set(memory.metadata.keys())
            )
            score += context_overlap * 0.2
        
        return min(1.0, score)
    
    async def retrieve_associated_memories(
        self,
        memory_id: str,
        depth: int = 2,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Retrieve memories associated with a given memory"""
        visited = set()
        to_visit = [(memory_id, 0)]
        associated = []
        
        while to_visit and len(associated) < limit:
            current_id, current_depth = to_visit.pop(0)
            
            if current_id in visited or current_depth > depth:
                continue
            
            visited.add(current_id)
            memory = await self.memory_store.retrieve(current_id)
            
            if memory and current_id != memory_id:
                associated.append(memory)
            
            if memory and current_depth < depth:
                for assoc_id in memory.associations:
                    if assoc_id not in visited:
                        to_visit.append((assoc_id, current_depth + 1))
        
        return associated[:limit]


class MemoryArchitecture:
    """Main memory architecture coordinating all memory components"""
    
    def __init__(self):
        self.episodic_store = InMemoryStore()
        self.semantic_store = InMemoryStore()
        self.working_memory = InMemoryStore()
        
        self.consolidation = MemoryConsolidation(self.episodic_store)
        self.episodic_retrieval = MemoryRetrieval(self.episodic_store)
        self.semantic_retrieval = MemoryRetrieval(self.semantic_store)
        
        self._consolidation_task = None
        self._decay_task = None
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        async def consolidation_loop():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run hourly
                    await self.consolidation.consolidate_episodic_to_semantic()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Consolidation error: {e}")
        
        async def decay_loop():
            while True:
                try:
                    await asyncio.sleep(86400)  # Run daily
                    await self.consolidation.decay_memories()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Decay error: {e}")
        
        try:
            self._consolidation_task = asyncio.create_task(consolidation_loop())
            self._decay_task = asyncio.create_task(decay_loop())
        except RuntimeError:
            logger.debug("No event loop for background tasks")
    
    async def store_episode(
        self,
        event_sequence: List[Dict[str, Any]],
        context: Dict[str, Any],
        participants: List[str],
        emotional_valence: float = 0.0,
        importance: float = 0.5
    ) -> str:
        """Store an episodic memory"""
        episode = EpisodicMemory(
            id=f"episode_{datetime.utcnow().timestamp()}",
            content=f"Episode with {len(event_sequence)} events",
            timestamp=datetime.utcnow(),
            event_sequence=event_sequence,
            context=context,
            participants=participants,
            emotional_valence=emotional_valence,
            importance=importance
        )
        
        return await self.episodic_store.store(episode)
    
    async def store_fact(
        self,
        concept: str,
        category: str,
        properties: Dict[str, Any],
        relations: Dict[str, List[str]],
        confidence: float = 0.8,
        importance: float = 0.6
    ) -> str:
        """Store a semantic fact"""
        fact = SemanticMemory(
            id=f"fact_{datetime.utcnow().timestamp()}",
            content=f"Fact about {concept}",
            timestamp=datetime.utcnow(),
            concept=concept,
            category=category,
            properties=properties,
            relations=relations,
            confidence=confidence,
            importance=importance
        )
        
        return await self.semantic_store.store(fact)
    
    async def recall_episodes(
        self,
        query: str,
        context: Dict[str, Any],
        limit: int = 5
    ) -> List[Tuple[EpisodicMemory, float]]:
        """Recall relevant episodic memories"""
        memories = await self.episodic_retrieval.retrieve_relevant_memories(
            query,
            context,
            [MemoryType.EPISODIC],
            limit
        )
        return [(m, s) for m, s in memories if isinstance(m, EpisodicMemory)]
    
    async def recall_facts(
        self,
        query: str,
        context: Dict[str, Any],
        limit: int = 10
    ) -> List[Tuple[SemanticMemory, float]]:
        """Recall relevant semantic facts"""
        memories = await self.semantic_retrieval.retrieve_relevant_memories(
            query,
            context,
            [MemoryType.SEMANTIC],
            limit
        )
        return [(m, s) for m, s in memories if isinstance(m, SemanticMemory)]
    
    async def update_working_memory(
        self,
        content: Any,
        importance: float = 0.8
    ) -> str:
        """Add item to working memory"""
        # Clear old working memory items
        working_items = await self.working_memory.search(
            {'memory_type': MemoryType.WORKING}
        )
        
        # Keep only recent high-importance items
        cutoff_time = datetime.utcnow() - timedelta(minutes=30)
        for item in working_items:
            if item.timestamp < cutoff_time or item.importance < 0.3:
                await self.working_memory.delete(item.id)
        
        # Add new item
        working_item = MemoryEntry(
            id=f"working_{datetime.utcnow().timestamp()}",
            memory_type=MemoryType.WORKING,
            content=content,
            timestamp=datetime.utcnow(),
            importance=importance
        )
        
        return await self.working_memory.store(working_item)
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory system"""
        episodic_count = len(await self.episodic_store.search({}, limit=10000))
        semantic_count = len(await self.semantic_store.search({}, limit=10000))
        working_count = len(await self.working_memory.search({}, limit=100))
        
        return {
            "episodic_memories": episodic_count,
            "semantic_memories": semantic_count,
            "working_memories": working_count,
            "total_memories": episodic_count + semantic_count + working_count
        }
    
    async def shutdown(self):
        """Shutdown background tasks"""
        if self._consolidation_task:
            self._consolidation_task.cancel()
        if self._decay_task:
            self._decay_task.cancel()