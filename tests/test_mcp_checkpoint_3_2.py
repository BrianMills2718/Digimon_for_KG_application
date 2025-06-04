"""
Tests for Phase 3, Checkpoint 3.2: Memory Architecture

Tests the comprehensive memory system with episodic and semantic memories,
consolidation mechanisms, and intelligent retrieval.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

from Core.Memory.memory_architecture import (
    MemoryType, MemoryEntry, EpisodicMemory, SemanticMemory,
    InMemoryStore, MemoryConsolidation, MemoryRetrieval,
    MemoryArchitecture
)


@pytest.fixture
def memory_store():
    """Create an in-memory store"""
    return InMemoryStore()


@pytest.fixture
def memory_architecture():
    """Create a memory architecture instance"""
    return MemoryArchitecture()


class TestMemoryEntry:
    """Test basic memory entry functionality"""
    
    def test_memory_entry_creation(self):
        """Test creating a basic memory entry"""
        entry = MemoryEntry(
            id="test_1",
            memory_type=MemoryType.SEMANTIC,
            content="Test content",
            timestamp=datetime.utcnow(),
            importance=0.8
        )
        
        assert entry.id == "test_1"
        assert entry.memory_type == MemoryType.SEMANTIC
        assert entry.content == "Test content"
        assert entry.importance == 0.8
        assert entry.access_count == 0
        assert entry.associations == []
    
    def test_memory_entry_serialization(self):
        """Test memory entry serialization"""
        entry = MemoryEntry(
            id="test_2",
            memory_type=MemoryType.EPISODIC,
            content={"event": "test"},
            timestamp=datetime.utcnow(),
            metadata={"source": "test"}
        )
        
        # Convert to dict
        data = entry.to_dict()
        assert data['id'] == "test_2"
        assert data['memory_type'] == "episodic"
        assert isinstance(data['timestamp'], str)
        
        # Create from dict
        restored = MemoryEntry.from_dict(data)
        assert restored.id == entry.id
        assert restored.memory_type == entry.memory_type
        assert restored.content == entry.content
    
    def test_episodic_memory(self):
        """Test episodic memory specific features"""
        episode = EpisodicMemory(
            id="episode_1",
            content="Meeting with team",
            timestamp=datetime.utcnow(),
            event_sequence=[
                {"time": "10:00", "event": "Introduction"},
                {"time": "10:15", "event": "Discussion"}
            ],
            participants=["Alice", "Bob"],
            emotional_valence=0.7
        )
        
        assert episode.memory_type == MemoryType.EPISODIC
        assert len(episode.event_sequence) == 2
        assert "Alice" in episode.participants
        assert episode.emotional_valence == 0.7
    
    def test_semantic_memory(self):
        """Test semantic memory specific features"""
        fact = SemanticMemory(
            id="fact_1",
            content="Paris is the capital of France",
            timestamp=datetime.utcnow(),
            concept="Paris",
            category="city",
            properties={"country": "France", "type": "capital"},
            relations={"capital_of": ["France"]},
            confidence=0.95
        )
        
        assert fact.memory_type == MemoryType.SEMANTIC
        assert fact.concept == "Paris"
        assert fact.category == "city"
        assert fact.properties["country"] == "France"
        assert "France" in fact.relations["capital_of"]
        assert fact.confidence == 0.95


class TestInMemoryStore:
    """Test the in-memory store implementation"""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, memory_store):
        """Test storing and retrieving memories"""
        entry = MemoryEntry(
            id="test_3",
            memory_type=MemoryType.WORKING,
            content="Working memory item",
            timestamp=datetime.utcnow()
        )
        
        # Store memory
        memory_id = await memory_store.store(entry)
        assert memory_id == "test_3"
        
        # Retrieve memory
        retrieved = await memory_store.retrieve("test_3")
        assert retrieved is not None
        assert retrieved.content == "Working memory item"
        assert retrieved.access_count == 1
        assert retrieved.last_accessed is not None
    
    @pytest.mark.asyncio
    async def test_search_by_type(self, memory_store):
        """Test searching memories by type"""
        # Store different types of memories
        episodic = EpisodicMemory(
            id="ep_1",
            content="Episode",
            timestamp=datetime.utcnow()
        )
        semantic = SemanticMemory(
            id="sem_1",
            content="Fact",
            timestamp=datetime.utcnow(),
            concept="test"
        )
        
        await memory_store.store(episodic)
        await memory_store.store(semantic)
        
        # Search by type
        episodic_results = await memory_store.search({
            'memory_type': MemoryType.EPISODIC
        })
        assert len(episodic_results) == 1
        assert episodic_results[0].id == "ep_1"
        
        semantic_results = await memory_store.search({
            'memory_type': MemoryType.SEMANTIC
        })
        assert len(semantic_results) == 1
        assert semantic_results[0].id == "sem_1"
    
    @pytest.mark.asyncio
    async def test_search_by_time_range(self, memory_store):
        """Test searching memories by time range"""
        now = datetime.utcnow()
        
        # Store memories at different times
        old_memory = MemoryEntry(
            id="old_1",
            memory_type=MemoryType.SEMANTIC,
            content="Old memory",
            timestamp=now - timedelta(days=2)
        )
        recent_memory = MemoryEntry(
            id="recent_1",
            memory_type=MemoryType.SEMANTIC,
            content="Recent memory",
            timestamp=now - timedelta(hours=1)
        )
        
        await memory_store.store(old_memory)
        await memory_store.store(recent_memory)
        
        # Search last 24 hours
        results = await memory_store.search({
            'start_time': now - timedelta(days=1)
        })
        assert len(results) == 1
        assert results[0].id == "recent_1"
    
    @pytest.mark.asyncio
    async def test_update_memory(self, memory_store):
        """Test updating memory entries"""
        entry = MemoryEntry(
            id="update_1",
            memory_type=MemoryType.SEMANTIC,
            content="Original content",
            timestamp=datetime.utcnow(),
            importance=0.5
        )
        
        await memory_store.store(entry)
        
        # Update memory
        success = await memory_store.update("update_1", {
            'content': "Updated content",
            'importance': 0.9
        })
        assert success is True
        
        # Verify update
        updated = await memory_store.retrieve("update_1")
        assert updated.content == "Updated content"
        assert updated.importance == 0.9
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_store):
        """Test deleting memory entries"""
        entry = MemoryEntry(
            id="delete_1",
            memory_type=MemoryType.WORKING,
            content="To be deleted",
            timestamp=datetime.utcnow()
        )
        
        await memory_store.store(entry)
        
        # Delete memory
        success = await memory_store.delete("delete_1")
        assert success is True
        
        # Verify deletion
        retrieved = await memory_store.retrieve("delete_1")
        assert retrieved is None


class TestMemoryConsolidation:
    """Test memory consolidation mechanisms"""
    
    @pytest.mark.asyncio
    async def test_episodic_to_semantic_consolidation(self, memory_store):
        """Test consolidating episodic memories into semantic knowledge"""
        consolidation = MemoryConsolidation(memory_store)
        consolidation.pattern_threshold = 2  # Lower threshold for testing
        
        # Create multiple episodes with common participants
        for i in range(3):
            episode = EpisodicMemory(
                id=f"episode_{i}",
                content=f"Meeting {i}",
                timestamp=datetime.utcnow() - timedelta(hours=i),
                event_sequence=[{"event": "discussion"}],
                participants=["Alice", "Bob"]
            )
            await memory_store.store(episode)
        
        # Run consolidation
        semantic_memories = await consolidation.consolidate_episodic_to_semantic()
        
        # Check that patterns were extracted
        assert len(semantic_memories) >= 1
        alice_facts = [m for m in semantic_memories if m.concept == "Alice"]
        assert len(alice_facts) > 0
        assert alice_facts[0].category == "entity"
    
    @pytest.mark.asyncio
    async def test_strengthen_associations(self, memory_store):
        """Test strengthening associations between memories"""
        consolidation = MemoryConsolidation(memory_store)
        
        # Create memories
        memory1 = MemoryEntry(
            id="mem_1",
            memory_type=MemoryType.SEMANTIC,
            content="Memory 1",
            timestamp=datetime.utcnow()
        )
        memory2 = MemoryEntry(
            id="mem_2",
            memory_type=MemoryType.SEMANTIC,
            content="Memory 2",
            timestamp=datetime.utcnow()
        )
        
        await memory_store.store(memory1)
        await memory_store.store(memory2)
        
        # Strengthen associations
        await consolidation.strengthen_associations("mem_1", ["mem_2"])
        
        # Verify associations
        updated1 = await memory_store.retrieve("mem_1")
        updated2 = await memory_store.retrieve("mem_2")
        
        assert "mem_2" in updated1.associations
        assert "mem_1" in updated2.associations
    
    @pytest.mark.asyncio
    async def test_memory_decay(self, memory_store):
        """Test memory decay mechanism"""
        consolidation = MemoryConsolidation(memory_store)
        
        # Create memory with high importance
        memory = MemoryEntry(
            id="decay_1",
            memory_type=MemoryType.SEMANTIC,
            content="Decaying memory",
            timestamp=datetime.utcnow() - timedelta(days=7),
            importance=1.0
        )
        
        await memory_store.store(memory)
        
        # Apply decay
        await consolidation.decay_memories(decay_factor=0.8)
        
        # Check importance decreased
        decayed = await memory_store.retrieve("decay_1")
        assert decayed.importance == 0.8


class TestMemoryRetrieval:
    """Test memory retrieval mechanisms"""
    
    @pytest.mark.asyncio
    async def test_relevance_scoring(self, memory_store):
        """Test relevance scoring for memory retrieval"""
        retrieval = MemoryRetrieval(memory_store)
        
        # Create memories with different relevance
        relevant = MemoryEntry(
            id="relevant_1",
            memory_type=MemoryType.SEMANTIC,
            content="Python programming language",
            timestamp=datetime.utcnow(),
            importance=0.9,
            access_count=5
        )
        
        less_relevant = MemoryEntry(
            id="less_relevant_1",
            memory_type=MemoryType.SEMANTIC,
            content="Java programming",
            timestamp=datetime.utcnow() - timedelta(days=7),
            importance=0.3,
            access_count=1
        )
        
        await memory_store.store(relevant)
        await memory_store.store(less_relevant)
        
        # Retrieve with query
        results = await retrieval.retrieve_relevant_memories(
            "Python language",
            {},
            limit=2
        )
        
        assert len(results) == 2
        assert results[0][0].id == "relevant_1"  # Higher score
        assert results[0][1] > results[1][1]  # Score comparison
    
    @pytest.mark.asyncio
    async def test_retrieve_associated_memories(self, memory_store):
        """Test retrieving associated memories"""
        retrieval = MemoryRetrieval(memory_store)
        
        # Create connected memories
        root = MemoryEntry(
            id="root",
            memory_type=MemoryType.SEMANTIC,
            content="Root memory",
            timestamp=datetime.utcnow(),
            associations=["child1", "child2"]
        )
        
        child1 = MemoryEntry(
            id="child1",
            memory_type=MemoryType.SEMANTIC,
            content="Child 1",
            timestamp=datetime.utcnow(),
            associations=["grandchild1"]
        )
        
        child2 = MemoryEntry(
            id="child2",
            memory_type=MemoryType.SEMANTIC,
            content="Child 2",
            timestamp=datetime.utcnow()
        )
        
        grandchild1 = MemoryEntry(
            id="grandchild1",
            memory_type=MemoryType.SEMANTIC,
            content="Grandchild 1",
            timestamp=datetime.utcnow()
        )
        
        await memory_store.store(root)
        await memory_store.store(child1)
        await memory_store.store(child2)
        await memory_store.store(grandchild1)
        
        # Retrieve associated memories with depth 2
        associated = await retrieval.retrieve_associated_memories(
            "root",
            depth=2
        )
        
        assert len(associated) == 3
        ids = [m.id for m in associated]
        assert "child1" in ids
        assert "child2" in ids
        assert "grandchild1" in ids


class TestMemoryArchitecture:
    """Test the complete memory architecture"""
    
    @pytest.mark.asyncio
    async def test_store_episode(self, memory_architecture):
        """Test storing episodic memories"""
        episode_id = await memory_architecture.store_episode(
            event_sequence=[
                {"time": "10:00", "event": "Start meeting"},
                {"time": "10:30", "event": "Discussion"},
                {"time": "11:00", "event": "End meeting"}
            ],
            context={"location": "Conference Room A"},
            participants=["Alice", "Bob", "Charlie"],
            emotional_valence=0.5,
            importance=0.7
        )
        
        assert episode_id.startswith("episode_")
        
        # Verify stored
        episode = await memory_architecture.episodic_store.retrieve(episode_id)
        assert episode is not None
        assert len(episode.event_sequence) == 3
        assert "Alice" in episode.participants
    
    @pytest.mark.asyncio
    async def test_store_fact(self, memory_architecture):
        """Test storing semantic facts"""
        fact_id = await memory_architecture.store_fact(
            concept="Python",
            category="programming_language",
            properties={
                "type": "interpreted",
                "creator": "Guido van Rossum",
                "year": 1991
            },
            relations={
                "used_for": ["web_development", "data_science", "automation"]
            },
            confidence=0.95,
            importance=0.8
        )
        
        assert fact_id.startswith("fact_")
        
        # Verify stored
        fact = await memory_architecture.semantic_store.retrieve(fact_id)
        assert fact is not None
        assert fact.concept == "Python"
        assert fact.properties["type"] == "interpreted"
    
    @pytest.mark.asyncio
    async def test_recall_episodes(self, memory_architecture):
        """Test recalling relevant episodes"""
        # Store test episodes
        await memory_architecture.store_episode(
            event_sequence=[{"event": "Python workshop"}],
            context={"topic": "programming"},
            participants=["instructor"],
            importance=0.8
        )
        
        await memory_architecture.store_episode(
            event_sequence=[{"event": "Coffee break"}],
            context={"topic": "social"},
            participants=["colleagues"],
            importance=0.3
        )
        
        # Recall relevant episodes
        results = await memory_architecture.recall_episodes(
            "Python programming workshop",
            {"interest": "learning"},
            limit=5
        )
        
        assert len(results) > 0
        assert results[0][1] > 0  # Has relevance score
    
    @pytest.mark.asyncio
    async def test_recall_facts(self, memory_architecture):
        """Test recalling relevant facts"""
        # Store test facts
        await memory_architecture.store_fact(
            concept="Machine Learning",
            category="technology",
            properties={"field": "AI"},
            relations={"subset_of": ["Artificial Intelligence"]},
            importance=0.9
        )
        
        await memory_architecture.store_fact(
            concept="Coffee",
            category="beverage",
            properties={"type": "hot"},
            relations={},
            importance=0.2
        )
        
        # Recall relevant facts
        results = await memory_architecture.recall_facts(
            "AI and machine learning",
            {"domain": "technology"},
            limit=5
        )
        
        assert len(results) > 0
        # Machine Learning should be more relevant
        ml_results = [r for r in results if r[0].concept == "Machine Learning"]
        assert len(ml_results) > 0
    
    @pytest.mark.asyncio
    async def test_working_memory(self, memory_architecture):
        """Test working memory management"""
        # Add to working memory
        wm_id = await memory_architecture.update_working_memory(
            "Current task: Implement memory system",
            importance=0.9
        )
        
        assert wm_id.startswith("working_")
        
        # Verify in working memory
        working_items = await memory_architecture.working_memory.search({
            'memory_type': MemoryType.WORKING
        })
        assert len(working_items) > 0
    
    @pytest.mark.asyncio
    async def test_memory_stats(self, memory_architecture):
        """Test getting memory statistics"""
        # Store various memories
        await memory_architecture.store_episode(
            [{"event": "test"}], {}, ["tester"]
        )
        await memory_architecture.store_fact(
            "test", "test", {}, {}
        )
        await memory_architecture.update_working_memory("test")
        
        # Get stats
        stats = await memory_architecture.get_memory_stats()
        
        assert stats["episodic_memories"] >= 1
        assert stats["semantic_memories"] >= 1
        assert stats["working_memories"] >= 1
        assert stats["total_memories"] >= 3
    
    @pytest.mark.asyncio
    async def test_shutdown(self, memory_architecture):
        """Test clean shutdown of background tasks"""
        await memory_architecture.shutdown()
        # Should complete without errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])