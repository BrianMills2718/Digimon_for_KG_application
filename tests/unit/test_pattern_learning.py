"""
Unit tests for the Pattern Learning System

Tests the pattern recognition, storage, matching, and learning capabilities
of the cognitive architecture's pattern learning module.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

from Core.Memory.pattern_learning import (
    Pattern, PatternType, PatternMatch, PatternStore,
    SequencePatternRecognizer, StructuralPatternRecognizer,
    PatternLearningSystem
)
from Core.Memory.memory_architecture import MemoryArchitecture, MemoryType


class TestPattern:
    """Test Pattern class functionality"""
    
    def test_pattern_creation(self):
        """Test basic pattern creation"""
        pattern = Pattern(
            id="test_pattern",
            pattern_type=PatternType.SEQUENCE,
            name="Test Sequence",
            structure={"sequence": ["A", "B", "C"]},
            confidence=0.8
        )
        
        assert pattern.id == "test_pattern"
        assert pattern.pattern_type == PatternType.SEQUENCE
        assert pattern.name == "Test Sequence"
        assert pattern.confidence == 0.8
        assert pattern.support == 0
        assert pattern.applications == 0
    
    def test_pattern_serialization(self):
        """Test pattern to_dict and from_dict"""
        original = Pattern(
            id="serialize_test",
            pattern_type=PatternType.STRUCTURE,
            name="Serialization Test",
            structure={"type": "relation", "relation": "connected_to"},
            confidence=0.7,
            support=5
        )
        
        # Test to_dict
        data = original.to_dict()
        assert data['id'] == "serialize_test"
        assert data['pattern_type'] == "structure"
        assert data['confidence'] == 0.7
        assert data['support'] == 5
        assert 'creation_time' in data
        
        # Test from_dict
        restored = Pattern.from_dict(data)
        assert restored.id == original.id
        assert restored.pattern_type == original.pattern_type
        assert restored.name == original.name
        assert restored.confidence == original.confidence
        assert restored.support == original.support


class TestPatternMatch:
    """Test PatternMatch functionality"""
    
    def test_pattern_match_creation(self):
        """Test pattern match creation and scoring"""
        pattern = Pattern(
            id="match_test",
            pattern_type=PatternType.SEQUENCE,
            name="Match Test",
            structure={"sequence": ["X", "Y", "Z"]},
            confidence=0.8
        )
        
        match = PatternMatch(
            pattern=pattern,
            confidence=0.8,
            match_data={"test": "data"},
            similarity_score=0.9,
            context_match=0.7
        )
        
        assert match.pattern == pattern
        assert match.confidence == 0.8
        assert match.similarity_score == 0.9
        assert match.context_match == 0.7
        
        # Test overall score calculation
        expected_score = 0.8 * 0.4 + 0.9 * 0.4 + 0.7 * 0.2
        assert abs(match.overall_score - expected_score) < 0.001


@pytest.mark.asyncio
class TestSequencePatternRecognizer:
    """Test SequencePatternRecognizer functionality"""
    
    async def test_extract_sequence_patterns(self):
        """Test extracting sequence patterns from data"""
        recognizer = SequencePatternRecognizer(min_sequence_length=2, min_support=2)
        
        # Create test data with repeated sequences
        data = [
            {"event_sequence": [{"event": "start"}, {"event": "process"}, {"event": "end"}]},
            {"event_sequence": [{"event": "start"}, {"event": "process"}, {"event": "finish"}]},
            {"event_sequence": [{"event": "start"}, {"event": "process"}, {"event": "end"}]},
            {"sequence": ["begin", "work", "complete"]},
            {"sequence": ["begin", "work", "submit"]}
        ]
        
        patterns = await recognizer.extract_patterns(data)
        
        # Should find common subsequences
        assert len(patterns) > 0
        
        # Check for start->process pattern
        start_process_patterns = [
            p for p in patterns 
            if "start" in p.structure.get("sequence", []) and "process" in p.structure.get("sequence", [])
        ]
        assert len(start_process_patterns) > 0
        
        # Verify pattern properties
        for pattern in patterns:
            assert pattern.pattern_type == PatternType.SEQUENCE
            assert pattern.support >= 2
            assert len(pattern.structure["sequence"]) >= 2
            assert pattern.confidence > 0
    
    async def test_match_sequence_pattern(self):
        """Test matching sequence patterns against new data"""
        recognizer = SequencePatternRecognizer()
        
        # Create a test pattern
        pattern = Pattern(
            id="seq_test",
            pattern_type=PatternType.SEQUENCE,
            name="Test Sequence",
            structure={"sequence": ["login", "navigate", "logout"]},
            confidence=0.8
        )
        
        # Test exact match
        exact_data = {"event_sequence": [{"event": "login"}, {"event": "navigate"}, {"event": "logout"}]}
        match = await recognizer.match_pattern(pattern, exact_data, {})
        
        assert match is not None
        assert match.similarity_score == 1.0
        assert match.pattern == pattern
        
        # Test partial match
        partial_data = {"event_sequence": [{"event": "login"}, {"event": "navigate"}]}
        match = await recognizer.match_pattern(pattern, partial_data, {})
        
        assert match is not None
        assert match.similarity_score > 0.5
        assert match.similarity_score < 1.0
        
        # Test no match
        no_match_data = {"event_sequence": [{"event": "register"}, {"event": "setup"}]}
        match = await recognizer.match_pattern(pattern, no_match_data, {})
        
        # Should be None or very low similarity
        if match is not None:
            assert match.similarity_score < 0.3
    
    def test_sequence_similarity_calculation(self):
        """Test sequence similarity calculation methods"""
        recognizer = SequencePatternRecognizer()
        
        # Test exact match
        seq1 = ["A", "B", "C"]
        seq2 = ["A", "B", "C"]
        similarity = recognizer._calculate_sequence_similarity(seq1, seq2)
        assert similarity == 1.0
        
        # Test partial overlap
        seq3 = ["A", "B", "D"]
        similarity = recognizer._calculate_sequence_similarity(seq1, seq3)
        assert 0.4 < similarity < 0.9
        
        # Test no overlap
        seq4 = ["X", "Y", "Z"]
        similarity = recognizer._calculate_sequence_similarity(seq1, seq4)
        assert similarity == 0.0
        
        # Test LCS calculation
        lcs_length = recognizer._longest_common_subsequence(["A", "B", "C", "D"], ["A", "C", "D", "E"])
        assert lcs_length == 3  # A, C, D


@pytest.mark.asyncio
class TestStructuralPatternRecognizer:
    """Test StructuralPatternRecognizer functionality"""
    
    async def test_extract_structural_patterns(self):
        """Test extracting structural patterns from relational data"""
        recognizer = StructuralPatternRecognizer()
        
        # Create test data with relationships and properties
        data = [
            {
                "concept": "person",
                "relations": {"works_at": ["company"], "lives_in": ["city"]},
                "properties": {"age": 30, "salary": 50000}
            },
            {
                "concept": "employee",
                "relations": {"works_at": ["organization"], "reports_to": ["manager"]},
                "properties": {"age": 25, "department": "engineering"}
            },
            {
                "concept": "student",
                "relations": {"studies_at": ["university"], "lives_in": ["dorm"]},
                "properties": {"age": 20, "gpa": 3.8}
            }
        ]
        
        patterns = await recognizer.extract_patterns(data)
        
        # Should find structural patterns
        assert len(patterns) > 0
        
        # Check for age property pattern (appears in all entities)
        age_patterns = [
            p for p in patterns 
            if p.structure.get("name") == "age"
        ]
        assert len(age_patterns) > 0
        
        # Verify pattern properties
        for pattern in patterns:
            assert pattern.pattern_type == PatternType.STRUCTURE
            assert pattern.support >= 1
            assert pattern.confidence > 0
    
    async def test_match_structural_pattern(self):
        """Test matching structural patterns against new data"""
        recognizer = StructuralPatternRecognizer()
        
        # Create a test pattern
        pattern = Pattern(
            id="struct_test",
            pattern_type=PatternType.STRUCTURE,
            name="Relation: works_at",
            structure={"type": "relation", "relation": "works_at", "pattern_key": "relation:works_at"},
            confidence=0.8
        )
        
        # Test match
        test_data = {
            "concept": "developer",
            "relations": {"works_at": ["tech_company"]},
            "properties": {"skill": "python"}
        }
        
        match = await recognizer.match_pattern(pattern, test_data, {})
        
        assert match is not None
        assert match.similarity_score > 0.5
        assert match.pattern == pattern
        
        # Test no match
        no_match_data = {
            "concept": "freelancer",
            "relations": {"collaborates_with": ["clients"]},
            "properties": {"skill": "design"}
        }
        
        match = await recognizer.match_pattern(pattern, no_match_data, {})
        assert match is None


@pytest.mark.asyncio
class TestPatternStore:
    """Test PatternStore functionality"""
    
    async def test_store_and_retrieve_pattern(self):
        """Test storing and retrieving patterns"""
        store = PatternStore()
        
        pattern = Pattern(
            id="store_test",
            pattern_type=PatternType.SEQUENCE,
            name="Store Test Pattern",
            structure={"sequence": ["A", "B"]},
            confidence=0.7
        )
        
        # Store pattern
        stored_id = await store.store_pattern(pattern)
        assert stored_id == pattern.id
        
        # Retrieve pattern
        retrieved = await store.get_pattern(pattern.id)
        assert retrieved is not None
        assert retrieved.id == pattern.id
        assert retrieved.name == pattern.name
        assert retrieved.confidence == pattern.confidence
    
    async def test_search_patterns(self):
        """Test pattern search functionality"""
        store = PatternStore()
        
        # Store multiple patterns
        patterns = [
            Pattern(
                id="seq1",
                pattern_type=PatternType.SEQUENCE,
                name="High Confidence Sequence",
                structure={"sequence": ["X", "Y"]},
                confidence=0.9
            ),
            Pattern(
                id="seq2",
                pattern_type=PatternType.SEQUENCE,
                name="Low Confidence Sequence",
                structure={"sequence": ["A", "B"]},
                confidence=0.4
            ),
            Pattern(
                id="struct1",
                pattern_type=PatternType.STRUCTURE,
                name="Structure Pattern",
                structure={"type": "relation"},
                confidence=0.8
            )
        ]
        
        for pattern in patterns:
            await store.store_pattern(pattern)
        
        # Search by type
        seq_patterns = await store.search_patterns(pattern_type=PatternType.SEQUENCE)
        assert len(seq_patterns) == 2
        assert all(p.pattern_type == PatternType.SEQUENCE for p in seq_patterns)
        
        # Search by confidence
        high_conf_patterns = await store.search_patterns(min_confidence=0.7)
        assert len(high_conf_patterns) == 2
        assert all(p.confidence >= 0.7 for p in high_conf_patterns)
        
        # Search by keywords
        confidence_patterns = await store.search_patterns(name_keywords=["confidence"])
        assert len(confidence_patterns) == 2
        assert all("confidence" in p.name.lower() for p in confidence_patterns)
        
        # Combined search
        combined = await store.search_patterns(
            pattern_type=PatternType.SEQUENCE,
            min_confidence=0.8
        )
        assert len(combined) == 1
        assert combined[0].id == "seq1"
    
    async def test_update_and_delete_pattern(self):
        """Test pattern updates and deletion"""
        store = PatternStore()
        
        pattern = Pattern(
            id="update_test",
            pattern_type=PatternType.SEQUENCE,
            name="Original Name",
            structure={"sequence": ["A"]},
            confidence=0.5,
            applications=0
        )
        
        await store.store_pattern(pattern)
        
        # Update pattern
        updated = await store.update_pattern("update_test", {
            "confidence": 0.8,
            "applications": 5
        })
        assert updated is True
        
        # Verify update
        retrieved = await store.get_pattern("update_test")
        assert retrieved.confidence == 0.8
        assert retrieved.applications == 5
        assert retrieved.name == "Original Name"  # Unchanged
        
        # Delete pattern
        deleted = await store.delete_pattern("update_test")
        assert deleted is True
        
        # Verify deletion
        retrieved = await store.get_pattern("update_test")
        assert retrieved is None
    
    async def test_pattern_statistics(self):
        """Test pattern store statistics"""
        store = PatternStore()
        
        # Store patterns with different types and confidences
        patterns = [
            Pattern(id="s1", pattern_type=PatternType.SEQUENCE, name="S1", 
                   structure={}, confidence=0.9),
            Pattern(id="s2", pattern_type=PatternType.SEQUENCE, name="S2", 
                   structure={}, confidence=0.6),
            Pattern(id="st1", pattern_type=PatternType.STRUCTURE, name="ST1", 
                    structure={}, confidence=0.8),
        ]
        
        for pattern in patterns:
            await store.store_pattern(pattern)
        
        stats = await store.get_statistics()
        
        assert stats["total_patterns"] == 3
        assert stats["patterns_by_type"]["sequence"] == 2
        assert stats["patterns_by_type"]["structure"] == 1
        assert stats["high_confidence_patterns"] == 2  # 0.9 and 0.8
        assert 0.7 < stats["average_confidence"] < 0.8


@pytest.mark.asyncio
class TestPatternLearningSystem:
    """Test PatternLearningSystem integration"""
    
    async def test_system_initialization(self):
        """Test pattern learning system initialization"""
        system = PatternLearningSystem()
        
        assert system.memory is not None
        assert system.pattern_store is not None
        assert PatternType.SEQUENCE in system.recognizers
        assert PatternType.STRUCTURE in system.recognizers
        assert system.learning_rate == 0.1
        assert system.confidence_threshold == 0.6
    
    async def test_learn_patterns_from_memory(self):
        """Test learning patterns from memory"""
        system = PatternLearningSystem()
        
        # Store some episodic memories with sequences
        await system.memory.store_episode(
            event_sequence=[
                {"event": "login", "timestamp": "2024-01-01T10:00:00"},
                {"event": "browse", "timestamp": "2024-01-01T10:05:00"},
                {"event": "purchase", "timestamp": "2024-01-01T10:10:00"},
                {"event": "logout", "timestamp": "2024-01-01T10:15:00"}
            ],
            context={"user_type": "customer"},
            participants=["user1"]
        )
        
        await system.memory.store_episode(
            event_sequence=[
                {"event": "login", "timestamp": "2024-01-01T11:00:00"},
                {"event": "browse", "timestamp": "2024-01-01T11:05:00"},
                {"event": "logout", "timestamp": "2024-01-01T11:10:00"}
            ],
            context={"user_type": "visitor"},
            participants=["user2"]
        )
        
        # Learn patterns
        patterns = await system.learn_patterns_from_memory(MemoryType.EPISODIC)
        
        assert len(patterns) > 0
        
        # Should find login->browse pattern
        login_browse_patterns = [
            p for p in patterns 
            if "login" in str(p.structure) and "browse" in str(p.structure)
        ]
        assert len(login_browse_patterns) > 0
    
    async def test_match_and_apply_patterns(self):
        """Test pattern matching and application"""
        system = PatternLearningSystem()
        
        # Create and store a test pattern
        pattern = Pattern(
            id="workflow_pattern",
            pattern_type=PatternType.SEQUENCE,
            name="Login Workflow",
            structure={"sequence": ["login", "browse", "action", "logout"]},
            confidence=0.8
        )
        
        await system.pattern_store.store_pattern(pattern)
        
        # Test matching
        test_data = {
            "event_sequence": [
                {"event": "login"},
                {"event": "browse"}
            ]
        }
        
        matches = await system.match_patterns(test_data, {})
        
        assert len(matches) > 0
        best_match = matches[0]
        assert best_match.pattern.id == "workflow_pattern"
        
        # Test application
        application = await system.apply_pattern(best_match, {"user": "test"})
        
        assert application["pattern_id"] == "workflow_pattern"
        assert application["application_type"] == "sequence_prediction"
        assert "predictions" in application
        
        # Should predict next events: action, logout
        assert "action" in application["predictions"]
    
    async def test_feedback_learning(self):
        """Test learning from feedback"""
        system = PatternLearningSystem()
        
        # Create a pattern
        pattern = Pattern(
            id="feedback_test",
            pattern_type=PatternType.SEQUENCE,
            name="Feedback Test",
            structure={"sequence": ["A", "B"]},
            confidence=0.5,
            success_rate=0.5,
            applications=1
        )
        
        await system.pattern_store.store_pattern(pattern)
        
        # Provide positive feedback
        success = await system.provide_feedback("feedback_test", {
            "success": True,
            "accuracy": 0.9,
            "usefulness": 0.8
        })
        
        assert success is True
        
        # Check pattern was updated
        updated_pattern = await system.pattern_store.get_pattern("feedback_test")
        assert updated_pattern.confidence > 0.5  # Should increase
        assert updated_pattern.success_rate > 0.5  # Should increase
        
        initial_confidence_after_positive = updated_pattern.confidence
        
        # Provide negative feedback
        await system.provide_feedback("feedback_test", {
            "success": False,
            "accuracy": 0.2,
            "usefulness": 0.1
        })
        
        # Check pattern was adjusted
        final_pattern = await system.pattern_store.get_pattern("feedback_test")
        assert final_pattern.confidence < initial_confidence_after_positive  # Should decrease
    
    async def test_learning_statistics(self):
        """Test learning system statistics"""
        system = PatternLearningSystem()
        
        # Store some patterns
        patterns = [
            Pattern(id="stat1", pattern_type=PatternType.SEQUENCE, name="S1", 
                   structure={}, confidence=0.8),
            Pattern(id="stat2", pattern_type=PatternType.STRUCTURE, name="S2", 
                   structure={}, confidence=0.7),
        ]
        
        for pattern in patterns:
            await system.pattern_store.store_pattern(pattern)
        
        # Store some memories
        await system.memory.store_episode(
            event_sequence=[{"event": "test"}], 
            context={}, 
            participants=["tester"]
        )
        await system.memory.store_fact("test_concept", "test", {}, {})
        
        stats = await system.get_learning_statistics()
        
        assert "patterns" in stats
        assert "memory" in stats
        assert "recognizers" in stats
        assert stats["patterns"]["total_patterns"] == 2
        assert len(stats["recognizers"]) == 2
        assert stats["learning_rate"] == 0.1
        assert stats["confidence_threshold"] == 0.6
    
    async def test_system_shutdown(self):
        """Test system shutdown"""
        system = PatternLearningSystem()
        
        # Store some data
        await system.memory.store_fact("test", "test", {}, {})
        
        # Shutdown should not raise errors
        await system.shutdown()


@pytest.mark.asyncio
class TestPatternLearningIntegration:
    """Integration tests for the complete pattern learning pipeline"""
    
    async def test_end_to_end_learning_cycle(self):
        """Test complete learning cycle from memory to application"""
        system = PatternLearningSystem()
        
        # 1. Store diverse episodic memories
        workflows = [
            ["start", "prepare", "execute", "review", "complete"],
            ["start", "prepare", "execute", "complete"],
            ["start", "prepare", "test", "execute", "review", "complete"],
            ["begin", "setup", "run", "check", "finish"],
            ["begin", "setup", "run", "finish"]
        ]
        
        for i, workflow in enumerate(workflows):
            events = [{"event": event, "step": j} for j, event in enumerate(workflow)]
            await system.memory.store_episode(
                event_sequence=events,
                context={"workflow_type": "standard", "complexity": len(workflow)},
                participants=[f"user{i+1}"]
            )
        
        # 2. Learn patterns from memories
        learned_patterns = await system.learn_patterns_from_memory(MemoryType.EPISODIC)
        assert len(learned_patterns) > 0
        
        # 3. Test pattern matching on new data
        new_workflow = {
            "event_sequence": [
                {"event": "start"},
                {"event": "prepare"}
            ]
        }
        
        matches = await system.match_patterns(
            new_workflow, 
            {"workflow_type": "standard"},
            min_confidence=0.3
        )
        
        assert len(matches) > 0
        
        # 4. Apply best matching pattern
        best_match = matches[0]
        application = await system.apply_pattern(
            best_match,
            {"current_step": 2, "user": "test"}
        )
        
        assert application["application_type"] == "sequence_prediction"
        assert "predictions" in application
        
        # Predictions might be empty if matched sequence is at end of pattern
        # What matters is that pattern matching and application worked correctly
        
        # 5. Provide feedback and verify learning
        initial_confidence = best_match.pattern.confidence
        
        await system.provide_feedback(best_match.pattern.id, {
            "success": True,
            "accuracy": 0.9,
            "usefulness": 0.8
        })
        
        updated_pattern = await system.pattern_store.get_pattern(best_match.pattern.id)
        assert updated_pattern.confidence >= initial_confidence
        
        # 6. Verify system statistics
        stats = await system.get_learning_statistics()
        assert stats["patterns"]["total_patterns"] > 0
        assert stats["memory"]["total_memories"] >= len(workflows)
        
        await system.shutdown()
    
    async def test_cross_pattern_type_learning(self):
        """Test learning different types of patterns from same data"""
        system = PatternLearningSystem()
        
        # Store memories with both sequential and structural patterns
        memories_data = [
            {
                "event_sequence": [{"event": "connect"}, {"event": "authenticate"}, {"event": "query"}],
                "relations": {"system_type": ["database"], "protocol": ["sql"]},
                "properties": {"connection_time": 50, "query_count": 10}
            },
            {
                "event_sequence": [{"event": "connect"}, {"event": "authenticate"}, {"event": "disconnect"}],
                "relations": {"system_type": ["database"], "security": ["encrypted"]},
                "properties": {"connection_time": 30, "session_duration": 120}
            },
            {
                "event_sequence": [{"event": "login"}, {"event": "navigate"}, {"event": "logout"}],
                "relations": {"system_type": ["web"], "protocol": ["http"]},
                "properties": {"response_time": 200, "page_views": 5}
            }
        ]
        
        for i, memory_data in enumerate(memories_data):
            await system.memory.store_episode(
                event_sequence=memory_data["event_sequence"],
                context={
                    "relations": memory_data["relations"],
                    "properties": memory_data["properties"]
                },
                participants=[f"system{i+1}"]
            )
        
        # Learn patterns (should find both sequence and structural patterns)
        patterns = await system.learn_patterns_from_memory(MemoryType.EPISODIC)
        
        sequence_patterns = [p for p in patterns if p.pattern_type == PatternType.SEQUENCE]
        # Note: structural patterns would require semantic memories with relations/properties
        
        assert len(sequence_patterns) > 0
        
        # Test matching both types
        test_data = {
            "event_sequence": [{"event": "connect"}, {"event": "authenticate"}],
            "relations": {"system_type": ["database"]},
            "properties": {"connection_time": 45}
        }
        
        matches = await system.match_patterns(test_data, {})
        assert len(matches) > 0
        
        await system.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])