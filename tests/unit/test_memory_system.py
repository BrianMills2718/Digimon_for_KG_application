# tests/unit/test_memory_system.py

import pytest
import asyncio
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from Core.Memory.memory_system import (
    GraphRAGMemory,
    SessionMemory,
    PatternMemory,
    UserMemory,
    SystemMemory,
    MemoryEntry,
    QueryPattern,
    UserPreference
)
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, DynamicToolChainConfig


class TestSessionMemory:
    """Test cases for session memory"""
    
    def test_conversation_tracking(self):
        """Test conversation turn tracking"""
        session = SessionMemory()
        
        # Add conversation turns
        session.add_conversation_turn(
            query="What is GraphRAG?",
            response={"answer": "GraphRAG is..."},
            metadata={"quality": 0.9}
        )
        
        session.add_conversation_turn(
            query="Tell me more",
            response={"answer": "Additionally..."},
            metadata={"quality": 0.85}
        )
        
        # Check history
        assert len(session.conversation_history) == 2
        assert session.conversation_history[0]["query"] == "What is GraphRAG?"
        
        # Get recent context
        recent = session.get_recent_context(1)
        assert len(recent) == 1
        assert recent[0]["query"] == "Tell me more"
        
    def test_context_update(self):
        """Test context management"""
        session = SessionMemory()
        
        # Update context
        session.update_context("current_topic", "GraphRAG")
        session.update_context("user_level", "expert")
        
        assert session.current_context["current_topic"] == "GraphRAG"
        assert session.current_context["user_level"] == "expert"
        
    def test_memory_expiration(self):
        """Test TTL expiration"""
        session = SessionMemory()
        session.ttl = timedelta(seconds=1)  # Short TTL for testing
        
        # Add entry
        entry = MemoryEntry(
            id="test",
            timestamp=datetime.utcnow() - timedelta(seconds=2),
            content={"data": "old"}
        )
        session.add("test", entry)
        
        # Should be expired
        assert session.get("test") is None


class TestPatternMemory:
    """Test cases for pattern memory"""
    
    def test_pattern_learning(self):
        """Test learning from successful executions"""
        pattern_mem = PatternMemory()
        
        # Learn pattern
        pattern_mem.learn_pattern(
            query="Find entities about France",
            query_type="entity_discovery",
            strategy_id="entity_vdb_search",
            tool_sequence=["Entity.VDBSearch", "Chunk.GetTextForEntities"],
            execution_time_ms=1500,
            quality_score=0.9
        )
        
        # Check pattern was learned
        patterns = pattern_mem.find_similar_patterns("entity_discovery")
        assert len(patterns) == 1
        assert patterns[0].strategy_id == "entity_vdb_search"
        assert patterns[0].quality_score == 0.9
        assert patterns[0].success_count == 1
        
    def test_pattern_update(self):
        """Test updating existing patterns"""
        pattern_mem = PatternMemory()
        
        # Learn initial pattern
        for i in range(3):
            pattern_mem.learn_pattern(
                query=f"Query {i}",
                query_type="entity_discovery",
                strategy_id="entity_vdb_search",
                tool_sequence=["Entity.VDBSearch"],
                execution_time_ms=1000 + i * 100,
                quality_score=0.8 + i * 0.05
            )
            
        patterns = pattern_mem.find_similar_patterns("entity_discovery")
        assert patterns[0].success_count == 3
        # Check exponential moving average was applied
        assert 0.85 < patterns[0].quality_score < 0.9
        
    def test_pattern_failure_tracking(self):
        """Test failure tracking"""
        pattern_mem = PatternMemory()
        
        # Learn pattern
        pattern_mem.learn_pattern(
            query="Test",
            query_type="test_type",
            strategy_id="test_strategy",
            tool_sequence=["Tool1"],
            execution_time_ms=1000,
            quality_score=0.8
        )
        
        # Record failures
        pattern_mem.update_pattern_failure("test_type", "test_strategy")
        pattern_mem.update_pattern_failure("test_type", "test_strategy")
        
        patterns = pattern_mem.find_similar_patterns("test_type")
        assert patterns[0].failure_count == 2
        assert patterns[0].success_rate == 1/3  # 1 success, 2 failures
        
    def test_pattern_filtering(self):
        """Test pattern filtering by quality and success rate"""
        pattern_mem = PatternMemory()
        
        # Add good pattern
        pattern_mem.learn_pattern(
            query="Good query",
            query_type="test",
            strategy_id="good_strategy",
            tool_sequence=["Tool1"],
            execution_time_ms=1000,
            quality_score=0.9
        )
        
        # Add poor pattern
        pattern_mem.learn_pattern(
            query="Poor query",
            query_type="test",
            strategy_id="poor_strategy",
            tool_sequence=["Tool2"],
            execution_time_ms=5000,
            quality_score=0.4
        )
        
        # Should only return high quality patterns
        patterns = pattern_mem.find_similar_patterns("test", min_quality_score=0.7)
        assert len(patterns) == 1
        assert patterns[0].strategy_id == "good_strategy"


class TestUserMemory:
    """Test cases for user memory"""
    
    def test_preference_management(self):
        """Test user preference storage and updates"""
        user_mem = UserMemory()
        
        # Add preferences
        user_mem.add_preference("user123", "response_detail", "verbose", 0.9)
        user_mem.add_preference("user123", "preferred_graphs", ["ERGraph", "TreeGraph"], 1.0)
        
        # Get preferences
        prefs = user_mem.get_user_preferences("user123")
        assert len(prefs) == 2
        
        # Update preference
        user_mem.add_preference("user123", "response_detail", "concise", 0.95)
        prefs = user_mem.get_user_preferences("user123")
        pref = prefs["user123_response_detail"]
        assert pref.value == "concise"
        assert pref.update_count == 1
        
    def test_query_history(self):
        """Test query history tracking"""
        user_mem = UserMemory()
        
        # Add queries
        for i in range(150):
            user_mem.add_query_to_history("user123", f"Query {i}")
            
        # Should keep only last 100
        assert len(user_mem.query_history["user123"]) == 100
        assert user_mem.query_history["user123"][0] == "Query 50"
        assert user_mem.query_history["user123"][-1] == "Query 149"


class TestSystemMemory:
    """Test cases for system memory"""
    
    def test_global_statistics(self):
        """Test global stats tracking"""
        sys_mem = SystemMemory()
        
        # Update stats
        sys_mem.update_query_stats(
            query_type="entity_discovery",
            success=True,
            response_time_ms=1500,
            tools_used=["Entity.VDBSearch", "Entity.PPR"]
        )
        
        sys_mem.update_query_stats(
            query_type="relationship_analysis",
            success=True,
            response_time_ms=2500,
            tools_used=["Relationship.VDBSearch"]
        )
        
        sys_mem.update_query_stats(
            query_type="entity_discovery",
            success=False,
            response_time_ms=5000,
            tools_used=["Entity.VDBSearch"]
        )
        
        # Check stats
        assert sys_mem.global_stats["total_queries"] == 3
        assert sys_mem.global_stats["successful_queries"] == 2
        assert sys_mem.global_stats["average_response_time"] == 3000
        assert sys_mem.global_stats["popular_query_types"]["entity_discovery"] == 2
        assert sys_mem.global_stats["tool_usage_stats"]["Entity.VDBSearch"] == 2
        
    def test_popular_strategies(self):
        """Test getting popular query types"""
        sys_mem = SystemMemory()
        
        # Add various query types
        for _ in range(5):
            sys_mem.update_query_stats("type_a", True, 1000, ["Tool1"])
        for _ in range(3):
            sys_mem.update_query_stats("type_b", True, 1000, ["Tool2"])
        for _ in range(7):
            sys_mem.update_query_stats("type_c", True, 1000, ["Tool3"])
            
        popular = sys_mem.get_popular_strategies(2)
        assert len(popular) == 2
        assert popular[0][0] == "type_c"
        assert popular[0][1] == 7
        assert popular[1][0] == "type_a"
        assert popular[1][1] == 5


class TestGraphRAGMemory:
    """Test cases for the unified memory system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for memory storage"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
        
    @pytest.fixture
    def memory(self, temp_dir):
        """Create memory instance with temp storage"""
        return GraphRAGMemory(storage_path=temp_dir)
        
    def test_learning_from_execution(self, memory):
        """Test learning from query execution"""
        # Create mock plan
        plan = ExecutionPlan(
            plan_id="test_plan",
            plan_description="Test entity search",
            target_dataset_name="test",
            steps=[
                ExecutionStep(
                    step_id="search",
                    description="Search entities",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(tool_id="Entity.VDBSearch", inputs={"query": "test"})
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
        
        # Learn from execution
        memory.learn_from_execution(
            query="Find entities about Paris",
            user_id="user123",
            plan=plan,
            execution_results={"outputs": {"entities": ["Paris", "France"]}},
            quality_score=0.9,
            execution_time_ms=1500
        )
        
        # Check that pattern was learned
        patterns = memory.pattern_memory.find_similar_patterns("entity_discovery")
        assert len(patterns) == 1
        
        # Check session was updated
        assert len(memory.session_memory.conversation_history) == 1
        
        # Check system stats
        stats = memory.get_system_stats()
        assert stats["stats"]["total_queries"] == 1
        assert stats["stats"]["successful_queries"] == 1
        
    def test_strategy_recommendation(self, memory):
        """Test strategy recommendation based on memory"""
        # Learn some patterns first
        plan = ExecutionPlan(
            plan_id="entity_search_plan",
            plan_description="Entity search strategy",
            target_dataset_name="test",
            steps=[],
            plan_inputs={}
        )
        
        # Learn successful pattern multiple times
        for i in range(5):
            memory.learn_from_execution(
                query=f"Find entities about {i}",
                user_id="user123",
                plan=plan,
                execution_results={"success": True},
                quality_score=0.85 + i * 0.02,
                execution_time_ms=1000 + i * 100
            )
            
        # Get recommendation
        recommendation = memory.recommend_strategy("Find entities about Rome")
        
        assert recommendation is not None
        assert recommendation["strategy_id"] == "entity_search_plan"
        assert recommendation["confidence"] > 0.9
        assert recommendation["expected_quality"] > 0.85
        
    def test_query_classification(self, memory):
        """Test query type classification"""
        test_cases = [
            ("Who is the president?", "entity_discovery"),
            ("What is the relationship between A and B?", "relationship_analysis"),
            ("Summarize the document", "summarization"),
            ("Compare X and Y", "comparison"),
            ("Random query", "general")
        ]
        
        for query, expected_type in test_cases:
            query_type = memory._classify_query(query)
            assert query_type == expected_type
            
    def test_persistence(self, memory, temp_dir):
        """Test memory persistence to disk"""
        # Add some data
        memory.pattern_memory.learn_pattern(
            query="Test",
            query_type="test",
            strategy_id="test_strategy",
            tool_sequence=["Tool1"],
            execution_time_ms=1000,
            quality_score=0.9
        )
        
        memory.user_memory.add_preference("user123", "pref1", "value1")
        memory.system_memory.update_query_stats("test", True, 1000, ["Tool1"])
        
        # Persist
        memory.persist_memories()
        
        # Create new memory instance and check data was loaded
        memory2 = GraphRAGMemory(storage_path=temp_dir)
        
        assert len(memory2.pattern_memory.patterns) == 1
        assert "user123" in memory2.user_memory.preferences
        assert memory2.system_memory.global_stats["total_queries"] == 1
        
    @pytest.mark.asyncio
    async def test_cleanup_expired(self, memory):
        """Test cleanup of expired entries"""
        # Add expired entry
        memory.session_memory.ttl = timedelta(seconds=1)
        memory.session_memory.add_conversation_turn("Old query", "Old response")
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Cleanup
        cleaned = await memory.cleanup_expired()
        assert cleaned >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])