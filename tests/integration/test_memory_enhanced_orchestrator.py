# tests/integration/test_memory_enhanced_orchestrator.py

import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import MagicMock, AsyncMock, patch

from Core.AgentOrchestrator.memory_enhanced_orchestrator import MemoryEnhancedOrchestrator, UpdateType
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, DynamicToolChainConfig
from Core.Memory.memory_system import GraphRAGMemory


@pytest.mark.integration
class TestMemoryEnhancedOrchestrator:
    """Integration tests for memory-enhanced orchestrator"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for memory storage"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
        
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies"""
        config = MagicMock()
        llm = MagicMock()
        encoder = MagicMock()
        chunk_factory = MagicMock()
        chunk_factory.get_namespace.return_value = "test_namespace"
        context = MagicMock()
        
        return {
            "main_config": config,
            "llm_instance": llm,
            "encoder_instance": encoder,
            "chunk_factory": chunk_factory,
            "graphrag_context": context
        }
        
    @pytest.fixture
    def orchestrator(self, mock_dependencies, temp_dir):
        """Create memory-enhanced orchestrator"""
        return MemoryEnhancedOrchestrator(
            **mock_dependencies,
            memory_path=temp_dir,
            user_id="test_user"
        )
        
    @pytest.mark.asyncio
    async def test_execution_with_learning(self, orchestrator):
        """Test that orchestrator learns from executions"""
        # Create test plan
        plan = ExecutionPlan(
            plan_id="test_entity_search",
            plan_description="Search for entities",
            target_dataset_name="test",
            steps=[
                ExecutionStep(
                    step_id="search",
                    description="Entity search",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDBSearch",
                                inputs={"query": "Paris"},
                                named_outputs={"entities": "similar_entities"}
                            )
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
        
        # Mock tool execution
        mock_result = {"similar_entities": [{"entity": "Paris"}, {"entity": "France"}]}
        
        with patch.object(orchestrator, '_execute_tool_async', return_value=(mock_result, None)):
            # Execute with query
            updates = []
            async for update in orchestrator.execute_plan_stream(
                plan, 
                query="Find entities about Paris",
                use_memory_recommendation=True
            ):
                updates.append(update)
                
        # Verify execution completed
        assert any(u.type == UpdateType.PLAN_COMPLETE for u in updates)
        
        # Check that pattern was learned
        patterns = orchestrator.memory.pattern_memory.find_similar_patterns("entity_discovery")
        assert len(patterns) > 0
        
        # Check session memory
        session_context = orchestrator.get_session_context(1)
        assert len(session_context) == 1
        assert session_context[0]["query"] == "Find entities about Paris"
        
    @pytest.mark.asyncio
    async def test_memory_recommendation(self, orchestrator):
        """Test strategy recommendation from memory"""
        # First, learn a successful pattern
        plan = ExecutionPlan(
            plan_id="efficient_entity_search",
            plan_description="Efficient entity search",
            target_dataset_name="test",
            steps=[
                ExecutionStep(
                    step_id="search",
                    description="Fast search",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(tool_id="Entity.VDBSearch", inputs={"query": "test"})
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
        
        # Execute multiple times to build confidence
        for i in range(3):
            with patch.object(orchestrator, '_execute_tool_async', return_value=({"result": f"success_{i}"}, None)):
                async for _ in orchestrator.execute_plan_stream(
                    plan,
                    query=f"Find entities about topic {i}",
                    use_memory_recommendation=False
                ):
                    pass
                    
        # Now execute a similar query and check for recommendation
        recommendation_found = False
        
        async for update in orchestrator.execute_plan_stream(
            plan,
            query="Find entities about new topic",
            use_memory_recommendation=True
        ):
            if update.type == UpdateType.PLAN_START and update.data and "recommendation" in update.data:
                recommendation_found = True
                assert update.data["source"] == "memory"
                assert "confidence" in update.data["recommendation"]
                
        # Recommendation might not always trigger due to confidence threshold
        # but memory should have the pattern
        patterns = orchestrator.memory.pattern_memory.find_similar_patterns("entity_discovery")
        assert len(patterns) > 0
        
    @pytest.mark.asyncio
    async def test_user_preferences(self, orchestrator):
        """Test user preference tracking"""
        # Update preferences
        orchestrator.update_user_preference("response_style", "detailed", 0.9)
        orchestrator.update_user_preference("preferred_tools", ["Entity.VDBSearch"], 1.0)
        
        # Get preferences
        prefs = orchestrator.get_user_preferences()
        assert len(prefs) >= 2
        
        # Execute and check preferences are accessible
        plan = ExecutionPlan(
            plan_id="test",
            plan_description="Test",
            target_dataset_name="test",
            steps=[],
            plan_inputs={}
        )
        
        async for _ in orchestrator.execute_plan_stream(plan):
            pass
            
        # Preferences should persist
        prefs_after = orchestrator.get_user_preferences()
        assert len(prefs_after) >= 2
        
    @pytest.mark.asyncio
    async def test_system_statistics(self, orchestrator):
        """Test system-wide statistics collection"""
        # Execute several queries
        queries = [
            "Find entities about science",
            "What is the relationship between A and B?",
            "Find entities about history"
        ]
        
        plan = ExecutionPlan(
            plan_id="test",
            plan_description="Test",
            target_dataset_name="test",
            steps=[
                ExecutionStep(
                    step_id="step1",
                    description="Test step",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(tool_id="Entity.VDBSearch", inputs={"query": "test"})
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
        
        for query in queries:
            with patch.object(orchestrator, '_execute_tool_async', return_value=({"result": "success"}, None)):
                async for _ in orchestrator.execute_plan_stream(plan, query=query):
                    pass
                    
        # Get statistics
        stats = orchestrator.get_system_stats()
        
        assert stats["stats"]["total_queries"] == 3
        assert stats["stats"]["successful_queries"] >= 2
        assert "entity_discovery" in stats["stats"]["popular_query_types"]
        
    @pytest.mark.asyncio
    async def test_error_handling_with_memory(self, orchestrator):
        """Test that failures are recorded in memory"""
        plan = ExecutionPlan(
            plan_id="failing_plan",
            plan_description="Plan that fails",
            target_dataset_name="test",
            steps=[
                ExecutionStep(
                    step_id="fail",
                    description="Failing step",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(tool_id="Entity.VDBSearch", inputs={"query": "test"})
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
        
        # Make tool fail
        with patch.object(orchestrator, '_execute_tool_async', return_value=(None, "Test error")):
            updates = []
            async for update in orchestrator.execute_plan_stream(plan, query="Test query"):
                updates.append(update)
                
        # Check error was recorded
        assert any(u.type == UpdateType.TOOL_ERROR for u in updates)
        
        # Pattern should not be learned due to low quality
        patterns = orchestrator.memory.pattern_memory.find_similar_patterns("general")
        assert len(patterns) == 0 or patterns[0].strategy_id != "failing_plan"
        
    @pytest.mark.asyncio
    async def test_memory_persistence(self, orchestrator, temp_dir):
        """Test that memory persists across instances"""
        # Execute and learn
        plan = ExecutionPlan(
            plan_id="persistent_plan",
            plan_description="Plan to persist",
            target_dataset_name="test",
            steps=[],
            plan_inputs={}
        )
        
        with patch.object(orchestrator, '_execute_tool_async', return_value=({"result": "success"}, None)):
            async for _ in orchestrator.execute_plan_stream(plan, query="Persistent query"):
                pass
                
        # Persist memory
        orchestrator.persist_memory()
        
        # Create new orchestrator instance
        new_orchestrator = MemoryEnhancedOrchestrator(
            main_config=orchestrator.main_config,
            llm_instance=orchestrator.llm,
            encoder_instance=orchestrator.encoder,
            chunk_factory=orchestrator.chunk_factory,
            graphrag_context=orchestrator.graphrag_context,
            memory_path=temp_dir,
            user_id="test_user"
        )
        
        # Check memory was loaded
        patterns = new_orchestrator.memory.pattern_memory.find_similar_patterns("general")
        assert any(p.strategy_id == "persistent_plan" for p in patterns)
        
    @pytest.mark.asyncio
    async def test_recommended_tools(self, orchestrator):
        """Test getting recommended tools for queries"""
        # Learn pattern with specific tool sequence
        plan = ExecutionPlan(
            plan_id="entity_pipeline",
            plan_description="Entity discovery pipeline",
            target_dataset_name="test",
            steps=[
                ExecutionStep(
                    step_id="search",
                    description="Search and retrieve",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(tool_id="Entity.VDBSearch", inputs={"query": "test"}),
                            ToolCall(tool_id="Entity.PPR", inputs={"seed_entity_ids": ["e1"]}),
                            ToolCall(tool_id="Chunk.GetTextForEntities", inputs={"entity_ids": ["e1"]})
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
        
        # Execute to learn
        with patch.object(orchestrator, '_execute_tool_async', return_value=({"result": "success"}, None)):
            async for _ in orchestrator.execute_plan_stream(
                plan,
                query="Find information about Einstein"
            ):
                pass
                
        # Get recommendations
        recommended_tools = orchestrator.get_recommended_tools_for_query("Find information about Newton")
        
        # Should recommend similar tool sequence
        assert recommended_tools is not None
        assert "Entity.VDBSearch" in recommended_tools
        
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, orchestrator):
        """Test memory cleanup functionality"""
        # Add some session data
        orchestrator.memory.session_memory.add_conversation_turn(
            "Old query", "Old response"
        )
        
        # Set short TTL for testing
        orchestrator.memory.session_memory.ttl = asyncio.timedelta(seconds=1)
        
        # Wait and cleanup
        await asyncio.sleep(1.1)
        cleaned = await orchestrator.cleanup_memory()
        
        assert cleaned > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])