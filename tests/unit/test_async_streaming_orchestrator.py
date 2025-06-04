# tests/unit/test_async_streaming_orchestrator.py

import pytest
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch

from Core.AgentOrchestrator.async_streaming_orchestrator import (
    AsyncStreamingOrchestrator, 
    StreamingUpdate, 
    UpdateType,
    ToolCategory
)
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, DynamicToolChainConfig
from Core.AgentSchema.tool_contracts import EntityVDBSearchInputs, EntityVDBSearchOutputs, VDBSearchResultItem
from Core.AgentSchema.context import GraphRAGContext


class TestAsyncStreamingOrchestrator:
    """Test cases for the async streaming orchestrator"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for the orchestrator"""
        config = MagicMock()
        llm = MagicMock()
        encoder = MagicMock()
        chunk_factory = MagicMock()
        chunk_factory.get_namespace.return_value = "test_namespace"
        context = MagicMock(spec=GraphRAGContext)
        
        return {
            "main_config": config,
            "llm_instance": llm,
            "encoder_instance": encoder,
            "chunk_factory": chunk_factory,
            "graphrag_context": context
        }
    
    @pytest.fixture
    def orchestrator(self, mock_dependencies):
        """Create an orchestrator instance with mocks"""
        return AsyncStreamingOrchestrator(**mock_dependencies)
    
    @pytest.fixture
    def simple_plan(self):
        """Create a simple execution plan for testing"""
        return ExecutionPlan(
            plan_id="test_plan_1",
            plan_description="Test plan for streaming",
            steps=[
                ExecutionStep(
                    step_id="step1",
                    description="Search entities",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDBSearch",
                                inputs={"query": "test query"},
                                named_outputs={"entities": "similar_entities"}
                            )
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
    
    @pytest.fixture
    def parallel_plan(self):
        """Create a plan with parallel read-only tools"""
        return ExecutionPlan(
            plan_id="test_plan_parallel",
            plan_description="Test plan with parallel execution",
            steps=[
                ExecutionStep(
                    step_id="step1",
                    description="Parallel searches",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDBSearch",
                                inputs={"query": "query1"},
                                named_outputs={"entities1": "similar_entities"}
                            ),
                            ToolCall(
                                tool_id="Entity.PPR",
                                inputs={"seed_entity_ids": ["entity1"]},
                                named_outputs={"entities2": "ranked_entities"}
                            ),
                            ToolCall(
                                tool_id="Entity.Onehop",
                                inputs={"entity_ids": ["entity1"]},
                                named_outputs={"neighbors": "neighbor_entities"}
                            )
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
    
    @pytest.mark.asyncio
    async def test_streaming_updates_basic(self, orchestrator, simple_plan):
        """Test that streaming updates are generated correctly"""
        # Mock the tool execution
        mock_output = EntityVDBSearchOutputs(
            similar_entities=[
                VDBSearchResultItem(entity_name="entity1", score=0.9),
                VDBSearchResultItem(entity_name="entity2", score=0.8)
            ]
        )
        
        with patch.object(orchestrator, '_execute_tool_async', return_value=(mock_output, None)):
            updates: List[StreamingUpdate] = []
            
            async for update in orchestrator.execute_plan_stream(simple_plan):
                updates.append(update)
            
            # Verify update sequence
            assert len(updates) >= 4  # plan_start, step_start, tool_start, tool_complete, step_complete, plan_complete
            
            # Check update types in order
            assert updates[0].type == UpdateType.PLAN_START
            assert updates[1].type == UpdateType.STEP_START
            assert any(u.type == UpdateType.TOOL_START for u in updates)
            assert any(u.type == UpdateType.TOOL_COMPLETE for u in updates)
            assert updates[-2].type == UpdateType.STEP_COMPLETE
            assert updates[-1].type == UpdateType.PLAN_COMPLETE
            
            # Verify timestamps
            for update in updates:
                assert isinstance(update.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, orchestrator, parallel_plan):
        """Test that read-only tools execute in parallel"""
        execution_order = []
        
        async def mock_execute_tool(tool_call, plan_inputs):
            tool_id = tool_call.tool_id
            execution_order.append(f"{tool_id}_start")
            
            # Simulate different execution times
            if tool_id == "Entity.VDBSearch":
                await asyncio.sleep(0.1)
            elif tool_id == "Entity.PPR":
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.02)
                
            execution_order.append(f"{tool_id}_end")
            return MagicMock(), None
        
        with patch.object(orchestrator, '_execute_tool_async', side_effect=mock_execute_tool):
            updates = []
            async for update in orchestrator.execute_plan_stream(parallel_plan):
                updates.append(update)
            
            # All tools should start before any complete (parallel execution)
            vdb_start_idx = execution_order.index("Entity.VDBSearch_start")
            ppr_start_idx = execution_order.index("Entity.PPR_start")
            onehop_start_idx = execution_order.index("Entity.Onehop_start")
            
            # Verify parallel execution - all start before any end
            first_end_idx = min(
                execution_order.index("Entity.VDBSearch_end"),
                execution_order.index("Entity.PPR_end"),
                execution_order.index("Entity.Onehop_end")
            )
            
            assert max(vdb_start_idx, ppr_start_idx, onehop_start_idx) < first_end_idx
    
    @pytest.mark.asyncio
    async def test_tool_categorization(self, orchestrator):
        """Test that tools are correctly categorized"""
        # Verify read-only tools
        assert orchestrator._tool_categories["Entity.VDBSearch"] == ToolCategory.READ_ONLY
        assert orchestrator._tool_categories["Entity.PPR"] == ToolCategory.READ_ONLY
        assert orchestrator._tool_categories["Chunk.GetTextForEntities"] == ToolCategory.READ_ONLY
        
        # Verify write tools
        assert orchestrator._tool_categories["Entity.VDB.Build"] == ToolCategory.WRITE
        assert orchestrator._tool_categories["corpus.PrepareFromDirectory"] == ToolCategory.WRITE
        
        # Verify build tools
        assert orchestrator._tool_categories["graph.BuildERGraph"] == ToolCategory.BUILD
        assert orchestrator._tool_categories["graph.BuildRKGraph"] == ToolCategory.BUILD
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator, simple_plan):
        """Test error handling during tool execution"""
        error_msg = "Test tool execution error"
        
        with patch.object(orchestrator, '_execute_tool_async', return_value=(None, error_msg)):
            updates = []
            
            async for update in orchestrator.execute_plan_stream(simple_plan):
                updates.append(update)
            
            # Find the tool error update
            error_updates = [u for u in updates if u.type == UpdateType.TOOL_ERROR]
            assert len(error_updates) == 1
            assert error_updates[0].error == error_msg
            
            # Plan should still complete
            assert updates[-1].type == UpdateType.PLAN_COMPLETE
    
    @pytest.mark.asyncio
    async def test_progress_tracking(self, orchestrator):
        """Test that progress is tracked correctly"""
        # Create a multi-step plan
        plan = ExecutionPlan(
            plan_id="test_progress",
            plan_description="Test progress tracking",
            steps=[
                ExecutionStep(
                    step_id=f"step{i}",
                    description=f"Step {i}",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDBSearch",
                                inputs={"query": f"query{i}"}
                            )
                        ]
                    )
                ) for i in range(1, 4)
            ],
            plan_inputs={}
        )
        
        with patch.object(orchestrator, '_execute_tool_async', return_value=(MagicMock(), None)):
            step_progress_values = []
            
            async for update in orchestrator.execute_plan_stream(plan):
                if update.type == UpdateType.STEP_START and update.progress is not None:
                    step_progress_values.append(update.progress)
            
            # Verify progress increases
            assert len(step_progress_values) == 3
            assert step_progress_values[0] < step_progress_values[1] < step_progress_values[2]
            assert 0 <= step_progress_values[0] <= 1
            assert 0 <= step_progress_values[2] <= 1
    
    @pytest.mark.asyncio
    async def test_output_storage(self, orchestrator, simple_plan):
        """Test that tool outputs are stored correctly"""
        mock_output = EntityVDBSearchOutputs(
            similar_entities=[
                VDBSearchResultItem(entity_name="entity1", score=0.9)
            ]
        )
        
        with patch.object(orchestrator, '_execute_tool_async', return_value=(mock_output, None)):
            async for _ in orchestrator.execute_plan_stream(simple_plan):
                pass
            
            # Verify outputs are stored
            assert "step1" in orchestrator.step_outputs
            assert "entities" in orchestrator.step_outputs["step1"]
            assert orchestrator.step_outputs["step1"]["entities"] == mock_output.similar_entities
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self, orchestrator, simple_plan):
        """Test the non-streaming execute_plan method"""
        mock_output = MagicMock()
        
        with patch.object(orchestrator, '_execute_tool_async', return_value=(mock_output, None)):
            result = await orchestrator.execute_plan(simple_plan)
            
            # Should return step outputs
            assert isinstance(result, dict)
            assert "step1" in result


@pytest.mark.asyncio
async def test_streaming_performance():
    """Test that streaming doesn't add significant overhead"""
    # This is a basic performance test
    import time
    
    # Create a simple mock orchestrator
    config = MagicMock()
    llm = MagicMock() 
    encoder = MagicMock()
    chunk_factory = MagicMock()
    context = MagicMock()
    
    orchestrator = AsyncStreamingOrchestrator(
        main_config=config,
        llm_instance=llm,
        encoder_instance=encoder,
        chunk_factory=chunk_factory,
        graphrag_context=context
    )
    
    # Mock fast tool execution
    async def fast_tool(*args, **kwargs):
        await asyncio.sleep(0.001)  # 1ms execution
        return MagicMock(), None
    
    with patch.object(orchestrator, '_execute_tool_async', side_effect=fast_tool):
        # Create plan with 10 tools
        plan = ExecutionPlan(
            plan_id="perf_test",
            plan_description="Performance test",
            steps=[
                ExecutionStep(
                    step_id="step1",
                    description="Multiple tools",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(tool_id="Entity.VDBSearch", inputs={"query": f"q{i}"})
                            for i in range(10)
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
        
        start_time = time.time()
        update_count = 0
        
        async for _ in orchestrator.execute_plan_stream(plan):
            update_count += 1
            
        elapsed = time.time() - start_time
        
        # Should complete quickly even with streaming
        assert elapsed < 1.0  # Less than 1 second for 10 tools
        assert update_count > 10  # Should have multiple updates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])