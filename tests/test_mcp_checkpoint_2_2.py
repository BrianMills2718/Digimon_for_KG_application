"""
Tests for Phase 2, Checkpoint 2.2: Parallel Execution Engine
"""

import asyncio
import pytest
from datetime import datetime
import time

from Core.AOT.aot_processor import AOTQueryProcessor, AtomicState
from Core.AOT.parallel_executor import (
    ParallelExecutor, ExecutionPlan, ExecutionResult, 
    ExecutionStatus
)


class TestParallelExecutor:
    """Test parallel execution of atomic states"""
    
    @pytest.fixture
    async def executor(self):
        """Create test executor"""
        return ParallelExecutor(max_concurrent=5)
    
    @pytest.fixture
    async def atomic_states(self):
        """Create test atomic states with dependencies"""
        states = [
            AtomicState(
                state_id="s1",
                content="French Revolution",
                state_type="entity",
                dependencies=set(),
                metadata={}
            ),
            AtomicState(
                state_id="s2",
                content="American Revolution",
                state_type="entity",
                dependencies=set(),
                metadata={}
            ),
            AtomicState(
                state_id="s3",
                content="causes",
                state_type="attribute",
                dependencies=set(),
                metadata={}
            ),
            AtomicState(
                state_id="s4",
                content="comparison",
                state_type="relationship",
                dependencies={"s1", "s2"},  # Depends on both entities
                metadata={}
            ),
            AtomicState(
                state_id="s5",
                content="retrieve_information",
                state_type="action",
                dependencies={"s1", "s2", "s3"},  # Depends on entities and attribute
                metadata={}
            )
        ]
        return states
    
    async def test_dependency_analysis(self, executor, atomic_states):
        """Test: Dependencies correctly analyzed"""
        plan = await executor.analyze_dependencies(atomic_states)
        
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.execution_levels) > 0
        
        # First level should have independent states (s1, s2, s3)
        first_level = set(plan.execution_levels[0])
        assert "s1" in first_level
        assert "s2" in first_level
        assert "s3" in first_level
        
        # s4 depends on s1 and s2, should be in later level
        for level in plan.execution_levels[1:]:
            if "s4" in level:
                break
        else:
            assert False, "s4 not found in any level after first"
        
        # s5 should be after s1, s2, s3
        s5_level_idx = None
        for i, level in enumerate(plan.execution_levels):
            if "s5" in level:
                s5_level_idx = i
                break
        
        assert s5_level_idx is not None
        assert s5_level_idx > 0  # Not in first level
    
    async def test_parallel_execution(self, executor, atomic_states):
        """Test: Independent tools run in parallel"""
        plan = await executor.analyze_dependencies(atomic_states)
        
        # Track execution timing
        start_time = time.time()
        results = await executor.execute_plan(plan)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000  # ms
        
        # All states should have results
        assert len(results) == len(atomic_states)
        
        # All should be successful (mock execution)
        for result in results.values():
            assert result.status == ExecutionStatus.COMPLETED
        
        # Check that first level states ran in parallel
        first_level_results = [results[sid] for sid in plan.execution_levels[0]]
        
        # If they ran in parallel, total time should be less than sum of individual times
        individual_times = sum(r.duration_ms for r in first_level_results)
        
        # With parallel execution, total time should be significantly less
        # (accounting for some overhead)
        assert total_time < individual_times * 0.8
    
    async def test_dependency_respect(self, executor, atomic_states):
        """Test: Dependencies respected during execution"""
        plan = await executor.analyze_dependencies(atomic_states)
        results = await executor.execute_plan(plan)
        
        # s4 should execute after s1 and s2
        s1_end = results["s1"].end_time
        s2_end = results["s2"].end_time
        s4_start = results["s4"].start_time
        
        assert s4_start >= s1_end
        assert s4_start >= s2_end
        
        # s5 should execute after s1, s2, and s3
        s3_end = results["s3"].end_time
        s5_start = results["s5"].start_time
        
        assert s5_start >= s1_end
        assert s5_start >= s2_end
        assert s5_start >= s3_end
    
    async def test_result_aggregation(self, executor, atomic_states):
        """Test: Results correctly aggregated"""
        plan = await executor.analyze_dependencies(atomic_states)
        results = await executor.execute_plan(plan)
        
        # Check all results are present
        assert set(results.keys()) == {s.state_id for s in atomic_states}
        
        # Check result structure
        for state_id, result in results.items():
            assert isinstance(result, ExecutionResult)
            assert result.state_id == state_id
            assert result.status in ExecutionStatus
            assert result.duration_ms >= 0
            assert result.start_time <= result.end_time
            
            # Mock execution should produce results
            if result.status == ExecutionStatus.COMPLETED:
                assert result.result is not None
    
    async def test_progress_tracking(self, executor, atomic_states):
        """Test: Real-time progress updates"""
        plan = await executor.analyze_dependencies(atomic_states)
        
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress)
        
        executor.add_progress_callback(progress_callback)
        
        await executor.execute_plan(plan)
        
        # Should have received progress updates
        assert len(progress_updates) > 0
        
        # Progress should increase
        percentages = [p["percentage"] for p in progress_updates]
        assert percentages[-1] == 100.0
        
        # Check progress structure
        for progress in progress_updates:
            assert "completed" in progress
            assert "total" in progress
            assert "percentage" in progress
            assert "current_state" in progress
            assert progress["completed"] <= progress["total"]
    
    async def test_failure_handling(self, executor):
        """Test: Graceful handling of execution failures"""
        # Create a state that will fail
        failing_state = AtomicState(
            state_id="fail",
            content="failing_state",
            state_type="unknown",  # Unknown type will cause default execution
            dependencies=set(),
            metadata={}
        )
        
        # Register a failing executor
        async def failing_executor(state, deps):
            raise ValueError("Simulated execution failure")
        
        executor.register_state_executor("unknown", failing_executor)
        
        plan = await executor.analyze_dependencies([failing_state])
        results = await executor.execute_plan(plan)
        
        assert results["fail"].status == ExecutionStatus.FAILED
        assert results["fail"].error is not None
        assert "Simulated execution failure" in results["fail"].error
    
    async def test_execution_stats(self, executor, atomic_states):
        """Test: Execution statistics calculation"""
        plan = await executor.analyze_dependencies(atomic_states)
        await executor.execute_plan(plan)
        
        stats = executor.get_execution_stats()
        
        assert stats["total_executions"] == len(atomic_states)
        assert stats["successful_executions"] == len(atomic_states)
        assert stats["failed_executions"] == 0
        assert stats["avg_duration_ms"] > 0
        assert stats["success_rate"] == 1.0
        assert 0 <= stats["parallel_efficiency"] <= 1.0
    
    async def test_circular_dependency_detection(self, executor):
        """Test: Circular dependencies detected"""
        # Create states with circular dependency
        states = [
            AtomicState("s1", "state1", "entity", {"s2"}, {}),
            AtomicState("s2", "state2", "entity", {"s1"}, {}),
        ]
        
        plan = await executor.analyze_dependencies(states)
        
        # Should still create a plan, but log error
        assert len(plan.execution_levels) > 0
        
        # Both states should be in the same level (forced)
        assert len(plan.execution_levels[-1]) == 2
    
    async def test_custom_executor_registration(self, executor):
        """Test: Custom executors for state types"""
        custom_result = {"custom": True, "data": "test"}
        
        async def custom_entity_executor(state, deps):
            return custom_result
        
        executor.register_state_executor("entity", custom_entity_executor)
        
        state = AtomicState("s1", "test", "entity", set(), {})
        plan = await executor.analyze_dependencies([state])
        results = await executor.execute_plan(plan)
        
        assert results["s1"].status == ExecutionStatus.COMPLETED
        assert results["s1"].result == custom_result

@pytest.mark.asyncio
async def test_end_to_end_parallel_execution():
    """Integration test of AOT + Parallel Execution"""
    # Create components
    processor = AOTQueryProcessor()
    executor = ParallelExecutor(max_concurrent=10)
    
    # Complex query
    query = ("What were the main economic causes of the French Revolution "
            "and how did they compare to the social causes of the American Revolution?")
    
    # Decompose
    atomic_states = await processor.decompose_query(query)
    
    # Create execution plan
    plan = await executor.analyze_dependencies(atomic_states)
    
    # Verify plan structure
    assert len(plan.execution_levels) >= 2  # Should have multiple levels
    
    # Execute with progress tracking
    progress_count = 0
    
    async def progress_callback(progress):
        nonlocal progress_count
        progress_count += 1
    
    executor.add_progress_callback(progress_callback)
    
    # Execute plan
    results = await executor.execute_plan(plan)
    
    # Verify results
    assert len(results) == len(atomic_states)
    assert all(r.status == ExecutionStatus.COMPLETED for r in results.values())
    assert progress_count > 0
    
    # Check performance
    stats = executor.get_execution_stats()
    assert stats["success_rate"] == 1.0
    assert stats["parallel_efficiency"] > 0.5  # Should have decent parallelism


if __name__ == "__main__":
    asyncio.run(test_end_to_end_parallel_execution())