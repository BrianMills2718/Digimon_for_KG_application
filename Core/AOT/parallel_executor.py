"""
Parallel Execution Engine for AOT
Executes independent atomic states concurrently with dependency resolution
"""

import asyncio
import logging
from typing import List, Dict, Any, Set, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
from collections import defaultdict
from enum import Enum

from Core.AOT.aot_processor import AtomicState
from Core.Common.Logger import logger


class ExecutionStatus(Enum):
    """Status of atomic state execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionResult:
    """Result of executing an atomic state"""
    state_id: str
    status: ExecutionStatus
    result: Any
    start_time: datetime
    end_time: datetime
    duration_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "state_id": self.state_id,
            "status": self.status.value,
            "result": self.result,
            "duration_ms": self.duration_ms,
            "error": self.error
        }


@dataclass
class ExecutionPlan:
    """Execution plan with dependency analysis"""
    atomic_states: List[AtomicState]
    dependency_graph: Dict[str, Set[str]]  # state_id -> dependencies
    execution_levels: List[List[str]]  # Parallel execution levels
    estimated_duration_ms: float = 0.0


class ParallelExecutor:
    """
    Executes atomic states in parallel while respecting dependencies
    """
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.execution_history: List[ExecutionResult] = []
        self.progress_callbacks: List[Callable] = []
        self.state_executors: Dict[str, Callable] = {}  # Maps state_type to executor
        
    def register_state_executor(self, state_type: str, executor: Callable):
        """Register an executor function for a state type"""
        self.state_executors[state_type] = executor
        
    def add_progress_callback(self, callback: Callable):
        """Add a callback for progress updates"""
        self.progress_callbacks.append(callback)
        
    async def analyze_dependencies(self, atomic_states: List[AtomicState]) -> ExecutionPlan:
        """
        Analyze dependencies and create execution plan
        
        Returns:
            ExecutionPlan with dependency graph and execution levels
        """
        logger.info(f"Analyzing dependencies for {len(atomic_states)} states")
        
        # Build dependency graph
        dependency_graph = {}
        state_map = {state.state_id: state for state in atomic_states}
        
        for state in atomic_states:
            dependency_graph[state.state_id] = state.dependencies.copy()
        
        # Topological sort to find execution levels
        execution_levels = []
        remaining_states = set(state.state_id for state in atomic_states)
        completed_states = set()
        
        while remaining_states:
            # Find states with no pending dependencies
            current_level = []
            for state_id in remaining_states:
                deps = dependency_graph[state_id]
                if deps.issubset(completed_states):
                    current_level.append(state_id)
            
            if not current_level:
                # Circular dependency detected
                logger.error(f"Circular dependency detected among states: {remaining_states}")
                # Add remaining states as final level (will likely fail)
                current_level = list(remaining_states)
            
            execution_levels.append(current_level)
            completed_states.update(current_level)
            remaining_states.difference_update(current_level)
        
        # Estimate duration (simplified: assume 50ms per state, parallel within levels)
        estimated_duration = len(execution_levels) * 50.0
        
        plan = ExecutionPlan(
            atomic_states=atomic_states,
            dependency_graph=dependency_graph,
            execution_levels=execution_levels,
            estimated_duration_ms=estimated_duration
        )
        
        logger.info(f"Execution plan created with {len(execution_levels)} levels")
        return plan
    
    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, ExecutionResult]:
        """
        Execute the plan with parallel execution of independent states
        
        Returns:
            Dictionary mapping state_id to ExecutionResult
        """
        logger.info(f"Executing plan with {len(plan.execution_levels)} levels")
        
        results = {}
        state_map = {state.state_id: state for state in plan.atomic_states}
        
        total_states = len(plan.atomic_states)
        completed_states = 0
        
        for level_idx, level_states in enumerate(plan.execution_levels):
            logger.info(f"Executing level {level_idx + 1}/{len(plan.execution_levels)} "
                       f"with {len(level_states)} parallel states")
            
            # Execute states in this level concurrently
            level_tasks = []
            for state_id in level_states:
                state = state_map[state_id]
                # Get dependency results
                dep_results = {dep_id: results.get(dep_id) 
                             for dep_id in state.dependencies}
                
                task = self._execute_state(state, dep_results)
                level_tasks.append(task)
            
            # Wait for all states in this level to complete
            level_results = await asyncio.gather(*level_tasks, return_exceptions=True)
            
            # Store results
            for state_id, result in zip(level_states, level_results):
                if isinstance(result, Exception):
                    # Handle execution failure
                    results[state_id] = ExecutionResult(
                        state_id=state_id,
                        status=ExecutionStatus.FAILED,
                        result=None,
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        duration_ms=0.0,
                        error=str(result)
                    )
                else:
                    results[state_id] = result
                
                completed_states += 1
                await self._report_progress(completed_states, total_states, state_id)
        
        logger.info(f"Execution complete. {len(results)} states executed")
        return results
    
    async def _execute_state(self, state: AtomicState, 
                           dependency_results: Dict[str, ExecutionResult]) -> ExecutionResult:
        """Execute a single atomic state"""
        start_time = datetime.utcnow()
        
        try:
            # Check if we have an executor for this state type
            if state.state_type in self.state_executors:
                executor = self.state_executors[state.state_type]
                # Pass state and dependency results to executor
                result = await executor(state, dependency_results)
            else:
                # Default mock execution
                result = await self._mock_execute_state(state, dependency_results)
            
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            execution_result = ExecutionResult(
                state_id=state.state_id,
                status=ExecutionStatus.COMPLETED,
                result=result,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms
            )
            
            self.execution_history.append(execution_result)
            return execution_result
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            execution_result = ExecutionResult(
                state_id=state.state_id,
                status=ExecutionStatus.FAILED,
                result=None,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                error=str(e)
            )
            
            self.execution_history.append(execution_result)
            return execution_result
    
    async def _mock_execute_state(self, state: AtomicState, 
                                dependency_results: Dict[str, ExecutionResult]) -> Dict[str, Any]:
        """Mock execution for testing"""
        # Simulate some work
        await asyncio.sleep(0.01)  # 10ms
        
        # Generate mock result based on state type
        if state.state_type == "entity":
            return {
                "found": True,
                "data": f"Information about {state.content}",
                "confidence": 0.9
            }
        elif state.state_type == "relationship":
            return {
                "exists": True,
                "strength": 0.8,
                "evidence": f"Evidence for {state.content}"
            }
        elif state.state_type == "attribute":
            return {
                "applied": True,
                "filter": state.content,
                "matches": 10
            }
        elif state.state_type == "action":
            return {
                "executed": True,
                "action": state.content,
                "results": "Action completed successfully"
            }
        else:
            return {"status": "completed"}
    
    async def _report_progress(self, completed: int, total: int, current_state: str):
        """Report execution progress"""
        progress = {
            "completed": completed,
            "total": total,
            "percentage": (completed / total) * 100 if total > 0 else 0,
            "current_state": current_state,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Call all registered callbacks
        for callback in self.progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress)
                else:
                    callback(progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "avg_duration_ms": 0,
                "success_rate": 0,
                "parallel_efficiency": 0
            }
        
        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history 
                        if r.status == ExecutionStatus.COMPLETED)
        
        durations = [r.duration_ms for r in self.execution_history]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate parallel efficiency
        # (ideal time / actual time) where ideal = sum of all durations / max concurrent
        total_work_ms = sum(durations)
        actual_time_ms = max((r.end_time - r.start_time).total_seconds() * 1000 
                           for r in self.execution_history) if self.execution_history else 0
        ideal_time_ms = total_work_ms / self.max_concurrent
        parallel_efficiency = ideal_time_ms / actual_time_ms if actual_time_ms > 0 else 0
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": total - successful,
            "avg_duration_ms": avg_duration,
            "total_duration_ms": actual_time_ms,
            "success_rate": successful / total if total > 0 else 0,
            "parallel_efficiency": min(parallel_efficiency, 1.0)  # Cap at 100%
        }
    
    async def execute_with_timeout(self, plan: ExecutionPlan, 
                                 timeout_seconds: float) -> Dict[str, ExecutionResult]:
        """Execute plan with timeout"""
        try:
            return await asyncio.wait_for(
                self.execute_plan(plan),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Execution timed out after {timeout_seconds} seconds")
            # Return partial results
            return {r.state_id: r for r in self.execution_history}


# Example usage for testing
async def example_parallel_execution():
    from Core.AOT.aot_processor import AOTQueryProcessor
    
    # Create processor and executor
    processor = AOTQueryProcessor()
    executor = ParallelExecutor(max_concurrent=5)
    
    # Complex query
    query = "Compare the causes and effects of the French Revolution with the American Revolution"
    
    # Decompose into atomic states
    atomic_states = await processor.decompose_query(query)
    
    # Analyze dependencies
    plan = await executor.analyze_dependencies(atomic_states)
    
    print(f"Execution plan has {len(plan.execution_levels)} levels:")
    for i, level in enumerate(plan.execution_levels):
        print(f"  Level {i+1}: {len(level)} states can run in parallel")
    
    # Add progress callback
    def progress_callback(progress):
        print(f"Progress: {progress['percentage']:.1f}% - Current: {progress['current_state']}")
    
    executor.add_progress_callback(progress_callback)
    
    # Execute plan
    results = await executor.execute_plan(plan)
    
    # Get statistics
    stats = executor.get_execution_stats()
    print(f"\nExecution Statistics:")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Average duration: {stats['avg_duration_ms']:.1f}ms")
    print(f"  Parallel efficiency: {stats['parallel_efficiency']:.1%}")


if __name__ == "__main__":
    asyncio.run(example_parallel_execution())