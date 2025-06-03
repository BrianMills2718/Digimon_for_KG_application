# Core/AgentOrchestrator/parallel_orchestrator.py

import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

from Core.AgentOrchestrator.enhanced_orchestrator import EnhancedAgentOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, ToolInputSource
from Core.Common.Logger import logger
from Core.Common.PerformanceMonitor import PerformanceMonitor


class ParallelAgentOrchestrator(EnhancedAgentOrchestrator):
    """Orchestrator with parallel execution capabilities for independent steps."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.execution_groups: List[List[str]] = []
        
    def _analyze_dependencies(self, plan: ExecutionPlan) -> Dict[str, Set[str]]:
        """
        Analyze step dependencies to build a dependency graph.
        Returns a dict mapping step_id -> set of step_ids it depends on.
        """
        dependencies = defaultdict(set)
        
        for step in plan.steps:
            step_id = step.step_id
            
            # Analyze tool calls in the step
            if hasattr(step.action, 'tools'):
                for tool_call in step.action.tools:
                    # Check inputs for dependencies
                    if tool_call.inputs:
                        for input_name, input_source in tool_call.inputs.items():
                            if isinstance(input_source, ToolInputSource):
                                dependencies[step_id].add(input_source.from_step_id)
                            elif isinstance(input_source, dict) and "from_step_id" in input_source:
                                dependencies[step_id].add(input_source["from_step_id"])
                    
                    # Check parameters for dependencies (less common but possible)
                    if tool_call.parameters:
                        for param_name, param_value in tool_call.parameters.items():
                            if isinstance(param_value, ToolInputSource):
                                dependencies[step_id].add(param_value.from_step_id)
                            elif isinstance(param_value, dict) and "from_step_id" in param_value:
                                dependencies[step_id].add(param_value["from_step_id"])
        
        return dict(dependencies)
    
    def _create_execution_groups(self, plan: ExecutionPlan, dependencies: Dict[str, Set[str]]) -> List[List[str]]:
        """
        Create groups of steps that can be executed in parallel.
        Uses topological sorting with level grouping.
        """
        # Build reverse dependency graph (step -> steps that depend on it)
        reverse_deps = defaultdict(set)
        all_steps = {step.step_id for step in plan.steps}
        
        for step_id, deps in dependencies.items():
            for dep in deps:
                reverse_deps[dep].add(step_id)
        
        # Find steps with no dependencies (can start immediately)
        in_degree = {step_id: len(dependencies.get(step_id, set())) for step_id in all_steps}
        
        # Group steps by execution level
        groups = []
        remaining_steps = all_steps.copy()
        
        while remaining_steps:
            # Find all steps that can be executed now (no pending dependencies)
            current_group = []
            for step_id in remaining_steps:
                if in_degree[step_id] == 0:
                    current_group.append(step_id)
            
            if not current_group:
                # Circular dependency detected
                logger.error(f"Circular dependency detected! Remaining steps: {remaining_steps}")
                # Fall back to sequential execution for remaining steps
                current_group = list(remaining_steps)
                remaining_steps.clear()
            else:
                # Remove executed steps and update in-degrees
                for step_id in current_group:
                    remaining_steps.remove(step_id)
                    # Decrease in-degree for dependent steps
                    for dependent in reverse_deps[step_id]:
                        in_degree[dependent] -= 1
            
            groups.append(current_group)
        
        return groups
    
    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute plan with parallel execution of independent steps."""
        logger.info(f"Parallel Orchestrator: Starting execution of plan ID: {plan.plan_id}")
        
        # Analyze dependencies
        self.dependency_graph = self._analyze_dependencies(plan)
        self.execution_groups = self._create_execution_groups(plan, self.dependency_graph)
        
        # Log execution strategy
        logger.info(f"Parallel Orchestrator: Identified {len(self.execution_groups)} execution groups")
        for i, group in enumerate(self.execution_groups):
            logger.info(f"  Group {i + 1}: {', '.join(group)} ({len(group)} steps)")
        
        # Monitor overall plan execution
        with self.performance_monitor.measure_operation("parallel_plan_execution"):
            self.step_outputs: Dict[str, Dict[str, Any]] = {}
            
            # Create step lookup
            step_lookup = {step.step_id: step for step in plan.steps}
            
            # Execute groups in order
            for group_index, group_step_ids in enumerate(self.execution_groups):
                logger.info(f"Parallel Orchestrator: Executing group {group_index + 1}/{len(self.execution_groups)} with {len(group_step_ids)} parallel steps")
                
                # Monitor group execution
                with self.performance_monitor.measure_operation(f"group_{group_index + 1}_parallel"):
                    if len(group_step_ids) == 1:
                        # Single step - execute normally
                        step_id = group_step_ids[0]
                        step = step_lookup[step_id]
                        await self._execute_step_with_error_handling(step, plan)
                    else:
                        # Multiple steps - execute in parallel
                        tasks = []
                        for step_id in group_step_ids:
                            step = step_lookup[step_id]
                            task = asyncio.create_task(
                                self._execute_step_with_error_handling(step, plan),
                                name=f"step_{step_id}"
                            )
                            tasks.append(task)
                        
                        # Wait for all tasks in the group to complete
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Log results
                        for i, (step_id, result) in enumerate(zip(group_step_ids, results)):
                            if isinstance(result, Exception):
                                logger.error(f"Step {step_id} failed with exception: {result}")
                            else:
                                logger.info(f"Step {step_id} completed successfully")
        
        # Log performance summary
        summary = self.performance_monitor.get_summary()
        logger.info(f"Parallel plan execution complete. Performance summary:")
        
        # Calculate speedup from parallelization
        if "parallel_plan_execution" in summary:
            total_time = summary["parallel_plan_execution"]["duration"]["total"]
            sequential_time = sum(
                summary.get(f"step_{step.step_id}", {}).get("duration", {}).get("total", 0)
                for step in plan.steps
            )
            if sequential_time > 0:
                speedup = sequential_time / total_time
                logger.info(f"  Parallelization speedup: {speedup:.2f}x (sequential: {sequential_time:.1f}s, parallel: {total_time:.1f}s)")
        
        return self.step_outputs
    
    async def _execute_step_with_error_handling(self, step: ExecutionStep, plan: ExecutionPlan):
        """Execute a step with proper error handling and monitoring."""
        try:
            # Monitor individual step
            with self.performance_monitor.measure_operation(f"step_{step.step_id}"):
                await self._execute_step(step, plan)
        except Exception as e:
            logger.error(f"Error in step {step.step_id}: {e}")
            self.step_outputs[step.step_id] = {"error": str(e)}
    
    def get_dependency_visualization(self) -> str:
        """Get a text visualization of the dependency graph."""
        lines = ["Dependency Graph:"]
        for step_id, deps in self.dependency_graph.items():
            if deps:
                lines.append(f"  {step_id} <- {', '.join(deps)}")
            else:
                lines.append(f"  {step_id} (no dependencies)")
        
        lines.append("\nExecution Groups:")
        for i, group in enumerate(self.execution_groups):
            lines.append(f"  Group {i + 1}: [{', '.join(group)}]")
        
        return "\n".join(lines)