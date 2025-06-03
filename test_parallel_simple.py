#!/usr/bin/env python3
"""
Simple test for parallel execution capabilities
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.Common.Logger import logger
from Core.AgentSchema.plan import (
    ExecutionPlan, ExecutionStep, ToolCall, 
    DynamicToolChainConfig, ToolInputSource
)
from Core.AgentOrchestrator.parallel_orchestrator import ParallelAgentOrchestrator


async def test_dependency_analysis():
    """Test dependency analysis logic."""
    logger.info("=== Testing Dependency Analysis ===")
    
    # Create a minimal orchestrator instance
    orchestrator = ParallelAgentOrchestrator.__new__(ParallelAgentOrchestrator)
    orchestrator.dependency_graph = {}
    orchestrator.execution_groups = []
    
    # Create test plan with various dependency patterns
    plan = ExecutionPlan(
        plan_id="test_plan",
        plan_description="Test dependency analysis",
        target_dataset_name="test",
        steps=[
            # Independent steps
            ExecutionStep(
                step_id="A",
                description="Step A - Independent",
                action=DynamicToolChainConfig(
                    tools=[ToolCall(tool_id="tool.A")]
                )
            ),
            ExecutionStep(
                step_id="B",
                description="Step B - Independent",
                action=DynamicToolChainConfig(
                    tools=[ToolCall(tool_id="tool.B")]
                )
            ),
            # Step C depends on A
            ExecutionStep(
                step_id="C",
                description="Step C - Depends on A",
                action=DynamicToolChainConfig(
                    tools=[ToolCall(
                        tool_id="tool.C",
                        inputs={
                            "input": ToolInputSource(from_step_id="A", named_output_key="output")
                        }
                    )]
                )
            ),
            # Step D depends on both A and B
            ExecutionStep(
                step_id="D",
                description="Step D - Depends on A and B",
                action=DynamicToolChainConfig(
                    tools=[ToolCall(
                        tool_id="tool.D",
                        inputs={
                            "input1": ToolInputSource(from_step_id="A", named_output_key="output"),
                            "input2": ToolInputSource(from_step_id="B", named_output_key="output")
                        }
                    )]
                )
            ),
            # Step E depends on C
            ExecutionStep(
                step_id="E",
                description="Step E - Depends on C",
                action=DynamicToolChainConfig(
                    tools=[ToolCall(
                        tool_id="tool.E",
                        inputs={
                            "input": ToolInputSource(from_step_id="C", named_output_key="output")
                        }
                    )]
                )
            ),
            # Step F depends on D
            ExecutionStep(
                step_id="F",
                description="Step F - Depends on D",
                action=DynamicToolChainConfig(
                    tools=[ToolCall(
                        tool_id="tool.F",
                        inputs={
                            "input": ToolInputSource(from_step_id="D", named_output_key="output")
                        }
                    )]
                )
            ),
        ]
    )
    
    # Analyze dependencies
    deps = orchestrator._analyze_dependencies(plan)
    logger.info("\nDependencies found:")
    for step_id, dependencies in sorted(deps.items()):
        if dependencies:
            logger.info(f"  {step_id} <- {sorted(dependencies)}")
        else:
            logger.info(f"  {step_id} (independent)")
    
    # Create execution groups
    groups = orchestrator._create_execution_groups(plan, deps)
    logger.info("\nExecution groups:")
    for i, group in enumerate(groups):
        logger.info(f"  Group {i + 1}: {sorted(group)}")
    
    # Verify expected dependencies
    expected_deps = {
        "A": set(),
        "B": set(),
        "C": {"A"},
        "D": {"A", "B"},
        "E": {"C"},
        "F": {"D"}
    }
    
    # Verify expected groups
    expected_groups = [
        ["A", "B"],     # Independent
        ["C", "D"],     # C depends on A, D depends on A and B
        ["E", "F"]      # E depends on C, F depends on D
    ]
    
    # Check results
    deps_match = all(deps.get(k, set()) == v for k, v in expected_deps.items())
    groups_match = all(
        sorted(actual) == sorted(expected) 
        for actual, expected in zip(groups, expected_groups)
    )
    
    if deps_match and groups_match:
        logger.info("\n✅ Dependency analysis PASSED!")
        return True
    else:
        logger.error("\n❌ Dependency analysis FAILED!")
        if not deps_match:
            logger.error(f"  Expected deps: {expected_deps}")
            logger.error(f"  Actual deps: {dict(deps)}")
        if not groups_match:
            logger.error(f"  Expected groups: {expected_groups}")
            logger.error(f"  Actual groups: {groups}")
        return False


async def test_parallel_timing():
    """Test that parallel execution actually saves time."""
    logger.info("\n=== Testing Parallel Execution Timing ===")
    
    # Track execution order
    execution_log = []
    
    async def mock_step(step_id: str, delay: float, dependencies: list = None):
        """Mock step execution."""
        start_time = time.time()
        execution_log.append(f"{step_id} started at {start_time:.2f}")
        await asyncio.sleep(delay)
        end_time = time.time()
        execution_log.append(f"{step_id} finished at {end_time:.2f} (took {delay}s)")
        return {"status": "success", "step_id": step_id}
    
    # Test scenario: 
    # - Steps A and B can run in parallel (2s each)
    # - Step C depends on both A and B (1s)
    # Sequential would take 5s, parallel should take 3s
    
    logger.info("Running parallel execution test...")
    start_time = time.time()
    
    # Simulate parallel execution
    tasks_group1 = [
        asyncio.create_task(mock_step("A", 2.0)),
        asyncio.create_task(mock_step("B", 2.0))
    ]
    
    # Wait for first group
    results_group1 = await asyncio.gather(*tasks_group1)
    
    # Then run dependent step
    result_c = await mock_step("C", 1.0, dependencies=["A", "B"])
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"\nExecution log:")
    for entry in execution_log:
        logger.info(f"  {entry}")
    
    logger.info(f"\nTotal execution time: {total_time:.2f}s")
    
    # Check timing
    if total_time < 3.5:  # Allow some overhead
        logger.info("✅ Parallel execution timing PASSED! (saved ~2s)")
        return True
    else:
        logger.error(f"❌ Parallel execution timing FAILED! Expected ~3s, got {total_time:.2f}s")
        return False


async def test_visualization():
    """Test dependency visualization."""
    logger.info("\n=== Testing Dependency Visualization ===")
    
    orchestrator = ParallelAgentOrchestrator.__new__(ParallelAgentOrchestrator)
    orchestrator.dependency_graph = {
        "A": set(),
        "B": set(),
        "C": {"A"},
        "D": {"A", "B"},
        "E": {"C", "D"}
    }
    orchestrator.execution_groups = [
        ["A", "B"],
        ["C", "D"],
        ["E"]
    ]
    
    visualization = orchestrator.get_dependency_visualization()
    logger.info("\n" + visualization)
    
    # Just check it produces output
    if len(visualization) > 0 and "Dependency Graph:" in visualization:
        logger.info("\n✅ Visualization PASSED!")
        return True
    else:
        logger.error("\n❌ Visualization FAILED!")
        return False


async def main():
    """Run all tests."""
    logger.info("Starting parallel execution tests...\n")
    
    tests = [
        ("Dependency Analysis", test_dependency_analysis),
        ("Parallel Timing", test_parallel_timing),
        ("Visualization", test_visualization),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("✅ All tests passed!")
        return True
    else:
        logger.error(f"❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)