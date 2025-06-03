#!/usr/bin/env python3
"""
Integration test for parallel orchestrator with the DIGIMON system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.Common.Logger import logger
from Core.AgentOrchestrator.parallel_orchestrator import ParallelAgentOrchestrator
from Core.AgentOrchestrator.enhanced_orchestrator import EnhancedAgentOrchestrator
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.plan import (
    ExecutionPlan, ExecutionStep, ToolCall, 
    DynamicToolChainConfig, ToolInputSource
)
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory
from Option.Config2 import default_config


async def test_orchestrator_compatibility():
    """Test that ParallelAgentOrchestrator is compatible with existing code."""
    logger.info("=== Testing Orchestrator Compatibility ===")
    
    try:
        config = default_config
        
        # Create minimal dependencies
        class MockLLM:
            _api_key = "test"
            _model_name = "gpt-3.5-turbo"
            _temperature = 0.7
        
        context = GraphRAGContext(
            main_config=config,
            embedding_provider=None,
            target_dataset_name="test"
        )
        
        # Test creating parallel orchestrator
        orchestrator = ParallelAgentOrchestrator(
            main_config=config,
            llm_instance=MockLLM(),
            encoder_instance=None,
            chunk_factory=None,
            graphrag_context=context
        )
        
        # Verify it has all necessary attributes
        assert hasattr(orchestrator, '_tool_registry'), "Missing _tool_registry"
        assert hasattr(orchestrator, 'execute_plan'), "Missing execute_plan method"
        assert hasattr(orchestrator, '_analyze_dependencies'), "Missing _analyze_dependencies"
        assert hasattr(orchestrator, '_create_execution_groups'), "Missing _create_execution_groups"
        assert hasattr(orchestrator, 'performance_monitor'), "Missing performance_monitor"
        assert hasattr(orchestrator, 'enhanced_llm'), "Missing enhanced_llm"
        
        logger.info("✅ Orchestrator creation and attributes check passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_parallel_vs_sequential():
    """Compare parallel vs sequential execution performance."""
    logger.info("\n=== Testing Parallel vs Sequential Performance ===")
    
    try:
        config = default_config
        
        # Create mock components
        class MockLLM:
            _api_key = "test"
            _model_name = "gpt-3.5-turbo"
            _temperature = 0.7
        
        context = GraphRAGContext(
            main_config=config,
            embedding_provider=None,
            target_dataset_name="test"
        )
        
        # Create both orchestrators
        enhanced_orchestrator = EnhancedAgentOrchestrator(
            main_config=config,
            llm_instance=MockLLM(),
            encoder_instance=None,
            chunk_factory=None,
            graphrag_context=context
        )
        
        parallel_orchestrator = ParallelAgentOrchestrator(
            main_config=config,
            llm_instance=MockLLM(),
            encoder_instance=None,
            chunk_factory=None,
            graphrag_context=context
        )
        
        # Create mock tools
        async def mock_slow_tool(params, context):
            """Mock tool that takes 1 second."""
            import time
            await asyncio.sleep(1.0)
            return {"status": "success", "tool_id": getattr(params, 'tool_id', 'unknown')}
        
        class MockParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        # Register mock tools in both orchestrators
        for orchestrator in [enhanced_orchestrator, parallel_orchestrator]:
            orchestrator._tool_registry["mock.SlowTool1"] = (mock_slow_tool, MockParams)
            orchestrator._tool_registry["mock.SlowTool2"] = (mock_slow_tool, MockParams)
            orchestrator._tool_registry["mock.SlowTool3"] = (mock_slow_tool, MockParams)
        
        # Create a plan with independent steps
        plan = ExecutionPlan(
            plan_id="perf_test",
            plan_description="Performance comparison test",
            target_dataset_name="test",
            steps=[
                # Two independent steps
                ExecutionStep(
                    step_id="independent1",
                    description="First independent step",
                    action=DynamicToolChainConfig(
                        tools=[ToolCall(
                            tool_id="mock.SlowTool1",
                            parameters={"tool_id": "tool1"}
                        )]
                    )
                ),
                ExecutionStep(
                    step_id="independent2",
                    description="Second independent step",
                    action=DynamicToolChainConfig(
                        tools=[ToolCall(
                            tool_id="mock.SlowTool2",
                            parameters={"tool_id": "tool2"}
                        )]
                    )
                ),
                # Dependent step
                ExecutionStep(
                    step_id="dependent",
                    description="Dependent step",
                    action=DynamicToolChainConfig(
                        tools=[ToolCall(
                            tool_id="mock.SlowTool3",
                            inputs={
                                "input1": ToolInputSource(
                                    from_step_id="independent1",
                                    named_output_key="mock.SlowTool1"
                                )
                            },
                            parameters={"tool_id": "tool3"}
                        )]
                    )
                )
            ]
        )
        
        # Execute with enhanced (sequential) orchestrator
        logger.info("\nExecuting with EnhancedAgentOrchestrator (sequential)...")
        import time
        start_sequential = time.time()
        await enhanced_orchestrator.execute_plan(plan)
        sequential_time = time.time() - start_sequential
        
        # Execute with parallel orchestrator
        logger.info("\nExecuting with ParallelAgentOrchestrator...")
        start_parallel = time.time()
        await parallel_orchestrator.execute_plan(plan)
        parallel_time = time.time() - start_parallel
        
        # Compare results
        logger.info(f"\nPerformance comparison:")
        logger.info(f"  Sequential execution: {sequential_time:.2f}s")
        logger.info(f"  Parallel execution: {parallel_time:.2f}s")
        logger.info(f"  Speedup: {sequential_time/parallel_time:.2f}x")
        
        # Parallel should be significantly faster (around 2s vs 3s)
        if parallel_time < sequential_time * 0.8:  # At least 20% faster
            logger.info("✅ Parallel execution is faster as expected")
            return True
        else:
            logger.error("❌ Parallel execution not significantly faster")
            return False
            
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_complex_dependencies():
    """Test handling of complex dependency patterns."""
    logger.info("\n=== Testing Complex Dependencies ===")
    
    orchestrator = ParallelAgentOrchestrator.__new__(ParallelAgentOrchestrator)
    orchestrator.dependency_graph = {}
    orchestrator.execution_groups = []
    
    # Create a plan with complex dependencies
    # A, B independent
    # C depends on A
    # D depends on B
    # E depends on C and D
    # F independent
    # G depends on E and F
    
    plan = ExecutionPlan(
        plan_id="complex_test",
        plan_description="Complex dependency test",
        target_dataset_name="test",
        steps=[
            ExecutionStep(step_id="A", action=DynamicToolChainConfig(tools=[ToolCall(tool_id="toolA")])),
            ExecutionStep(step_id="B", action=DynamicToolChainConfig(tools=[ToolCall(tool_id="toolB")])),
            ExecutionStep(
                step_id="C",
                action=DynamicToolChainConfig(tools=[ToolCall(
                    tool_id="toolC",
                    inputs={"in": ToolInputSource(from_step_id="A", named_output_key="out")}
                )])
            ),
            ExecutionStep(
                step_id="D",
                action=DynamicToolChainConfig(tools=[ToolCall(
                    tool_id="toolD",
                    inputs={"in": ToolInputSource(from_step_id="B", named_output_key="out")}
                )])
            ),
            ExecutionStep(
                step_id="E",
                action=DynamicToolChainConfig(tools=[ToolCall(
                    tool_id="toolE",
                    inputs={
                        "in1": ToolInputSource(from_step_id="C", named_output_key="out"),
                        "in2": ToolInputSource(from_step_id="D", named_output_key="out")
                    }
                )])
            ),
            ExecutionStep(step_id="F", action=DynamicToolChainConfig(tools=[ToolCall(tool_id="toolF")])),
            ExecutionStep(
                step_id="G",
                action=DynamicToolChainConfig(tools=[ToolCall(
                    tool_id="toolG",
                    inputs={
                        "in1": ToolInputSource(from_step_id="E", named_output_key="out"),
                        "in2": ToolInputSource(from_step_id="F", named_output_key="out")
                    }
                )])
            ),
        ]
    )
    
    # Analyze dependencies
    deps = orchestrator._analyze_dependencies(plan)
    groups = orchestrator._create_execution_groups(plan, deps)
    
    # Expected groups:
    # Group 1: A, B, F (all independent)
    # Group 2: C, D (C depends on A, D depends on B)
    # Group 3: E (depends on C and D)
    # Group 4: G (depends on E and F)
    
    expected_groups = [
        set(["A", "B", "F"]),
        set(["C", "D"]),
        set(["E"]),
        set(["G"])
    ]
    
    actual_groups = [set(group) for group in groups]
    
    if actual_groups == expected_groups:
        logger.info("✅ Complex dependency handling passed")
        logger.info(f"  Groups: {groups}")
        return True
    else:
        logger.error("❌ Complex dependency handling failed")
        logger.error(f"  Expected: {expected_groups}")
        logger.error(f"  Actual: {actual_groups}")
        return False


async def main():
    """Run all integration tests."""
    logger.info("Starting parallel orchestrator integration tests...\n")
    
    tests = [
        ("Orchestrator Compatibility", test_orchestrator_compatibility),
        ("Parallel vs Sequential Performance", test_parallel_vs_sequential),
        ("Complex Dependencies", test_complex_dependencies),
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
    logger.info(f"Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("✅ All integration tests passed!")
        return True
    else:
        logger.error(f"❌ {failed} integration tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)