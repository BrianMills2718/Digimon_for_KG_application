#!/usr/bin/env python3
"""
Test script for parallel execution capabilities
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
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentOrchestrator.parallel_orchestrator import ParallelAgentOrchestrator
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory
from Option.Config2 import default_config


async def create_test_plan() -> ExecutionPlan:
    """Create a test plan with both independent and dependent steps."""
    
    # Create a plan with mixed dependencies
    plan = ExecutionPlan(
        plan_id="test_parallel_plan",
        plan_description="Test plan for parallel execution",
        target_dataset_name="MySampleTexts",
        plan_inputs={"query": "test query", "dataset": "MySampleTexts"},
        steps=[
            # Group 1: Independent steps (can run in parallel)
            ExecutionStep(
                step_id="step_1",
                description="Prepare corpus",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="corpus.PrepareFromDirectory",
                            inputs={"corpus_directory": "plan_inputs.dataset"},
                            named_outputs={"corpus_output": "corpus_json_path"}
                        )
                    ]
                )
            ),
            ExecutionStep(
                step_id="step_2",
                description="Analyze existing graph (independent)",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="graph.Analyze",
                            inputs={"graph_id": "MySampleTexts_ERGraph"},
                            named_outputs={"analysis_output": "graph_stats"}
                        )
                    ]
                )
            ),
            
            # Group 2: Dependent on step_1
            ExecutionStep(
                step_id="step_3",
                description="Build ER graph (depends on corpus)",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="graph.BuildERGraph",
                            inputs={
                                "corpus_json_path": ToolInputSource(
                                    from_step_id="step_1",
                                    named_output_key="corpus_output"
                                )
                            },
                            named_outputs={"graph_build_output": "graph_id"}
                        )
                    ]
                )
            ),
            
            # Group 3: Can run in parallel, both depend on step_3
            ExecutionStep(
                step_id="step_4",
                description="Build entity VDB (depends on graph)",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Entity.VDB.Build",
                            inputs={
                                "graph_id": ToolInputSource(
                                    from_step_id="step_3",
                                    named_output_key="graph_build_output"
                                )
                            },
                            named_outputs={"entity_vdb": "entity_vdb_path"}
                        )
                    ]
                )
            ),
            ExecutionStep(
                step_id="step_5",
                description="Visualize graph (depends on graph, parallel with VDB build)",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="graph.Visualize",
                            inputs={
                                "graph_id": ToolInputSource(
                                    from_step_id="step_3",
                                    named_output_key="graph_build_output"
                                )
                            },
                            named_outputs={"visualization": "graph_image_path"}
                        )
                    ]
                )
            ),
            
            # Group 4: Final step depending on multiple previous steps
            ExecutionStep(
                step_id="step_6",
                description="Search entities (depends on VDB)",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Entity.VDBSearch",
                            inputs={
                                "query": "plan_inputs.query",
                                "vdb_path": ToolInputSource(
                                    from_step_id="step_4",
                                    named_output_key="entity_vdb"
                                )
                            },
                            named_outputs={"search_results": "similar_entities"}
                        )
                    ]
                )
            )
        ]
    )
    
    return plan


async def test_dependency_analysis():
    """Test dependency analysis without executing."""
    logger.info("=== Testing Dependency Analysis ===")
    
    # Create orchestrator without LLM wrapper (just for dependency analysis)
    config = default_config
    
    # Create a minimal orchestrator just for testing dependency analysis
    from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
    orchestrator = ParallelAgentOrchestrator.__new__(ParallelAgentOrchestrator)
    orchestrator.dependency_graph = {}
    orchestrator.execution_groups = []
    
    # Create test plan
    plan = await create_test_plan()
    
    # Analyze dependencies
    deps = orchestrator._analyze_dependencies(plan)
    groups = orchestrator._create_execution_groups(plan, deps)
    
    logger.info("Dependencies:")
    for step_id, dependencies in deps.items():
        if dependencies:
            logger.info(f"  {step_id} depends on: {dependencies}")
        else:
            logger.info(f"  {step_id} has no dependencies")
    
    logger.info("\nExecution Groups:")
    for i, group in enumerate(groups):
        logger.info(f"  Group {i + 1}: {group}")
    
    # Verify expected grouping
    expected_groups = [
        ["step_1", "step_2"],  # Independent steps
        ["step_3"],            # Depends on step_1
        ["step_4", "step_5"],  # Both depend on step_3
        ["step_6"]             # Depends on step_4
    ]
    
    if groups == expected_groups:
        logger.info("✓ Dependency analysis correct!")
    else:
        logger.error(f"✗ Dependency analysis incorrect. Expected: {expected_groups}")
    
    return groups == expected_groups


async def test_parallel_mock_execution():
    """Test parallel execution with mock tools."""
    logger.info("\n=== Testing Parallel Mock Execution ===")
    
    # Create mock tools that simulate work
    async def mock_tool(params, context=None):
        """Mock tool that simulates work."""
        if hasattr(params, '_data'):
            tool_id = params._data.get("_tool_id", "unknown")
            delay = params._data.get("_delay", 1.0)
        else:
            tool_id = "unknown"
            delay = 1.0
        
        logger.info(f"Mock tool {tool_id} starting (will take {delay}s)")
        await asyncio.sleep(delay)
        logger.info(f"Mock tool {tool_id} completed")
        
        return {
            "status": "success",
            "tool_id": tool_id,
            "timestamp": time.time()
        }
    
    # Create a minimal orchestrator for testing
    config = default_config
    
    # Create minimal context
    context = GraphRAGContext(
        main_config=config, 
        embedding_provider=None,
        target_dataset_name="test"
    )
    
    # Create a mock LLM to avoid initialization issues
    class MockLLM:
        pass
    
    orchestrator = ParallelAgentOrchestrator(
        main_config=config,
        llm_instance=MockLLM(),
        encoder_instance=None,
        chunk_factory=None,
        graphrag_context=context
    )
    
    # Override tools with mocks
    class MockInput:
        def __init__(self, **kwargs):
            self._data = kwargs
            self._data["_tool_id"] = kwargs.get("tool_id", "unknown")
            self._data["_delay"] = kwargs.get("delay", 1.0)
        
        def get(self, key, default=None):
            return self._data.get(key, default)
    
    # Register mock tools
    for tool_id in ["corpus.PrepareFromDirectory", "graph.Analyze", 
                    "graph.BuildERGraph", "Entity.VDB.Build", 
                    "graph.Visualize", "Entity.VDBSearch"]:
        orchestrator._tool_registry[tool_id] = (mock_tool, MockInput)
    
    # Create simple test plan with delays
    plan = ExecutionPlan(
        plan_id="mock_parallel_plan",
        plan_description="Mock plan for testing parallel execution",
        target_dataset_name="test",
        steps=[
            # Two independent steps that should run in parallel
            ExecutionStep(
                step_id="independent_1",
                description="Independent task 1",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="corpus.PrepareFromDirectory",
                            parameters={"tool_id": "task1", "delay": 2.0}
                        )
                    ]
                )
            ),
            ExecutionStep(
                step_id="independent_2",
                description="Independent task 2",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="graph.Analyze",
                            parameters={"tool_id": "task2", "delay": 2.0}
                        )
                    ]
                )
            ),
            # Dependent step
            ExecutionStep(
                step_id="dependent_1",
                description="Dependent task",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="graph.BuildERGraph",
                            inputs={
                                "input_data": ToolInputSource(
                                    from_step_id="independent_1",
                                    named_output_key="corpus.PrepareFromDirectory"
                                )
                            },
                            parameters={"tool_id": "task3", "delay": 1.0}
                        )
                    ]
                )
            )
        ]
    )
    
    # Execute with timing
    start_time = time.time()
    results = await orchestrator.execute_plan(plan)
    end_time = time.time()
    
    total_time = end_time - start_time
    logger.info(f"\nTotal execution time: {total_time:.2f}s")
    
    # If parallel execution worked, should take ~3s (2s parallel + 1s sequential)
    # If sequential, would take ~5s (2s + 2s + 1s)
    if total_time < 4.0:
        logger.info("✓ Parallel execution confirmed! Saved ~2s")
    else:
        logger.warning("✗ Execution took longer than expected for parallel")
    
    # Check results
    logger.info("\nStep outputs:")
    for step_id, output in results.items():
        logger.info(f"  {step_id}: {output}")
    
    return total_time < 4.0


async def main():
    """Run all tests."""
    logger.info("Starting parallel execution tests...\n")
    
    tests = [
        ("Dependency Analysis", test_dependency_analysis),
        ("Parallel Mock Execution", test_parallel_mock_execution),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
                logger.info(f"✅ {test_name} passed\n")
            else:
                failed += 1
                logger.error(f"❌ {test_name} failed\n")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} crashed: {e}\n")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("✅ All parallel execution tests passed!")
    else:
        logger.error(f"❌ {failed} tests failed")


if __name__ == "__main__":
    asyncio.run(main())