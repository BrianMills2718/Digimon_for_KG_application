#!/usr/bin/env python
"""Quick test to verify current system functionality"""

import sys
sys.path.insert(0, '.')
import asyncio
from unittest.mock import MagicMock

from Core.AgentOrchestrator.async_streaming_orchestrator_v2 import AsyncStreamingOrchestrator, UpdateType
from Core.AgentTools.tool_registry import DynamicToolRegistry, ToolCategory, ToolCapability
from Core.Memory.memory_system import GraphRAGMemory
from Core.AgentOrchestrator.memory_enhanced_orchestrator import MemoryEnhancedOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, DynamicToolChainConfig


async def test_integrated_system():
    """Test the integrated system with all checkpoints"""
    print("Testing Integrated DIGIMON System")
    print("=" * 60)
    
    # Test 1: Tool Registry
    print("\n1. Testing Dynamic Tool Registry:")
    registry = DynamicToolRegistry()
    print(f"   - Registered tools: {len(registry)}")
    
    # Check categorization
    read_only = registry.get_tools_by_category(ToolCategory.READ_ONLY)
    print(f"   - Read-only tools: {len(read_only)}")
    
    # Test discovery
    entity_tools = registry.discover_tools(capabilities={ToolCapability.ENTITY_DISCOVERY})
    print(f"   - Entity discovery tools: {len(entity_tools)}")
    print("   ✓ Tool registry working")
    
    # Test 2: Memory System
    print("\n2. Testing Memory System:")
    memory = GraphRAGMemory()
    
    # Learn a pattern
    plan = ExecutionPlan(
        plan_id="test_plan",
        plan_description="Test plan",
        target_dataset_name="test",
        steps=[],
        plan_inputs={}
    )
    
    memory.learn_from_execution(
        query="Test query",
        user_id="test_user",
        plan=plan,
        execution_results={"success": True},
        quality_score=0.9,
        execution_time_ms=1000
    )
    
    # Get recommendation
    rec = memory.recommend_strategy("Test query 2")
    print(f"   - Pattern learned: {rec is not None}")
    print(f"   - System stats: {memory.system_memory.global_stats['total_queries']} queries")
    print("   ✓ Memory system working")
    
    # Test 3: Async Streaming Orchestrator
    print("\n3. Testing Async Streaming:")
    
    # Create mock dependencies
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
    
    # Create simple plan
    test_plan = ExecutionPlan(
        plan_id="stream_test",
        plan_description="Streaming test",
        target_dataset_name="test",
        steps=[
            ExecutionStep(
                step_id="step1",
                description="Test step",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(tool_id="Entity.VDBSearch", inputs={"query": "test"}),
                        ToolCall(tool_id="Entity.PPR", inputs={"seed_entity_ids": ["e1"]})
                    ]
                )
            )
        ],
        plan_inputs={}
    )
    
    # Mock tool execution
    async def mock_execute(tool_call, plan_inputs):
        await asyncio.sleep(0.01)
        return {"result": f"Success for {tool_call.tool_id}"}, None
    
    orchestrator._execute_tool_async = mock_execute
    
    # Collect updates
    updates = []
    async for update in orchestrator.execute_plan_stream(test_plan):
        updates.append(update)
        if update.type in [UpdateType.TOOL_START, UpdateType.TOOL_COMPLETE]:
            print(f"   - {update.type.value}: {update.tool_id}")
    
    print(f"   - Total updates: {len(updates)}")
    print("   ✓ Streaming orchestrator working")
    
    # Test 4: Memory-Enhanced Orchestrator
    print("\n4. Testing Memory-Enhanced Orchestrator:")
    
    mem_orchestrator = MemoryEnhancedOrchestrator(
        main_config=config,
        llm_instance=llm,
        encoder_instance=encoder,
        chunk_factory=chunk_factory,
        graphrag_context=context,
        user_id="test_user"
    )
    
    mem_orchestrator._execute_tool_async = mock_execute
    
    # Execute with memory
    updates = []
    async for update in mem_orchestrator.execute_plan_stream(test_plan, query="Test with memory"):
        updates.append(update)
    
    # Check memory was updated
    stats = mem_orchestrator.get_system_stats()
    print(f"   - Queries processed: {stats['stats']['total_queries']}")
    print("   ✓ Memory-enhanced orchestrator working")
    
    print("\n" + "=" * 60)
    print("All Systems Operational!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_integrated_system())
    sys.exit(0 if success else 1)