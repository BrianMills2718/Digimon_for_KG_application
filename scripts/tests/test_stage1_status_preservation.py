#!/usr/bin/env python3
"""Test Stage 1: Verify orchestrator preserves status/message fields"""

import asyncio
import logging
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, Step, ToolCall
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Chunk.ChunkFactory import ChunkFactory

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_status_preservation():
    """Test that status and message are preserved when using named_outputs"""
    
    # Initialize components
    config = Config.default()
    llm = LiteLLMProvider(config.llm)
    chunk_factory = ChunkFactory(config)
    context = GraphRAGContext()
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        config=config,
        llm=llm,
        encoder=None,  # Not needed for this test
        chunk_factory=chunk_factory,
        graphrag_context=context
    )
    
    # Create a test plan that will fail (PPR on non-existent graph)
    test_plan = ExecutionPlan(
        plan_id="test_status_preservation",
        description="Test plan to verify status preservation",
        steps=[
            Step(
                step_id="step1_ppr",
                description="Run PPR on non-existent graph",
                tool_calls=[
                    ToolCall(
                        tool_id="Entity.PPR",
                        arguments={
                            "graph_reference_id": "NonExistentGraph",
                            "seed_entity_ids": ["test"],
                            "personalization_weight_alpha": 0.85
                        },
                        named_outputs={
                            "entities": "ranked_entities"
                        }
                    )
                ]
            )
        ]
    )
    
    # Execute the plan
    print("\n=== Executing test plan ===")
    try:
        results = await orchestrator.execute_plan(test_plan)
        
        print("\n=== Step outputs ===")
        for step_id, outputs in results.items():
            print(f"\nStep: {step_id}")
            for key, value in outputs.items():
                if key.startswith('_'):
                    print(f"  {key}: {value}")  # Highlight preserved fields
                else:
                    print(f"  {key}: {value}")
        
        # Check if _status was preserved
        step1_outputs = results.get("step1_ppr", {})
        if "_status" in step1_outputs:
            print(f"\n✓ SUCCESS: _status field was preserved: {step1_outputs['_status']}")
        else:
            print("\n✗ FAILURE: _status field was NOT preserved")
            print(f"Available keys: {list(step1_outputs.keys())}")
            
        if "_message" in step1_outputs:
            print(f"✓ SUCCESS: _message field was preserved: {step1_outputs['_message'][:100]}...")
        else:
            print("✗ FAILURE: _message field was NOT preserved")
            
    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_status_preservation())