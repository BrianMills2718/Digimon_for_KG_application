#!/usr/bin/env python3
"""Test failure detection in agent brain"""

import asyncio
import sys
import json
sys.path.append('.')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Common.Logger import logger
from Core.AgentSchema.plan import ExecutionPlan

# Create a test plan that will fail
TEST_PLAN = {
    "plan_id": "test_failure_detection",
    "plan_description": "Test failure detection",
    "target_dataset_name": "NonExistentDataset",  # This will cause a failure
    "plan_inputs": {},
    "steps": [
        {
            "step_id": "step_1",
            "description": "Build graph for non-existent dataset",
            "action": {
                "tools": [{
                    "tool_id": "graph.BuildERGraph",
                    "inputs": {
                        "target_dataset_name": "NonExistentDataset"
                    },
                    "named_outputs": {
                        "graph_id_alias": "graph_id"
                    }
                }]
            }
        }
    ]
}

async def test_failure_detection():
    config = Config.default()
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    context = GraphRAGContext(
        main_config=config,
        target_dataset_name="NonExistentDataset",
        llm_provider=llm,
        embedding_provider=encoder,
        chunk_storage_manager=chunk_factory
    )
    
    agent = PlanningAgent(config=config, graphrag_context=context)
    
    print("=" * 80)
    print("FAILURE DETECTION TEST")
    print("=" * 80)
    print("Testing if agent correctly detects tool failures")
    print("")
    
    # Convert test plan to proper Pydantic object
    plan = ExecutionPlan(**TEST_PLAN)
    
    # Execute the plan directly
    step_outputs = await agent.orchestrator.execute_plan(plan)
    
    print("\n" + "=" * 80)
    print("STEP OUTPUTS:")
    print("=" * 80)
    print(json.dumps(step_outputs, indent=2, default=str))
    
    # Simulate what the agent brain does
    print("\n" + "=" * 80)
    print("AGENT BRAIN FAILURE DETECTION:")
    print("=" * 80)
    
    # Check each step result (as agent brain does)
    for step_id, step_result in step_outputs.items():
        print(f"\nChecking {step_id}:")
        print(f"  Type: {type(step_result)}")
        print(f"  Content: {step_result}")
        
        if isinstance(step_result, dict):
            # Check for explicit failure status
            if step_result.get("status") == "failure":
                print(f"  ✓ Found 'status': 'failure' at step level")
            else:
                print(f"  ✗ No 'status': 'failure' at step level")
                
            # Check for status in any of the values
            for key, value in step_result.items():
                print(f"    Checking '{key}': {value}")
                if hasattr(value, 'status') and value.status == "failure":
                    print(f"    ✓ Found failure status in Pydantic object")
                elif isinstance(value, dict) and value.get("status") == "failure":
                    print(f"    ✓ Found failure status in dict")
                elif isinstance(value, str) and "status='failure'" in value:
                    print(f"    ✓ Found failure status in string representation")

if __name__ == "__main__":
    asyncio.run(test_failure_detection())