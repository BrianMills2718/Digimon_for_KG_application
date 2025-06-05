#!/usr/bin/env python3
"""Test output key mismatch between what agent expects and what tools produce"""

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

# Create a simple test plan that expects a specific output key
TEST_PLAN = {
    "plan_id": "test_output_keys",
    "plan_description": "Test output key handling",
    "target_dataset_name": "Synthetic_Test",
    "plan_inputs": {},
    "steps": [
        {
            "step_id": "step_1",
            "description": "Build ER graph",
            "action": {
                "tools": [{
                    "tool_id": "graph.BuildERGraph",
                    "inputs": {
                        "target_dataset_name": "Synthetic_Test"
                    },
                    "named_outputs": {
                        "my_custom_graph_id": "graph_id"  # Expect 'graph_id' output as 'my_custom_graph_id'
                    }
                }]
            }
        },
        {
            "step_id": "step_2",
            "description": "Try to use the output",
            "action": {
                "tools": [{
                    "tool_id": "Entity.VDB.Build",
                    "inputs": {
                        "graph_reference_id": {
                            "from_step_id": "step_1",
                            "named_output_key": "my_custom_graph_id"  # Reference by alias
                        },
                        "vdb_collection_name": "test_collection"
                    },
                    "named_outputs": {}
                }]
            }
        }
    ]
}

async def test_output_keys():
    config = Config.default()
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    context = GraphRAGContext(
        main_config=config,
        target_dataset_name="Synthetic_Test",
        llm_provider=llm,
        embedding_provider=encoder,
        chunk_storage_manager=chunk_factory
    )
    
    agent = PlanningAgent(config=config, graphrag_context=context)
    
    print("=" * 80)
    print("OUTPUT KEY MISMATCH TEST")
    print("=" * 80)
    print("Testing if orchestrator correctly handles named_outputs aliasing")
    print("")
    
    # Convert test plan to proper Pydantic object
    from Core.AgentSchema.plan import ExecutionPlan
    plan = ExecutionPlan(**TEST_PLAN)
    
    # Execute the plan directly
    step_outputs = await agent.orchestrator.execute_plan(plan)
    
    print("\n" + "=" * 80)
    print("STEP OUTPUTS:")
    print("=" * 80)
    print(json.dumps(step_outputs, indent=2, default=str))
    
    # Check if the aliasing worked
    if "step_1" in step_outputs and "my_custom_graph_id" in step_outputs["step_1"]:
        print("\n✓ SUCCESS: Output was correctly aliased as 'my_custom_graph_id'")
        print(f"  Value: {step_outputs['step_1']['my_custom_graph_id']}")
    else:
        print("\n✗ FAILURE: Output aliasing did not work")
        if "step_1" in step_outputs:
            print(f"  Available keys in step_1: {list(step_outputs['step_1'].keys())}")
    
    # Check if step 2 could resolve the reference
    if "step_2" in step_outputs:
        print("\n✓ Step 2 executed (reference resolution worked)")
    else:
        print("\n✗ Step 2 failed to execute (reference resolution failed)")

if __name__ == "__main__":
    asyncio.run(test_output_keys())