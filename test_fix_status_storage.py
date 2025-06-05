#!/usr/bin/env python3
"""Test fix for storing status alongside named outputs"""

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

# Monkey patch the orchestrator to also store status
def patched_execute_plan(self, plan):
    # Call the original method
    original_execute_plan = self.__class__.execute_plan.__wrapped__
    
    async def execute_with_status_storage(plan_obj):
        # Store reference to original execute
        step_outputs = await original_execute_plan(self, plan_obj)
        
        # Now enhance the outputs with status information
        # This would be done in the actual orchestrator code
        logger.info("PATCH: Enhancing step outputs with status information")
        
        return step_outputs
    
    return execute_with_status_storage(plan)

async def test_with_fixed_orchestrator():
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
    print("PROPOSED FIX: Store tool status alongside named outputs")
    print("=" * 80)
    print("")
    print("The issue: When named_outputs is used, only the requested fields are stored.")
    print("This means 'status' and 'message' fields are lost, breaking failure detection.")
    print("")
    print("Solution options:")
    print("1. Always store 'status' and 'message' fields if present (alongside named outputs)")
    print("2. Store the full tool output under a special key like '_tool_output'")
    print("3. Change agent brain to check the actual output values, not just 'status' field")
    print("")
    print("Recommended: Option 1 - Always preserve status/message fields")
    print("")
    
    # Show what the enhanced output structure would look like
    print("Current structure (loses status):")
    print(json.dumps({
        "step_1": {
            "graph_id_alias": ""  # Only the named output is stored
        }
    }, indent=2))
    
    print("\nProposed structure (preserves status):")
    print(json.dumps({
        "step_1": {
            "graph_id_alias": "",  # Named output
            "_status": "failure",  # Always preserved
            "_message": "No input chunks found for dataset: NonExistentDataset"  # Always preserved
        }
    }, indent=2))
    
    print("\nThis would allow the agent brain to detect failures by checking:")
    print("  if step_result.get('_status') == 'failure':")
    print("      # Handle failure")

if __name__ == "__main__":
    asyncio.run(test_with_fixed_orchestrator())