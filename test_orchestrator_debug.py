#!/usr/bin/env python3
"""Debug orchestrator step output handling"""

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

async def debug_orchestrator():
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
    
    # Simple query that should create a plan with named outputs
    query = "Search for entities about artificial intelligence in Synthetic_Test"
    
    print("=" * 80)
    print("ORCHESTRATOR DEBUG TEST")
    print("=" * 80)
    print(f"Query: {query}")
    print("")
    
    # Process the query
    result = await agent.process_query(query, "Synthetic_Test")
    
    # Print the orchestrator's step_outputs after execution
    if hasattr(agent, 'orchestrator') and hasattr(agent.orchestrator, 'step_outputs'):
        print("\n" + "=" * 80)
        print("ORCHESTRATOR STEP OUTPUTS:")
        print("=" * 80)
        for step_id, outputs in agent.orchestrator.step_outputs.items():
            print(f"\nStep: {step_id}")
            print(f"Outputs: {json.dumps(outputs, indent=2, default=str)}")
    
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(result)

if __name__ == "__main__":
    asyncio.run(debug_orchestrator())