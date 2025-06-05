#!/usr/bin/env python3
"""Test to see what plan is generated for VDB operations"""

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

async def test_plan_generation():
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
    
    # Add a fake VDB to context
    context.add_vdb_instance("Synthetic_Test_entities", {"fake": "vdb"})
    
    agent = PlanningAgent(config=config, graphrag_context=context)
    
    print("Testing Plan Generation with Existing VDB")
    print("=" * 60)
    
    # Generate plan with existing VDB
    available_resources = {
        'corpus_prepared': True,
        'graphs': ['Synthetic_Test_er_graph'],
        'vdbs': ['Synthetic_Test_entities']
    }
    
    query = "Search for entities related to 'climate change' in Synthetic_Test."
    
    plan = await agent.generate_plan(query, "Synthetic_Test", available_resources)
    
    if plan:
        print("\nGenerated Plan:")
        print(json.dumps(plan.model_dump(), indent=2))
        
        # Check if VDB build step is present
        vdb_build_found = False
        for step in plan.steps:
            if step.action and step.action.tools:
                for tool in step.action.tools:
                    if tool.tool_id == "Entity.VDB.Build":
                        vdb_build_found = True
                        print(f"\n⚠️  VDB Build step found: {step.step_id}")
                        print(f"  Inputs: {tool.inputs}")
                        break
        
        if not vdb_build_found:
            print("\n✓ Good! No VDB build step in plan (VDB already exists)")
        else:
            print("\n✗ Problem: VDB build step included despite existing VDB")
    else:
        print("Failed to generate plan")

if __name__ == "__main__":
    asyncio.run(test_plan_generation())