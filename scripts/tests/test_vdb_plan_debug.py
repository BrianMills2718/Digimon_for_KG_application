#!/usr/bin/env python3
"""Debug why VDB is rebuilt even when available"""

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

async def debug_vdb_plan():
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
    
    # Simulate existing resources
    context.add_graph_instance("Synthetic_Test_ERGraph", {"fake": "graph"})
    context.add_vdb_instance("Synthetic_Test_entities", {"fake": "vdb"})
    
    agent = PlanningAgent(config=config, graphrag_context=context)
    
    print("Debugging VDB Plan Generation")
    print("=" * 60)
    
    print(f"\nInitial VDBs: {context.list_vdbs()}")
    
    # Query that should use existing VDB
    query = "Search for entities about 'AI' using the existing VDB in Synthetic_Test."
    
    # Check available resources
    available_resources = {
        'corpus_prepared': True,
        'graphs': ['Synthetic_Test_ERGraph'],
        'vdbs': ['Synthetic_Test_entities']
    }
    
    print(f"\nAvailable resources: {available_resources}")
    
    # Generate plan
    plan = await agent.generate_plan(query, "Synthetic_Test", available_resources)
    
    if plan:
        print("\nGenerated Plan:")
        print(f"Plan ID: {plan.plan_id}")
        print(f"Description: {plan.plan_description}")
        print(f"\nSteps:")
        for i, step in enumerate(plan.steps):
            print(f"\n{i+1}. {step.step_id}: {step.description}")
            if step.action and step.action.tools:
                for tool in step.action.tools:
                    print(f"   Tool: {tool.tool_id}")
                    print(f"   Inputs: {json.dumps(tool.inputs, indent=6)}")
                    
                    # Check if this is a VDB build
                    if tool.tool_id == "Entity.VDB.Build":
                        print("   ⚠️  WARNING: VDB Build included despite existing VDB!")
                        print(f"   vdb_collection_name: {tool.inputs.get('vdb_collection_name')}")
                        print(f"   force_rebuild: {tool.inputs.get('force_rebuild', 'not specified')}")

if __name__ == "__main__":
    asyncio.run(debug_vdb_plan())