#!/usr/bin/env python3
"""Test VDB with explicit instruction to use existing resources"""

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

async def test_vdb_explicit():
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
    
    print("Test VDB with Explicit Instructions")
    print("=" * 60)
    
    # First query - build VDB
    print("\nQuery 1: Build VDB")
    query1 = "Build entity VDB for Synthetic_Test."
    
    result1 = await agent.process_query(query1, "Synthetic_Test")
    
    print(f"\nAfter query 1:")
    print(f"  VDBs: {context.list_vdbs()}")
    
    # Second query - explicitly use existing
    print("\n\nQuery 2: Use existing VDB")
    query2 = """
    Use the existing VDB 'Synthetic_Test_entities' to search for climate entities.
    Do NOT build a new VDB. The VDB already exists.
    """
    
    # Generate plan for second query to see what it does
    available_resources = {
        'corpus_prepared': True,
        'graphs': ['Synthetic_Test_ERGraph'],
        'vdbs': ['Synthetic_Test_entities']
    }
    
    plan2 = await agent.generate_plan(query2, "Synthetic_Test", available_resources)
    
    if plan2:
        print("\nPlan for query 2:")
        for i, step in enumerate(plan2.steps):
            print(f"\n{i+1}. {step.step_id}: {step.description}")
            if step.action and step.action.tools:
                for tool in step.action.tools:
                    print(f"   Tool: {tool.tool_id}")
                    if tool.tool_id == "Entity.VDB.Build":
                        print("   ❌ ERROR: Still trying to build VDB!")
                    elif tool.tool_id == "Entity.VDBSearch":
                        print(f"   ✓ Good: Using existing VDB")
                        print(f"   VDB ID: {tool.inputs.get('vdb_reference_id')}")

if __name__ == "__main__":
    asyncio.run(test_vdb_explicit())