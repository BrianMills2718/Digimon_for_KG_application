#!/usr/bin/env python3
"""Test VDB recognition after fix"""

import asyncio
import sys
sys.path.append('.')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory

async def test_vdb_recognition():
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
    
    print("Testing VDB Recognition After Fix")
    print("=" * 60)
    
    # First query - should build everything
    print("\nQuery 1: Initial build")
    query1 = """
    Search for entities related to 'artificial intelligence' in Synthetic_Test.
    Build any required resources (corpus, graph, VDB) if they don't exist.
    """
    
    result1 = await agent.process_query(query1, "Synthetic_Test")
    answer1 = result1.get("generated_answer", "")
    context1 = result1.get("retrieved_context", {})
    
    # Count steps executed
    steps1 = len(context1.keys()) if isinstance(context1, dict) else 0
    print(f"Steps executed in query 1: {steps1}")
    
    # Second query - should NOT rebuild VDB
    print("\n\nQuery 2: Should use existing VDB")
    query2 = """
    Search for entities related to 'climate change' in Synthetic_Test.
    Use existing resources where possible.
    """
    
    result2 = await agent.process_query(query2, "Synthetic_Test")
    answer2 = result2.get("generated_answer", "")
    context2 = result2.get("retrieved_context", {})
    
    # Count steps executed
    steps2 = len(context2.keys()) if isinstance(context2, dict) else 0
    print(f"Steps executed in query 2: {steps2}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Check if VDB was built in first query
    vdb_built_q1 = any("vdb" in step.lower() and "build" in step.lower() for step in context1.keys())
    print(f"VDB built in query 1: {vdb_built_q1}")
    
    # Check if VDB was rebuilt in second query
    vdb_built_q2 = any("vdb" in step.lower() and "build" in step.lower() for step in context2.keys())
    print(f"VDB rebuilt in query 2: {vdb_built_q2}")
    
    # Success criteria
    success = vdb_built_q1 and not vdb_built_q2 and steps2 < steps1
    
    if success:
        print("\n✓ VDB recognition FIXED! Second query used existing VDB without rebuilding.")
        print(f"  Query 1: {steps1} steps (built VDB)")
        print(f"  Query 2: {steps2} steps (reused VDB)")
    else:
        print("\n✗ VDB recognition still has issues")
        if vdb_built_q2:
            print("  Problem: VDB was rebuilt unnecessarily in second query")
        if steps2 >= steps1:
            print(f"  Problem: Second query used {steps2} steps, should be fewer than {steps1}")
    
    # Show available resources after queries
    print("\nAvailable resources after queries:")
    print(f"  Graphs: {list(context.list_graphs().keys())}")
    print(f"  VDBs: {list(context.list_vdbs().keys())}")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(test_vdb_recognition())
    exit(0 if success else 1)