#!/usr/bin/env python3
"""Final test of VDB recognition fix"""

import asyncio
import sys
sys.path.append('.')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory

async def test_vdb_final():
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
    
    print("Final VDB Recognition Test")
    print("=" * 60)
    
    # First query - build everything
    print("\nQuery 1: Build VDB")
    query1 = """
    I need to search for AI-related entities in Synthetic_Test.
    Please ensure all necessary resources are prepared (corpus, graph, VDB).
    """
    
    result1 = await agent.process_query(query1, "Synthetic_Test")
    context1 = result1.get("retrieved_context", {})
    
    # Count VDB builds in first query
    vdb_builds_q1 = sum(1 for step in context1.keys() if "vdb" in step.lower() and "build" in step.lower())
    print(f"VDB builds in query 1: {vdb_builds_q1}")
    
    # Show available resources
    print(f"\nAfter query 1:")
    print(f"  Graphs: {context.list_graphs()}")
    print(f"  VDBs: {context.list_vdbs()}")
    
    # Second query - should use existing resources
    print("\n\nQuery 2: Search again")
    query2 = """
    Search for climate-related entities in Synthetic_Test.
    """
    
    result2 = await agent.process_query(query2, "Synthetic_Test")
    context2 = result2.get("retrieved_context", {})
    
    # Count VDB builds in second query
    vdb_builds_q2 = sum(1 for step in context2.keys() if "vdb" in step.lower() and "build" in step.lower())
    print(f"VDB builds in query 2: {vdb_builds_q2}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    success = vdb_builds_q1 > 0 and vdb_builds_q2 == 0
    
    if success:
        print("✓ SUCCESS: VDB was built in query 1 but NOT rebuilt in query 2")
        print(f"  Query 1: {len(context1)} steps ({vdb_builds_q1} VDB builds)")
        print(f"  Query 2: {len(context2)} steps ({vdb_builds_q2} VDB builds)")
    else:
        print("✗ FAILURE: VDB recognition issue persists")
        if vdb_builds_q1 == 0:
            print("  Problem: No VDB built in query 1")
        if vdb_builds_q2 > 0:
            print(f"  Problem: VDB rebuilt {vdb_builds_q2} times in query 2")
    
    # Show final state
    print(f"\nFinal state:")
    print(f"  Graphs: {context.list_graphs()}")
    print(f"  VDBs: {context.list_vdbs()}")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(test_vdb_final())
    exit(0 if success else 1)