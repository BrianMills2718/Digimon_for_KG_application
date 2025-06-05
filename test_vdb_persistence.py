#!/usr/bin/env python3
"""Test VDB persistence in context"""

import asyncio
import sys
sys.path.append('.')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory

async def test_vdb_persistence():
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
    
    print("Testing VDB Persistence in Context")
    print("=" * 60)
    
    # Check initial state
    print("\nInitial state:")
    print(f"  Graphs: {list(context.list_graphs().keys()) if hasattr(context.list_graphs(), 'keys') else context.list_graphs()}")
    print(f"  VDBs: {list(context.list_vdbs().keys()) if hasattr(context.list_vdbs(), 'keys') else context.list_vdbs()}")
    
    # Build VDB
    print("\n\nBuilding VDB...")
    query1 = """
    Build an entity vector database for Synthetic_Test.
    First prepare corpus and build ER graph if needed.
    """
    
    result1 = await agent.process_query(query1, "Synthetic_Test")
    
    # Check state after first query
    print("\n\nState after VDB build:")
    graphs = context.list_graphs()
    vdbs = context.list_vdbs()
    print(f"  Graphs type: {type(graphs)}")
    print(f"  Graphs: {list(graphs.keys()) if hasattr(graphs, 'keys') else graphs}")
    print(f"  VDBs type: {type(vdbs)}")
    print(f"  VDBs: {list(vdbs.keys()) if hasattr(vdbs, 'keys') else vdbs}")
    
    # Direct check of internal state
    if hasattr(context, '_graphs'):
        print(f"\n  Context._graphs: {list(context._graphs.keys())}")
    if hasattr(context, '_vdbs'):
        print(f"  Context._vdbs: {list(context._vdbs.keys())}")
    
    # Try to use VDB
    print("\n\nUsing VDB...")
    query2 = "Search for entities about 'AI' using the existing VDB."
    
    result2 = await agent.process_query(query2, "Synthetic_Test")
    
    # Final state
    print("\n\nFinal state:")
    print(f"  Graphs: {list(context.list_graphs().keys()) if hasattr(context.list_graphs(), 'keys') else context.list_graphs()}")
    print(f"  VDBs: {list(context.list_vdbs().keys()) if hasattr(context.list_vdbs(), 'keys') else context.list_vdbs()}")
    
    # Check if second query rebuilt VDB
    context2 = result2.get("retrieved_context", {})
    vdb_rebuilt = any("vdb" in step.lower() and "build" in step.lower() for step in context2.keys())
    
    if vdb_rebuilt:
        print("\n✗ VDB was rebuilt in second query")
    else:
        print("\n✓ VDB was reused in second query")
    
    return not vdb_rebuilt

if __name__ == "__main__":
    success = asyncio.run(test_vdb_persistence())
    exit(0 if success else 1)