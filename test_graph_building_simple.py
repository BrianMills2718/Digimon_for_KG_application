#!/usr/bin/env python3
"""Simple test to verify graph building capabilities"""

import asyncio
import sys
import os
sys.path.append('.')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory

async def test_graph_building():
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
    
    print("GRAPH BUILDING CAPABILITY TEST")
    print("=" * 80)
    
    # First, let's see what graphs already exist
    print("\n1. Checking existing graphs...")
    existing_graphs = context.list_graphs()
    print(f"   Existing graphs in context: {existing_graphs}")
    
    # Check file system
    graph_dirs = []
    results_dir = "results/Synthetic_Test"
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            if "graph" in item and os.path.isdir(os.path.join(results_dir, item)):
                graph_dirs.append(item)
    print(f"   Graph directories on disk: {graph_dirs}")
    
    # Test building one missing graph type - RK Graph
    print("\n2. Testing RK Graph Building...")
    rk_paths = [
        "results/Synthetic_Test/rkg_graph/nx_data.graphml",
        "results/Synthetic_Test/rkg_graph/graph_storage/nx_data.graphml"
    ]
    
    rk_exists = any(os.path.exists(path) for path in rk_paths)
    
    if rk_exists:
        print("   ✓ RK Graph already exists")
    else:
        query = "Build an RK graph for Synthetic_Test"
        result = await agent.process_query(query, "Synthetic_Test")
        
        # Check if it was built
        rk_exists = any(os.path.exists(path) for path in rk_paths)
        if rk_exists:
            print("   ✓ RK Graph built successfully")
        else:
            print("   ✗ RK Graph build failed")
            # Print the result for debugging
            print(f"   Result: {result.get('generated_answer', 'No answer')[:200]}")
    
    # Test building Tree Graph
    print("\n3. Testing Tree Graph Building...")
    tree_paths = [
        "results/Synthetic_Test/tree_graph/nx_data.graphml",
        "results/Synthetic_Test/tree_graph/tree_data.pkl",
        "results/Synthetic_Test/tree_graph/graph_storage_tree_data.pkl"
    ]
    
    tree_exists = any(os.path.exists(path) for path in tree_paths)
    
    if tree_exists:
        print("   ✓ Tree Graph already exists")
    else:
        query = "Build a tree graph for Synthetic_Test"
        result = await agent.process_query(query, "Synthetic_Test")
        
        # Check if it was built
        tree_exists = any(os.path.exists(path) for path in tree_paths)
        if tree_exists:
            print("   ✓ Tree Graph built successfully")
        else:
            print("   ✗ Tree Graph build failed")
            print(f"   Result: {result.get('generated_answer', 'No answer')[:200]}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Re-check what graphs exist now
    final_graphs = context.list_graphs()
    print(f"Graphs in context: {final_graphs}")
    
    # Check file system again
    final_graph_dirs = []
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            if "graph" in item and os.path.isdir(os.path.join(results_dir, item)):
                final_graph_dirs.append(item)
    print(f"Graph directories on disk: {final_graph_dirs}")
    
    return len(final_graph_dirs) >= 2  # Success if we have at least 2 graph types

if __name__ == "__main__":
    success = asyncio.run(test_graph_building())
    exit(0 if success else 1)