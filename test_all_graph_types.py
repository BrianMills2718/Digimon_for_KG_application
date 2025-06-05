#!/usr/bin/env python3
"""Test all 5 graph types"""

import asyncio
import os
import sys
sys.path.append('.')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory

async def test_graph_type(graph_type: str, dataset: str):
    """Test building a specific graph type"""
    
    config = Config.default()
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    context = GraphRAGContext(
        main_config=config,
        target_dataset_name=dataset,
        llm_provider=llm,
        embedding_provider=encoder,
        chunk_storage_manager=chunk_factory
    )
    
    agent = PlanningAgent(config=config, graphrag_context=context)
    
    print(f"\nTesting {graph_type} Graph")
    print("-" * 40)
    
    # Map graph types to proper names
    graph_commands = {
        "ER": "Build an entity-relationship graph",
        "RK": "Build a relation-knowledge graph", 
        "Tree": "Build a tree graph",
        "TreeBalanced": "Build a balanced tree graph",
        "Passage": "Build a passage graph"
    }
    
    query = f"{graph_commands[graph_type]} for {dataset}. Force rebuild to ensure fresh construction."
    
    result = await agent.process_query(query, dataset)
    answer = result.get("generated_answer", "")
    
    # Check for success
    success = False
    if "success" in answer.lower() or "built" in answer.lower():
        # Also check if graph files exist
        graph_dir_map = {
            "ER": "er_graph",
            "RK": "rkg_graph", 
            "Tree": "tree_graph",
            "TreeBalanced": "tree_graph_balanced",
            "Passage": "passage_graph"
        }
        
        graph_path = f"results/{dataset}/{graph_dir_map[graph_type]}"
        if os.path.exists(graph_path):
            files = os.listdir(graph_path)
            if any(f.endswith('.graphml') for f in files):
                success = True
                print(f"✓ {graph_type} graph built successfully")
                print(f"  Files: {[f for f in files if f.endswith(('.graphml', '.json'))]}")
            else:
                print(f"✗ {graph_type} graph directory exists but no graph files found")
        else:
            print(f"✗ {graph_type} graph directory not created at {graph_path}")
    else:
        print(f"✗ {graph_type} graph build failed")
        print(f"  Answer: {answer[:200]}")
    
    return success

async def main():
    """Test all graph types"""
    
    print("Testing All Graph Types")
    print("=" * 60)
    
    # Ensure corpus is prepared first
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
    
    # Prepare corpus
    print("Preparing corpus...")
    await agent.process_query("Prepare corpus from Data/Synthetic_Test", "Synthetic_Test")
    
    # Test each graph type
    graph_types = ["ER", "RK", "Tree", "TreeBalanced", "Passage"]
    results = {}
    
    for graph_type in graph_types:
        try:
            success = await test_graph_type(graph_type, "Synthetic_Test")
            results[graph_type] = success
        except Exception as e:
            print(f"✗ {graph_type} graph test failed with error: {e}")
            results[graph_type] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("GRAPH TYPE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for graph_type, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {graph_type} Graph")
    
    print(f"\nTotal: {passed}/{total} graph types working")
    
    if passed == total:
        print("\n🎉 All graph types working perfectly!")
    else:
        print(f"\n⚠️  {total - passed} graph types need fixing")

if __name__ == "__main__":
    asyncio.run(main())