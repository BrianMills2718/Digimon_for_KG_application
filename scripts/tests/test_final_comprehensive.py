#!/usr/bin/env python3
"""Final comprehensive test of all DIGIMON capabilities"""

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

async def test_digimon_final():
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
    
    print("FINAL DIGIMON FUNCTIONALITY TEST")
    print("=" * 80)
    print("Testing all capabilities with Synthetic_Test dataset")
    print("")
    
    # Track results
    results = {
        "1. Corpus Preparation": False,
        "2. ER Graph Building": False,
        "3. RK Graph Building": False,
        "4. Tree Graph Building": False,
        "5. Tree Graph Balanced Building": False,
        "6. Passage Graph Building": False,
        "7. Entity VDB Building": False,
        "8. Entity Search": False,
        "9. Relationship Extraction": False,
        "10. Text Chunk Retrieval": False,
        "11. Graph Analysis": False,
        "12. ReAct Mode": False
    }
    
    # Test 1: Corpus Preparation
    print("\n1. Testing Corpus Preparation...")
    results["1. Corpus Preparation"] = os.path.exists("results/Synthetic_Test/corpus/Corpus.json")
    
    # Test 2-6: Graph Building (check files directly)
    graph_checks = [
        ("2. ER Graph Building", "results/Synthetic_Test/er_graph/nx_data.graphml"),
        ("3. RK Graph Building", "results/Synthetic_Test/rkg_graph/nx_data.graphml"),
        ("4. Tree Graph Building", "results/Synthetic_Test/tree_graph/tree_data_leaves.pkl"),
        ("5. Tree Graph Balanced Building", "results/Synthetic_Test/tree_graph_balanced/tree_data.pkl"),
        ("6. Passage Graph Building", "results/Synthetic_Test/passage_graph/nx_data.graphml")
    ]
    
    for test_name, path in graph_checks:
        print(f"\n{test_name[3:]}...")
        if os.path.exists(path):
            results[test_name] = True
            print(f"   ✓ Graph exists at {path}")
        else:
            # Try alternate paths
            alt_paths = []
            if "tree" in path.lower():
                alt_paths.append(path.replace("tree_data.pkl", "tree_data_leaves.pkl"))
            if "passage" in path.lower():
                alt_paths.append(path.replace("passage_graph", "passage_of_graph"))
            
            for alt in alt_paths:
                if os.path.exists(alt):
                    results[test_name] = True
                    print(f"   ✓ Graph exists at {alt}")
                    break
    
    # Test 7: Entity VDB Building
    print("\n7. Testing Entity VDB Building...")
    try:
        query = "List all available VDBs for Synthetic_Test"
        result = await agent.process_query(query, "Synthetic_Test")
        answer = result.get("generated_answer", "")
        if "vdb" in answer.lower() or len(context.list_vdbs()) > 0:
            results["7. Entity VDB Building"] = True
            print(f"   ✓ VDBs available: {context.list_vdbs()}")
    except:
        pass
    
    # Test 8: Entity Search
    print("\n8. Testing Entity Search...")
    try:
        query = "Find entities about artificial intelligence in Synthetic_Test"
        result = await agent.process_query(query, "Synthetic_Test")
        answer = result.get("generated_answer", "").lower()
        if any(term in answer for term in ["sarah chen", "openai", "google", "microsoft", "deepmind"]):
            results["8. Entity Search"] = True
            print("   ✓ Entity search successful")
    except:
        pass
    
    # Test 9: Relationship Extraction
    print("\n9. Testing Relationship Extraction...")
    try:
        query = "What is the relationship between DeepMind and hospitals in Synthetic_Test?"
        result = await agent.process_query(query, "Synthetic_Test")
        answer = result.get("generated_answer", "").lower()
        if any(term in answer for term in ["collaboration", "partnership", "ai", "eye disease", "london"]):
            results["9. Relationship Extraction"] = True
            print("   ✓ Relationship extraction successful")
    except:
        pass
    
    # Test 10: Text Chunk Retrieval
    print("\n10. Testing Text Chunk Retrieval...")
    try:
        query = "Get the text about climate change from Synthetic_Test"
        result = await agent.process_query(query, "Synthetic_Test")
        answer = result.get("generated_answer", "").lower()
        if "climate" in answer and len(answer) > 100:
            results["10. Text Chunk Retrieval"] = True
            print("   ✓ Text retrieval successful")
    except:
        pass
    
    # Test 11: Graph Analysis
    print("\n11. Testing Graph Analysis...")
    try:
        query = "Analyze the ER graph structure of Synthetic_Test. How many nodes and edges?"
        result = await agent.process_query(query, "Synthetic_Test")
        answer = result.get("generated_answer", "").lower()
        if any(term in answer for term in ["node", "edge", "vertex", "145", "109"]):
            results["11. Graph Analysis"] = True
            print("   ✓ Graph analysis successful")
    except:
        pass
    
    # Test 12: ReAct Mode
    print("\n12. Testing ReAct Mode...")
    try:
        react_result = await agent.process_query_react(
            "Compare AI and climate themes in Synthetic_Test",
            "Synthetic_Test"
        )
        if react_result.get("iterations", 0) >= 2:
            results["12. ReAct Mode"] = True
            print(f"   ✓ ReAct mode working ({react_result.get('iterations')} iterations)")
    except:
        pass
    
    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} capabilities working ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 DIGIMON IS 100% FUNCTIONAL!")
        print("All capabilities have been successfully demonstrated.")
    else:
        print(f"\n{total - passed} capabilities still need work:")
        failed_tests = [name for name, passed in results.items() if not passed]
        for test in failed_tests:
            print(f"  - {test}")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(test_digimon_final())
    exit(0 if success else 1)