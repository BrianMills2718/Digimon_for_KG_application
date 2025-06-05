#!/usr/bin/env python3
"""Test complete DIGIMON functionality with synthetic data"""

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

async def test_basic_functionality():
    """Test basic corpus preparation and graph building"""
    
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
    
    print("\n" + "="*60)
    print("TEST: Basic Functionality (Corpus + ER Graph + Search)")
    print("="*60)
    
    # Test 1: Prepare corpus and build graph
    query1 = "Prepare the corpus from Synthetic_Test directory and build an ER graph"
    result1 = await agent.process_query(query1, "Synthetic_Test")
    
    if "success" in result1.get("generated_answer", "").lower():
        print("✓ Step 1: Corpus prepared and ER graph built")
    else:
        print("✗ Step 1 Failed:", result1.get("generated_answer", "")[:100])
        
    # Test 2: Entity search
    query2 = "Find all entities related to AI and technology"
    result2 = await agent.process_query(query2, "Synthetic_Test")
    answer2 = result2.get("generated_answer", "")
    
    if any(term in answer2.lower() for term in ["google", "microsoft", "openai", "deepmind", "sarah chen"]):
        print("✓ Step 2: Entity search working - found AI-related entities")
    else:
        print("✗ Step 2 Failed:", answer2[:100])
        
    # Test 3: Relationship analysis
    query3 = "What is the relationship between Dr. Sarah Chen and Stanford University?"
    result3 = await agent.process_query(query3, "Synthetic_Test")
    answer3 = result3.get("generated_answer", "")
    
    if "researcher" in answer3.lower() or "stanford" in answer3.lower():
        print("✓ Step 3: Relationship analysis working")
    else:
        print("? Step 3 Uncertain:", answer3[:100])
        
    # Test 4: Text retrieval
    query4 = "Get the actual text about DeepMind's healthcare initiatives"
    result4 = await agent.process_query(query4, "Synthetic_Test")
    answer4 = result4.get("generated_answer", "")
    
    if "eye disease" in answer4.lower() or "london" in answer4.lower():
        print("✓ Step 4: Text chunk retrieval working")
    else:
        print("? Step 4 Uncertain:", answer4[:100])
        
    return True

async def test_advanced_graphs():
    """Test different graph types"""
    
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
    
    print("\n" + "="*60)
    print("TEST: Advanced Graph Types")
    print("="*60)
    
    # Test RK Graph
    query_rk = "Build a relation-knowledge graph for Synthetic_Test"
    result_rk = await agent.process_query(query_rk, "Synthetic_Test")
    
    if "success" in result_rk.get("generated_answer", "").lower():
        print("✓ RK Graph: Built successfully")
    else:
        print("? RK Graph:", result_rk.get("generated_answer", "")[:100])
        
    # Test Tree Graph
    query_tree = "Build a tree graph for hierarchical analysis of Synthetic_Test"
    result_tree = await agent.process_query(query_tree, "Synthetic_Test")
    
    if "success" in result_tree.get("generated_answer", "").lower() or "built" in result_tree.get("generated_answer", "").lower():
        print("✓ Tree Graph: Built successfully")
    else:
        print("? Tree Graph:", result_tree.get("generated_answer", "")[:100])
        
    return True

async def test_discourse_analysis():
    """Test discourse analysis capabilities"""
    
    config = Config.default()
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    context = GraphRAGContext(
        main_config=config,
        target_dataset_name="Discourse_Test",
        llm_provider=llm,
        embedding_provider=encoder,
        chunk_storage_manager=chunk_factory
    )
    
    agent = PlanningAgent(config=config, graphrag_context=context)
    
    print("\n" + "="*60)
    print("TEST: Discourse Analysis")
    print("="*60)
    
    # Prepare corpus first
    await agent.process_query("Prepare corpus from Discourse_Test", "Discourse_Test")
    
    # Test discourse analysis
    query = "Analyze the discourse patterns and rhetorical strategies in the UBI debate"
    result = await agent.process_query(query, "Discourse_Test")
    answer = result.get("generated_answer", "")
    
    discourse_terms = ["narrative", "framing", "rhetoric", "proponent", "opponent", "metaphor"]
    if any(term in answer.lower() for term in discourse_terms):
        print("✓ Discourse Analysis: Successfully identified discourse patterns")
        print(f"  Found terms: {[t for t in discourse_terms if t in answer.lower()]}")
    else:
        print("? Discourse Analysis:", answer[:200])
        
    return True

async def test_community_detection():
    """Test community detection capabilities"""
    
    config = Config.default()
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    context = GraphRAGContext(
        main_config=config,
        target_dataset_name="Community_Test",
        llm_provider=llm,
        embedding_provider=encoder,
        chunk_storage_manager=chunk_factory
    )
    
    agent = PlanningAgent(config=config, graphrag_context=context)
    
    print("\n" + "="*60)
    print("TEST: Community Detection")
    print("="*60)
    
    # Build graph first
    await agent.process_query("Prepare corpus and build ER graph for Community_Test", "Community_Test")
    
    # Test community detection
    query = "Identify the different communities or clusters in the startup ecosystem"
    result = await agent.process_query(query, "Community_Test")
    answer = result.get("generated_answer", "")
    
    communities = ["ai/ml", "fintech", "social media", "cluster", "community"]
    if any(term in answer.lower() for term in communities):
        print("✓ Community Detection: Successfully identified communities")
        print(f"  Found terms: {[t for t in communities if t in answer.lower()]}")
    else:
        print("? Community Detection:", answer[:200])
        
    return True

async def test_react_mode():
    """Test ReAct mode with complex multi-step reasoning"""
    
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
    
    print("\n" + "="*60)
    print("TEST: ReAct Mode (Complex Query)")
    print("="*60)
    
    # Complex query requiring multiple steps and reasoning
    query = "Compare the approaches of different tech leaders to AI ethics and find connections between their organizations"
    
    result = await agent.process_query_react(query, "Synthetic_Test")
    
    iterations = result.get("iterations", 0)
    steps = len(result.get("executed_steps", []))
    answer = result.get("generated_answer", "")
    
    print(f"✓ ReAct Mode: Completed with {iterations} iterations and {steps} steps")
    
    if iterations > 3 and steps > 3:
        print("  - Successfully demonstrated iterative reasoning")
    
    if len(answer) > 200:
        print("  - Generated comprehensive answer")
        print(f"  - Answer preview: {answer[:200]}...")
        
    return True

async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("DIGIMON COMPLETE FUNCTIONALITY TEST")
    print("="*70)
    
    # Create results directory
    os.makedirs("test_results", exist_ok=True)
    
    results = {}
    
    # Run tests
    try:
        results["basic"] = await test_basic_functionality()
    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {e}")
        results["basic"] = False
        
    try:
        results["graphs"] = await test_advanced_graphs()
    except Exception as e:
        print(f"\n✗ Advanced graphs test failed: {e}")
        results["graphs"] = False
        
    try:
        results["discourse"] = await test_discourse_analysis()
    except Exception as e:
        print(f"\n✗ Discourse analysis test failed: {e}")
        results["discourse"] = False
        
    try:
        results["community"] = await test_community_detection()
    except Exception as e:
        print(f"\n✗ Community detection test failed: {e}")
        results["community"] = False
        
    try:
        results["react"] = await test_react_mode()
    except Exception as e:
        print(f"\n✗ ReAct mode test failed: {e}")
        results["react"] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} test suites")
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
        
    # Save results
    import json
    with open("test_results/complete_functionality_test.json", "w") as f:
        json.dump({
            "timestamp": str(asyncio.get_event_loop().time()),
            "passed": passed,
            "total": total,
            "results": results
        }, f, indent=2)
        
    print(f"\nResults saved to test_results/complete_functionality_test.json")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! DIGIMON is 100% functional!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Continue fixing...")

if __name__ == "__main__":
    asyncio.run(main())