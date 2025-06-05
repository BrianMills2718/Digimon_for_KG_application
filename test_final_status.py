#!/usr/bin/env python3
"""Final test of DIGIMON status"""

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

async def test_final_status():
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
    
    print("DIGIMON FINAL STATUS TEST")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Corpus exists
    results["Corpus Prepared"] = os.path.exists("results/Synthetic_Test/corpus/Corpus.json")
    
    # Test 2: ER Graph exists
    results["ER Graph"] = os.path.exists("results/Synthetic_Test/er_graph/nx_data.graphml")
    
    # Test 3: Entity Search
    print("\nTesting entity search...")
    query = "Find entities about AI in Synthetic_Test"
    result = await agent.process_query(query, "Synthetic_Test")
    answer = result.get("generated_answer", "")
    results["Entity Search"] = "sarah chen" in answer.lower() or "openai" in answer.lower()
    
    # Test 4: VDB Operations
    print("\nChecking VDB status...")
    results["VDB Available"] = len(context.list_vdbs()) > 0
    print(f"Available VDBs: {context.list_vdbs()}")
    
    # Test 5: Text Retrieval
    print("\nTesting text retrieval...")
    query = "Show me text about climate change from Synthetic_Test"
    result = await agent.process_query(query, "Synthetic_Test")
    answer = result.get("generated_answer", "")
    results["Text Retrieval"] = "climate" in answer.lower() and len(answer) > 200
    
    # Test 6: Relationship Extraction
    print("\nTesting relationship extraction...")
    query = "What is the relationship between DeepMind and hospitals?"
    result = await agent.process_query(query, "Synthetic_Test")
    answer = result.get("generated_answer", "")
    results["Relationships"] = "collaboration" in answer.lower() or "eye disease" in answer.lower()
    
    # Test 7: ReAct Mode
    print("\nTesting ReAct mode...")
    react_result = await agent.process_query_react(
        "Compare AI and climate themes in Synthetic_Test",
        "Synthetic_Test"
    )
    results["ReAct Mode"] = react_result.get("iterations", 0) >= 2
    
    # Test 8: Complex Multi-step
    print("\nTesting complex query...")
    query = """
    Find all technology companies mentioned in Synthetic_Test.
    Then find people associated with them.
    """
    result = await agent.process_query(query, "Synthetic_Test")
    answer = result.get("generated_answer", "")
    results["Complex Queries"] = (
        ("google" in answer.lower() or "microsoft" in answer.lower()) and
        ("sarah chen" in answer.lower() or "john martinez" in answer.lower())
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL STATUS SUMMARY")
    print("=" * 60)
    
    for capability, working in results.items():
        status = "✓" if working else "✗"
        print(f"{status} {capability}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} capabilities working ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 DIGIMON IS FULLY FUNCTIONAL!")
    elif passed >= total * 0.75:
        print("\n✓ DIGIMON is mostly functional with minor issues")
    else:
        print(f"\n⚠️  {total - passed} core capabilities need fixing")
    
    return passed >= total * 0.75

if __name__ == "__main__":
    success = asyncio.run(test_final_status())
    exit(0 if success else 1)