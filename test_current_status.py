#!/usr/bin/env python3
"""Test current status of DIGIMON capabilities"""

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

async def main():
    """Test current DIGIMON status"""
    
    print("DIGIMON CURRENT STATUS TEST")
    print("=" * 60)
    print("Testing with Synthetic_Test dataset")
    print("")
    
    # Initialize
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
    
    # Test suite
    tests = {
        "1. Corpus Preparation": False,
        "2. ER Graph Building": False,
        "3. Entity Extraction": False,
        "4. VDB Operations": False,
        "5. Entity Search": False,
        "6. Text Retrieval": False,
        "7. ReAct Mode": False,
        "8. Complex Queries": False
    }
    
    # Check existing artifacts
    print("Checking existing artifacts...")
    if os.path.exists("results/Synthetic_Test/corpus/Corpus.json"):
        tests["1. Corpus Preparation"] = True
        print("✓ Corpus already prepared")
    
    if os.path.exists("results/Synthetic_Test/er_graph/nx_data.graphml"):
        tests["2. ER Graph Building"] = True
        print("✓ ER graph already built")
        
    # Test entity search and retrieval
    print("\nTesting entity search and text retrieval...")
    query = """
    Build entity VDB if needed.
    Search for entities about 'artificial intelligence' or 'AI'.
    Get the text chunks for the top 3 entities found.
    """
    
    result = await agent.process_query(query, "Synthetic_Test")
    answer = result.get("generated_answer", "")
    
    # Check results
    if "dr. sarah chen" in answer.lower() or "openai" in answer.lower() or "google" in answer.lower():
        tests["3. Entity Extraction"] = True
        tests["5. Entity Search"] = True
        print("✓ Entity search working")
    
    if "artificial intelligence" in answer.lower() and len(answer) > 200:
        tests["6. Text Retrieval"] = True
        print("✓ Text retrieval working")
        
    # Check context for VDB
    context_data = result.get("retrieved_context", {})
    for step_id, outputs in context_data.items():
        if "vdb" in step_id.lower() and isinstance(outputs, dict) and "vdb_reference_id" in str(outputs):
            tests["4. VDB Operations"] = True
            print("✓ VDB operations working")
            
    # Test ReAct mode
    print("\nTesting ReAct mode...")
    react_result = await agent.process_query_react(
        "What are the connections between AI researchers and their organizations?",
        "Synthetic_Test"
    )
    
    if react_result.get("iterations", 0) > 2:
        tests["7. ReAct Mode"] = True
        print(f"✓ ReAct mode working ({react_result.get('iterations')} iterations)")
    
    # Test complex query
    print("\nTesting complex multi-step query...")
    complex_query = """
    Find all technology companies mentioned in the corpus.
    Then find all people associated with those companies.
    Finally, describe the relationships between them.
    """
    
    complex_result = await agent.process_query(complex_query, "Synthetic_Test")
    complex_answer = complex_result.get("generated_answer", "")
    
    tech_companies = ["google", "microsoft", "openai", "deepmind", "spacex"]
    people = ["sarah chen", "john martinez", "kevin liu", "elena musk"]
    
    companies_found = sum(1 for company in tech_companies if company in complex_answer.lower())
    people_found = sum(1 for person in people if person in complex_answer.lower())
    
    if companies_found >= 2 and people_found >= 2:
        tests["8. Complex Queries"] = True
        print(f"✓ Complex queries working (found {companies_found} companies, {people_found} people)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working = sum(1 for v in tests.values() if v)
    total = len(tests)
    
    for test_name, passed in tests.items():
        status = "✓" if passed else "✗"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {working}/{total} capabilities working ({working/total*100:.0f}%)")
    
    if working == total:
        print("\n🎉 DIGIMON IS 100% FUNCTIONAL!")
    else:
        print(f"\n⚠️  {total - working} capabilities still need work")
        
        # Specific recommendations
        if not tests["3. Entity Extraction"]:
            print("\nRecommendation: Check entity extraction prompts and NER functionality")
        if not tests["4. VDB Operations"]:
            print("\nRecommendation: Verify FAISS index building and storage")
        if not tests["8. Complex Queries"]:
            print("\nRecommendation: Improve agent's multi-step planning")

if __name__ == "__main__":
    asyncio.run(main())