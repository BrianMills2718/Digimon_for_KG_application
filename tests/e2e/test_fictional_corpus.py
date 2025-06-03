#!/usr/bin/env python3
"""
Test Fictional Corpus - Verify system uses knowledge base, not general knowledge
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append('/home/brian/digimon_cc')

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config

# Fictional test queries that require specific knowledge base content
FICTIONAL_QUERIES = [
    "What is crystal technology and how does it work?",
    "Tell me about the Zorathian Empire and their culture",
    "What is the Mystaran connection?",
    "Who are the main characters in the crystal technology story?",
    "What powers does crystal technology provide?"
]

async def test_fictional_corpus():
    """Test queries against fictional corpus to verify knowledge base usage"""
    print("🧪 Testing Fictional Corpus")
    print("=" * 60)
    
    os.chdir('/home/brian/digimon_cc')
    
    # Check if fictional test data exists
    fictional_path = Path("Data/Fictional_Test")
    if not fictional_path.exists():
        print(f"❌ Fictional test data not found at: {fictional_path}")
        return False
    
    print(f"✅ Found fictional test data at: {fictional_path}")
    
    # List the files
    files = list(fictional_path.glob("*.txt"))
    print(f"📁 Files in fictional corpus: {[f.name for f in files]}")
    
    try:
        # Create config for fictional test
        config_options = Config.parse(
            Path("Option/Method/LGraphRAG.yaml"),
            dataset_name="Fictional_Test",
            exp_name="Fictional_Test"
        )
        
        print(f"✅ Config created for Fictional_Test")
        
        # Create GraphRAG instance
        graphrag_instance = GraphRAG(config=config_options)
        print(f"✅ GraphRAG instance created")
        
        # Check if we need to build first
        results_path = Path("results/Fictional_Test")
        if not results_path.exists():
            print(f"🔨 No existing build found, building artifacts first...")
            build_result = await graphrag_instance.build_and_persist_artifacts(str(fictional_path))
            print(f"✅ Build completed")
        else:
            print(f"✅ Using existing build at: {results_path}")
        
        # Setup for querying
        setup_success = await graphrag_instance.setup_for_querying()
        if not setup_success:
            print("❌ Setup for querying failed")
            return False
        
        print(f"✅ Setup for querying successful")
        
        # Test each fictional query
        successful_queries = 0
        for i, query in enumerate(FICTIONAL_QUERIES, 1):
            print(f"\n🔍 Query {i}: {query}")
            try:
                answer = await graphrag_instance.query(query)
                
                if answer and len(str(answer).strip()) > 0:
                    print(f"✅ Answer received ({len(str(answer))} characters)")
                    print(f"📝 Preview: {str(answer)[:200]}...")
                    
                    # Check if answer seems to be from knowledge base vs general knowledge
                    answer_text = str(answer).lower()
                    
                    # Fictional-specific terms that shouldn't be in general knowledge
                    fictional_terms = ["crystal technology", "zorathian", "mystaran"]
                    found_terms = [term for term in fictional_terms if term in answer_text]
                    
                    if found_terms:
                        print(f"🎯 GOOD: Answer contains fictional terms: {found_terms}")
                        successful_queries += 1
                    else:
                        print(f"⚠️ WARNING: Answer may be using general knowledge (no fictional terms found)")
                    
                else:
                    print(f"❌ Empty or no answer received")
                    
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")
        
        # Summary
        print(f"\n" + "=" * 60)
        print(f"📊 FICTIONAL CORPUS TEST SUMMARY")
        print(f"=" * 60)
        print(f"✅ Successful queries with fictional content: {successful_queries}/{len(FICTIONAL_QUERIES)}")
        
        if successful_queries >= len(FICTIONAL_QUERIES) // 2:
            print(f"🎉 SUCCESS: System is using knowledge base content!")
            print(f"The system retrieves information from the fictional corpus,")
            print(f"not from general AI knowledge.")
            return True
        else:
            print(f"⚠️ MIXED RESULTS: Some queries may be using general knowledge")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run fictional corpus test"""
    success = await test_fictional_corpus()
    
    if success:
        print(f"\n🎯 CONCLUSION: Backend properly uses knowledge base content")
    else:
        print(f"\n⚠️ CONCLUSION: Backend may have issues with knowledge base retrieval")

if __name__ == "__main__":
    asyncio.run(main())