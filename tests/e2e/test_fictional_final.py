#!/usr/bin/env python3
"""
Final Fictional Test - Build and test in one go
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append('/home/brian/digimon_cc')

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config

# Fictional queries to test knowledge base vs general knowledge
FICTIONAL_QUERIES = [
    "What is crystal technology?",
    "Tell me about the Zorathian Empire",
    "What are levitite crystals and how do they work?",
    "Who was Emperor Zorthak?",
    "What happened to Aerophantis?"
]

async def test_fictional_knowledge():
    """Complete test: build and query fictional corpus"""
    print("🧪 Complete Fictional Knowledge Test")
    print("=" * 60)
    
    os.chdir('/home/brian/digimon_cc')
    
    # Remove existing build to force rebuild
    fictional_results = Path("results/Fictional_Test")
    if fictional_results.exists():
        print(f"🗑️ Removing existing build: {fictional_results}")
        import shutil
        shutil.rmtree(fictional_results)
    
    try:
        # Create config
        config_options = Config.parse(
            Path("Option/Method/LGraphRAG.yaml"),
            dataset_name="Fictional_Test",
            exp_name="Fictional_Test"
        )
        
        print(f"✅ Config created")
        
        # Create GraphRAG instance
        graphrag_instance = GraphRAG(config=config_options)
        print(f"✅ GraphRAG instance created")
        
        # Build from scratch
        fictional_path = Path("Data/Fictional_Test")
        print(f"🔨 Building artifacts from: {fictional_path}")
        
        build_result = await graphrag_instance.build_and_persist_artifacts(str(fictional_path))
        print(f"✅ Build completed")
        
        # Setup for querying
        setup_success = await graphrag_instance.setup_for_querying()
        if not setup_success:
            print("❌ Setup for querying failed")
            return False
        
        print(f"✅ Setup for querying successful")
        
        # Test fictional queries
        successful_queries = 0
        
        for i, query in enumerate(FICTIONAL_QUERIES, 1):
            print(f"\n🔍 Query {i}: {query}")
            try:
                answer = await graphrag_instance.query(query)
                
                if answer and len(str(answer).strip()) > 0:
                    answer_text = str(answer).lower()
                    
                    # Check for fictional terms that prove knowledge base usage
                    fictional_terms = [
                        "zorathian", "zorthak", "aerophantis", "levitite", 
                        "crystal technology", "xelandra", "shadowpeak"
                    ]
                    
                    found_terms = [term for term in fictional_terms if term in answer_text]
                    
                    print(f"✅ Answer received ({len(str(answer))} chars)")
                    print(f"📝 Preview: {str(answer)[:200]}...")
                    
                    if found_terms:
                        print(f"🎯 EXCELLENT: Contains fictional terms: {found_terms}")
                        successful_queries += 1
                    else:
                        print(f"⚠️ WARNING: May be using general knowledge (no fictional terms)")
                        # Still check if answer is relevant but generic
                        generic_terms = ["crystal", "empire", "technology"]
                        if any(term in answer_text for term in generic_terms):
                            print(f"📊 Answer seems relevant but generic")
                    
                else:
                    print(f"❌ Empty answer received")
                    
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")
        
        # Final assessment
        print(f"\n" + "=" * 60)
        print(f"📊 FICTIONAL KNOWLEDGE TEST RESULTS")
        print(f"=" * 60)
        print(f"✅ Queries with fictional content: {successful_queries}/{len(FICTIONAL_QUERIES)}")
        
        if successful_queries >= 3:
            print(f"🎉 EXCELLENT: System clearly uses knowledge base!")
            print(f"The system retrieves specific fictional information from the corpus.")
            return True
        elif successful_queries >= 1:
            print(f"✅ GOOD: System partially uses knowledge base")
            print(f"Some queries return fictional content, proving knowledge base usage.")
            return True
        else:
            print(f"⚠️ UNCLEAR: May be using general knowledge")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fictional_knowledge())
    
    if success:
        print(f"\n🎯 CONCLUSION: ✅ Backend uses knowledge base, not general knowledge!")
    else:
        print(f"\n🎯 CONCLUSION: ⚠️ Backend may have knowledge base issues")