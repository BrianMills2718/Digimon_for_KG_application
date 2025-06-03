#!/usr/bin/env python3
"""
Direct GraphRAG Test - Test core GraphRAG functionality without dependencies
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append('/home/brian/digimon_cc')

# Test minimal config parsing first
def test_config_parsing():
    """Test basic config parsing"""
    print("🔧 Testing Config Parsing...")
    try:
        from Option.Config2 import Config
        
        method_file = Path("Option/Method/LGraphRAG.yaml")
        if not method_file.exists():
            print(f"❌ Method file not found: {method_file}")
            return False
        
        config_options = Config.parse(
            method_file,
            dataset_name="MySampleTexts",
            exp_name="MySampleTexts"
        )
        
        print("✅ Config parsing successful")
        print(f"   - Dataset: {config_options.dataset_name}")
        print(f"   - Exp name: {config_options.exp_name}")
        return True, config_options
        
    except Exception as e:
        print(f"❌ Config parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_graphrag_creation():
    """Test GraphRAG instance creation"""
    print("\n🏗️ Testing GraphRAG Creation...")
    
    try:
        from Core.GraphRAG import GraphRAG
        print("✅ GraphRAG import successful")
        
        # Get config from previous test
        success, config_options = test_config_parsing()
        if not success:
            return False
        
        # Try to create GraphRAG instance
        graphrag_instance = GraphRAG(config=config_options)
        print("✅ GraphRAG instance created successfully")
        
        return True, graphrag_instance
        
    except Exception as e:
        print(f"❌ GraphRAG creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_build_process():
    """Test the build process directly"""
    print("\n🔨 Testing Build Process...")
    
    try:
        success, graphrag_instance = test_graphrag_creation()
        if not success:
            return False
        
        # Set document path
        docs_path = Path("Data/MySampleTexts")
        if not docs_path.exists():
            print(f"❌ Documents path not found: {docs_path}")
            return False
        
        print(f"✅ Documents path exists: {docs_path}")
        
        # Try build process
        print("🔨 Starting build process...")
        build_result = asyncio.run(graphrag_instance.build_and_persist_artifacts(str(docs_path)))
        
        if isinstance(build_result, dict) and "error" in build_result:
            print(f"❌ Build failed: {build_result['error']}")
            return False
        
        print("✅ Build process completed")
        return True
        
    except Exception as e:
        print(f"❌ Build process failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_query_process():
    """Test the query process directly"""
    print("\n🔍 Testing Query Process...")
    
    try:
        success, graphrag_instance = test_graphrag_creation()
        if not success:
            return False
        
        # Try setup for querying
        print("🔧 Setting up for querying...")
        setup_success = await graphrag_instance.setup_for_querying()
        
        if not setup_success:
            print("❌ Setup for querying failed")
            return False
        
        print("✅ Setup for querying successful")
        
        # Try query
        test_query = "who were some main people in the american revolution"
        print(f"🔍 Testing query: {test_query}")
        
        answer = await graphrag_instance.query(test_query)
        
        if answer and len(str(answer).strip()) > 0:
            print("✅ Query successful")
            print(f"📝 Answer preview: {str(answer)[:200]}...")
            return True
        else:
            print("❌ Query returned empty answer")
            return False
        
    except Exception as e:
        print(f"❌ Query process failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run direct GraphRAG tests"""
    print("🚀 Direct GraphRAG Testing")
    print("=" * 50)
    
    os.chdir('/home/brian/digimon_cc')
    
    # Test progression
    tests = [
        ("Config Parsing", test_config_parsing),
        ("GraphRAG Creation", test_graphrag_creation),
        ("Build Process", test_build_process),
        ("Query Process", lambda: asyncio.run(test_query_process())),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_name == "Query Process":
                result = test_func()
            else:
                result = test_func()
                if isinstance(result, tuple):
                    result = result[0]  # Just get the success boolean
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 DIRECT TEST SUMMARY")
    print(f"{'='*50}")
    
    for test_name, result in results.items():
        print(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
    
    working_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    if working_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("GraphRAG core functionality is working correctly.")
    elif working_tests >= 2:
        print("\n⚠️ PARTIAL SUCCESS")
        print(f"{working_tests}/{total_tests} tests passed. Some functionality is working.")
    else:
        print("\n❌ MAJOR ISSUES")
        print("Core GraphRAG functionality has problems.")

if __name__ == "__main__":
    main()