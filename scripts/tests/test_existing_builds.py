#!/usr/bin/env python3
"""
Test Existing Builds - Work with pre-built artifacts to bypass dependency issues
"""

import asyncio
import sys
import os
import requests
from pathlib import Path

# Test configurations
WORKING_DATASET = "MySampleTexts"
WORKING_METHOD = "LGraphRAG"
TEST_QUERY = "who were some main people in the american revolution"
API_BASE_URL = "http://localhost:5000"

def check_existing_artifacts():
    """Check what artifacts exist in the results directory"""
    print("🔍 Checking Existing Artifacts...")
    
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ No results directory found")
        return False, []
    
    found_builds = []
    
    for dataset_dir in results_dir.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            print(f"\n📂 Dataset: {dataset_name}")
            
            for method_dir in dataset_dir.iterdir():
                if method_dir.is_dir():
                    method_name = method_dir.name
                    
                    # Check for essential files
                    chunk_files = list(method_dir.glob("chunk_storage_*.pkl"))
                    graph_files = list(method_dir.rglob("*.graphml"))
                    
                    if chunk_files and graph_files:
                        print(f"  ✅ {method_name}: Complete build")
                        print(f"     - Chunks: {len(chunk_files)} files")
                        print(f"     - Graphs: {len(graph_files)} files")
                        found_builds.append((dataset_name, method_name))
                    else:
                        print(f"  ❌ {method_name}: Incomplete build")
                        print(f"     - Chunks: {len(chunk_files)} files")
                        print(f"     - Graphs: {len(graph_files)} files")
    
    return len(found_builds) > 0, found_builds

def test_api_query_only():
    """Test API query endpoint with existing build"""
    print(f"\n🔍 Testing API Query: {WORKING_DATASET} + {WORKING_METHOD}")
    
    try:
        query_data = {
            "datasetName": WORKING_DATASET,
            "selectedMethod": WORKING_METHOD,
            "question": TEST_QUERY
        }
        
        print(f"📡 Sending query to API...")
        response = requests.post(f"{API_BASE_URL}/api/query", json=query_data, timeout=120)
        
        if response.status_code == 200:
            response_json = response.json()
            if "answer" in response_json and response_json["answer"]:
                print("✅ Query successful!")
                print(f"📝 Answer preview: {str(response_json['answer'])[:300]}...")
                return True, response_json
            else:
                print("❌ Query returned empty answer")
                print(f"Response: {response_json}")
                return False, response_json
        else:
            print(f"❌ Query failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False, None

def test_other_api_endpoints():
    """Test other API endpoints that might work"""
    print(f"\n🔗 Testing Other API Endpoints...")
    
    endpoints_to_test = [
        ("GET", "/api/ontology", None, "Ontology endpoint"),
    ]
    
    results = {}
    
    for method, endpoint, data, description in endpoints_to_test:
        try:
            print(f"🔍 Testing {description}...")
            
            if method == "GET":
                response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
            else:
                response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {description}: SUCCESS")
                results[description] = True
            else:
                print(f"❌ {description}: HTTP {response.status_code}")
                results[description] = False
                
        except Exception as e:
            print(f"❌ {description}: {str(e)}")
            results[description] = False
    
    return results

def test_different_query_combinations():
    """Test different dataset/method combinations that exist"""
    print(f"\n🔄 Testing Different Combinations...")
    
    # Check what builds exist and test those
    builds_exist, found_builds = check_existing_artifacts()
    
    if not builds_exist:
        print("❌ No complete builds found to test")
        return []
    
    working_combinations = []
    
    for dataset, method in found_builds[:3]:  # Test first 3 combinations
        print(f"\n🔍 Testing: {dataset} + {method}")
        
        try:
            query_data = {
                "datasetName": dataset,
                "selectedMethod": method,
                "question": TEST_QUERY
            }
            
            response = requests.post(f"{API_BASE_URL}/api/query", json=query_data, timeout=60)
            
            if response.status_code == 200:
                response_json = response.json()
                if "answer" in response_json and response_json["answer"]:
                    print(f"✅ {dataset} + {method}: SUCCESS")
                    working_combinations.append((dataset, method, len(str(response_json["answer"]))))
                else:
                    print(f"❌ {dataset} + {method}: Empty answer")
            else:
                print(f"❌ {dataset} + {method}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {dataset} + {method}: {str(e)}")
    
    return working_combinations

def main():
    """Test existing builds without dependency issues"""
    print("🚀 Testing Existing Builds")
    print("=" * 50)
    
    os.chdir('/home/brian/digimon_cc')
    
    # Test 1: Check what exists
    builds_exist, found_builds = check_existing_artifacts()
    
    if not builds_exist:
        print("\n❌ No complete builds found - cannot test")
        return
    
    print(f"\n✅ Found {len(found_builds)} complete builds")
    
    # Test 2: API basic functionality
    api_results = test_other_api_endpoints()
    api_working = any(api_results.values())
    
    if not api_working:
        print("\n❌ API not working - cannot test queries")
        return
    
    # Test 3: Query with primary combination
    primary_query_ok, result = test_api_query_only()
    
    # Test 4: Try other combinations
    working_combinations = test_different_query_combinations()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 EXISTING BUILDS TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Builds Available: {len(found_builds)}")
    print(f"{'✅' if api_working else '❌'} API Accessible: {'YES' if api_working else 'NO'}")
    print(f"{'✅' if primary_query_ok else '❌'} Primary Query: {'SUCCESS' if primary_query_ok else 'FAILED'}")
    print(f"✅ Working Combinations: {len(working_combinations)}")
    
    if working_combinations:
        print("\n🎯 WORKING COMBINATIONS:")
        for dataset, method, answer_length in working_combinations:
            print(f"  - {dataset} + {method} (Answer: {answer_length} chars)")
        
        best = working_combinations[0]
        print(f"\n🏆 RECOMMENDED: {best[0]} + {best[1]}")
        print("The backend has working functionality with existing builds!")
        
    elif primary_query_ok:
        print(f"\n🎯 PRIMARY COMBINATION WORKS: {WORKING_DATASET} + {WORKING_METHOD}")
        print("Backend is functional for the main use case!")
        
    elif builds_exist and api_working:
        print("\n⚠️ BUILDS EXIST BUT QUERIES FAILING")
        print("API is accessible but query processing has issues.")
        
    else:
        print("\n❌ BACKEND NOT FUNCTIONAL")
        print("Cannot process queries with existing builds.")

if __name__ == "__main__":
    main()