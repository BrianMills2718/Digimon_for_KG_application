#!/usr/bin/env python3
"""
Simple Backend Test - Bypass dependency issues and focus on core functionality
"""

import asyncio
import sys
import os
import requests
import json
from pathlib import Path

# Test configurations
WORKING_DATASET = "MySampleTexts"
WORKING_METHOD = "LGraphRAG"
TEST_QUERY = "who were some main people in the american revolution"
API_BASE_URL = "http://localhost:5000"

def test_api_connection():
    """Test basic API connectivity"""
    print("🔗 Testing API Connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/ontology", timeout=10)
        if response.status_code == 200:
            print("✅ API is accessible")
            return True
        else:
            print(f"❌ API returned HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API is not running - start with: python api.py")
        return False
    except Exception as e:
        print(f"❌ API connection error: {e}")
        return False

def test_api_build():
    """Test API build endpoint"""
    print(f"\n🔨 Testing Build: {WORKING_DATASET} + {WORKING_METHOD}")
    try:
        build_data = {
            "datasetName": WORKING_DATASET,
            "selectedMethod": WORKING_METHOD
        }
        response = requests.post(f"{API_BASE_URL}/api/build", json=build_data, timeout=300)
        
        if response.status_code == 200:
            print("✅ Build completed successfully")
            return True
        else:
            print(f"❌ Build failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

def test_api_query():
    """Test API query endpoint"""
    print(f"\n🔍 Testing Query: {TEST_QUERY}")
    try:
        query_data = {
            "datasetName": WORKING_DATASET,
            "selectedMethod": WORKING_METHOD,
            "question": TEST_QUERY
        }
        response = requests.post(f"{API_BASE_URL}/api/query", json=query_data, timeout=120)
        
        if response.status_code == 200:
            response_json = response.json()
            if "answer" in response_json and response_json["answer"]:
                print("✅ Query successful")
                print(f"📝 Answer preview: {str(response_json['answer'])[:200]}...")
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

def check_file_structure():
    """Check basic file structure"""
    print("\n📂 Checking File Structure...")
    
    checks = {
        "Dataset exists": Path(f"Data/{WORKING_DATASET}").exists(),
        "Method config exists": Path(f"Option/Method/{WORKING_METHOD}.yaml").exists(),
        "API file exists": Path("api.py").exists(),
        "CLI file exists": Path("digimon_cli.py").exists(),
    }
    
    all_good = True
    for check, result in checks.items():
        print(f"{'✅' if result else '❌'} {check}")
        if not result:
            all_good = False
    
    return all_good

def check_existing_builds():
    """Check for existing builds"""
    print(f"\n🏗️ Checking Existing Builds for {WORKING_DATASET}...")
    
    results_dir = Path("results")
    dataset_dir = results_dir / WORKING_DATASET
    
    if not dataset_dir.exists():
        print(f"❌ No results directory for {WORKING_DATASET}")
        return False
    
    method_dir = dataset_dir / WORKING_METHOD
    if not method_dir.exists():
        print(f"❌ No build directory for {WORKING_METHOD}")
        return False
    
    # Check for essential files
    essential_files = [
        "chunk_storage_chunks.pkl",
        "chunk_storage_text_units.pkl",
    ]
    
    # Check for graph files
    graph_files = list(method_dir.rglob("*.graphml"))
    
    files_found = []
    for file_pattern in essential_files:
        if (method_dir / file_pattern).exists():
            files_found.append(file_pattern)
    
    print(f"✅ Found {len(files_found)} essential files: {files_found}")
    print(f"✅ Found {len(graph_files)} graph files")
    
    if len(files_found) > 0 and len(graph_files) > 0:
        print(f"✅ Build appears to exist for {WORKING_DATASET} + {WORKING_METHOD}")
        return True
    else:
        print(f"❌ Build missing or incomplete for {WORKING_DATASET} + {WORKING_METHOD}")
        return False

def main():
    """Run simplified backend tests"""
    print("🚀 DIGIMON Backend Simple Test")
    print("=" * 50)
    
    os.chdir('/home/brian/digimon_cc')
    
    # Test 1: File Structure
    structure_ok = check_file_structure()
    if not structure_ok:
        print("\n❌ Basic file structure issues - cannot proceed")
        return
    
    # Test 2: API Connection
    api_ok = test_api_connection()
    if not api_ok:
        print("\n❌ API not accessible - start with: python api.py")
        return
    
    # Test 3: Check Existing Builds
    build_exists = check_existing_builds()
    
    # Test 4: Try Query (if build exists) or Build+Query
    if build_exists:
        print(f"\n✅ Build exists - testing query directly")
        query_ok, result = test_api_query()
    else:
        print(f"\n⚠️ No build found - trying build first, then query")
        build_ok = test_api_build()
        if build_ok:
            query_ok, result = test_api_query()
        else:
            query_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SIMPLE TEST SUMMARY")
    print("=" * 50)
    print(f"{'✅' if structure_ok else '❌'} File Structure: {'OK' if structure_ok else 'FAILED'}")
    print(f"{'✅' if api_ok else '❌'} API Connection: {'OK' if api_ok else 'FAILED'}")
    print(f"{'✅' if build_exists else '❌'} Build Exists: {'YES' if build_exists else 'NO'}")
    print(f"{'✅' if query_ok else '❌'} Query Works: {'YES' if query_ok else 'NO'}")
    
    if structure_ok and api_ok and query_ok:
        print("\n🎉 BACKEND IS WORKING!")
        print("The system can successfully query the knowledge base.")
    elif structure_ok and api_ok:
        print("\n⚠️ BACKEND PARTIALLY WORKING")
        print("API is accessible but query/build has issues.")
    else:
        print("\n❌ BACKEND HAS ISSUES")
        print("Basic connectivity or structure problems.")

if __name__ == "__main__":
    main()