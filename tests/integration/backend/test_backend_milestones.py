#!/usr/bin/env python3
"""
Backend Testing Milestones for DIGIMON
Test each component systematically before proceeding to the next
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append('/home/brian/digimon_cc')

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config

# Test configurations
WORKING_DATASET = "MySampleTexts"
WORKING_METHODS = ["LGraphRAG", "KGP"]
TEST_QUERY = "who were some main people in the american revolution"

async def milestone_1_direct_graphrag():
    """
    MILESTONE 1: Test direct GraphRAG functionality
    This tests the core GraphRAG system without API layer
    """
    print("🎯 MILESTONE 1: Testing Direct GraphRAG Functionality")
    print("=" * 60)
    
    results = {}
    
    for method in WORKING_METHODS:
        print(f"\n🔍 Testing {WORKING_DATASET} + {method}")
        try:
            # Create config
            config_options = Config.parse(
                Path(f"Option/Method/{method}.yaml"), 
                dataset_name=WORKING_DATASET, 
                exp_name=WORKING_DATASET
            )
            print(f"✅ Config created for {method}")
            
            # Create GraphRAG instance
            graphrag_instance = GraphRAG(config=config_options)
            print(f"✅ GraphRAG instance created for {method}")
            
            # Setup for querying
            setup_success = await graphrag_instance.setup_for_querying()
            if not setup_success:
                print(f"❌ Setup failed for {method}")
                results[method] = {"success": False, "error": "Setup failed"}
                continue
            print(f"✅ Setup successful for {method}")
            
            # Execute query
            answer = await graphrag_instance.query(TEST_QUERY)
            if answer and len(str(answer).strip()) > 0:
                print(f"✅ Query successful for {method}")
                results[method] = {
                    "success": True, 
                    "answer_length": len(str(answer)),
                    "answer_preview": str(answer)[:200] + "..."
                }
            else:
                print(f"❌ Empty answer for {method}")
                results[method] = {"success": False, "error": "Empty answer"}
                
        except Exception as e:
            print(f"❌ Exception for {method}: {str(e)}")
            results[method] = {"success": False, "error": str(e)}
    
    # Summary
    print(f"\n📊 MILESTONE 1 RESULTS:")
    working_methods = []
    for method, result in results.items():
        if result["success"]:
            print(f"✅ {method}: SUCCESS")
            working_methods.append(method)
        else:
            print(f"❌ {method}: {result['error']}")
    
    milestone_1_passed = len(working_methods) > 0
    print(f"\n🎯 MILESTONE 1: {'PASSED' if milestone_1_passed else 'FAILED'}")
    
    return milestone_1_passed, working_methods

async def milestone_2_cli_commands():
    """
    MILESTONE 2: Test CLI commands
    Test the main CLI interface commands
    """
    print("\n🎯 MILESTONE 2: Testing CLI Commands")
    print("=" * 60)
    
    # Test 1: Basic help command
    print("\n🔍 Testing CLI help command...")
    help_result = os.system("python digimon_cli.py --help > /dev/null 2>&1")
    help_works = help_result == 0
    print(f"✅ CLI help: {'WORKS' if help_works else 'FAILED'}")
    
    # Test 2: List available methods
    print("\n🔍 Testing method listing...")
    try:
        methods_dir = Path("Option/Method")
        available_methods = [f.stem for f in methods_dir.glob("*.yaml")]
        print(f"✅ Found {len(available_methods)} method configs: {available_methods}")
        methods_available = len(available_methods) > 0
    except Exception as e:
        print(f"❌ Error listing methods: {e}")
        methods_available = False
    
    # Test 3: List available datasets
    print("\n🔍 Testing dataset listing...")
    try:
        data_dir = Path("Data")
        available_datasets = [d.name for d in data_dir.iterdir() if d.is_dir()]
        print(f"✅ Found {len(available_datasets)} datasets: {available_datasets}")
        datasets_available = len(available_datasets) > 0
    except Exception as e:
        print(f"❌ Error listing datasets: {e}")
        datasets_available = False
    
    # Test 4: Try a simple CLI query (if CLI supports it)
    print("\n🔍 Testing CLI query capability...")
    cli_query_works = False
    try:
        # Check if CLI supports direct querying
        cli_test_cmd = f'python digimon_cli.py --dataset {WORKING_DATASET} --method {WORKING_METHODS[0]} --query "{TEST_QUERY}" --dry-run'
        print(f"Testing command: {cli_test_cmd}")
        cli_result = os.system(f"{cli_test_cmd} > /dev/null 2>&1")
        cli_query_works = cli_result == 0
        print(f"✅ CLI query: {'WORKS' if cli_query_works else 'NOT SUPPORTED/FAILED'}")
    except Exception as e:
        print(f"❌ CLI query test failed: {e}")
    
    milestone_2_passed = help_works and methods_available and datasets_available
    print(f"\n🎯 MILESTONE 2: {'PASSED' if milestone_2_passed else 'FAILED'}")
    
    return milestone_2_passed

async def milestone_3_api_endpoints():
    """
    MILESTONE 3: Test Flask API endpoints individually
    """
    print("\n🎯 MILESTONE 3: Testing Flask API Endpoints")
    print("=" * 60)
    
    import requests
    import json
    
    base_url = "http://localhost:5000"
    results = {}
    
    # Test 1: Health check / ontology endpoint
    print("\n🔍 Testing ontology endpoint...")
    try:
        response = requests.get(f"{base_url}/api/ontology", timeout=10)
        if response.status_code == 200:
            print("✅ Ontology endpoint: SUCCESS")
            results["ontology"] = True
        else:
            print(f"❌ Ontology endpoint: HTTP {response.status_code}")
            results["ontology"] = False
    except Exception as e:
        print(f"❌ Ontology endpoint: {str(e)}")
        results["ontology"] = False
    
    # Test 2: Build endpoint
    print("\n🔍 Testing build endpoint...")
    try:
        build_data = {
            "datasetName": WORKING_DATASET,
            "selectedMethod": WORKING_METHODS[0]
        }
        response = requests.post(f"{base_url}/api/build", json=build_data, timeout=300)
        if response.status_code == 200:
            print("✅ Build endpoint: SUCCESS")
            results["build"] = True
        else:
            print(f"❌ Build endpoint: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            results["build"] = False
    except Exception as e:
        print(f"❌ Build endpoint: {str(e)}")
        results["build"] = False
    
    # Test 3: Query endpoint
    print("\n🔍 Testing query endpoint...")
    try:
        query_data = {
            "datasetName": WORKING_DATASET,
            "selectedMethod": WORKING_METHODS[0],
            "question": TEST_QUERY
        }
        response = requests.post(f"{base_url}/api/query", json=query_data, timeout=120)
        if response.status_code == 200:
            response_json = response.json()
            if "answer" in response_json:
                print("✅ Query endpoint: SUCCESS")
                print(f"Answer preview: {str(response_json['answer'])[:100]}...")
                results["query"] = True
            else:
                print("❌ Query endpoint: No answer in response")
                results["query"] = False
        else:
            print(f"❌ Query endpoint: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            results["query"] = False
    except Exception as e:
        print(f"❌ Query endpoint: {str(e)}")
        results["query"] = False
    
    milestone_3_passed = all(results.values())
    print(f"\n📊 API Endpoint Results:")
    for endpoint, success in results.items():
        print(f"{'✅' if success else '❌'} {endpoint}: {'SUCCESS' if success else 'FAILED'}")
    
    print(f"\n🎯 MILESTONE 3: {'PASSED' if milestone_3_passed else 'FAILED'}")
    
    return milestone_3_passed, results

async def main():
    """Run all milestones sequentially"""
    print("🚀 DIGIMON Backend Testing - Systematic Milestone Approach")
    print("=" * 80)
    
    os.chdir('/home/brian/digimon_cc')
    
    # Milestone 1: Direct GraphRAG
    m1_passed, working_methods = await milestone_1_direct_graphrag()
    if not m1_passed:
        print("\n❌ MILESTONE 1 FAILED - Cannot proceed with broken GraphRAG core")
        return
    
    # Milestone 2: CLI Commands
    m2_passed = await milestone_2_cli_commands()
    if not m2_passed:
        print("\n⚠️ MILESTONE 2 FAILED - CLI issues detected but proceeding with API tests")
    
    # Milestone 3: API Endpoints
    m3_passed, api_results = await milestone_3_api_endpoints()
    
    # Final Summary
    print("\n" + "=" * 80)
    print("🏁 FINAL MILESTONE SUMMARY")
    print("=" * 80)
    print(f"✅ Milestone 1 (Direct GraphRAG): {'PASSED' if m1_passed else 'FAILED'}")
    print(f"{'✅' if m2_passed else '❌'} Milestone 2 (CLI Commands): {'PASSED' if m2_passed else 'FAILED'}")
    print(f"{'✅' if m3_passed else '❌'} Milestone 3 (API Endpoints): {'PASSED' if m3_passed else 'FAILED'}")
    
    if m1_passed and m3_passed:
        print("\n🎉 BACKEND IS FUNCTIONAL!")
        print(f"Working methods: {working_methods}")
        print("Ready for frontend integration.")
    elif m1_passed:
        print("\n⚠️ BACKEND CORE WORKS - API LAYER NEEDS FIXING")
        print("Direct GraphRAG works, but API has issues.")
    else:
        print("\n❌ BACKEND BROKEN - CORE GRAPHRAG ISSUES")
        print("Need to fix fundamental GraphRAG problems first.")

if __name__ == "__main__":
    asyncio.run(main())