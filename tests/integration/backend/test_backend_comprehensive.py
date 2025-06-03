#!/usr/bin/env python3
"""
Comprehensive Backend Testing Suite for DIGIMON Agent System

This script tests all available methods and modes to ensure full functionality
before integrating with the frontend.
"""

import asyncio
import sys
import os
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to Python path
sys.path.append('/home/brian/digimon_cc')

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config

# Test Configuration
AVAILABLE_METHODS = [
    "RAPTOR", "HippoRAG", "LightRAG", "LGraphRAG", "GGraphRAG", 
    "KGP", "Dalk", "GR", "ToG", "MedG"
]

AVAILABLE_DATASETS = [
    "MySampleTexts", "Fictional_Test", "HotpotQAsmallest"
]

TEST_QUERIES = [
    "who were some main people in the american revolution",
    "what were the main causes of the french revolution", 
    "tell me about George Washington",
    "what is the Declaration of Independence"
]

# Results tracking
test_results = {}

class TestResult:
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.success = False
        self.error_message = ""
        self.details = {}
        
    def mark_success(self, details: Dict = None):
        self.success = True
        self.details = details or {}
        
    def mark_failure(self, error: str, details: Dict = None):
        self.success = False
        self.error_message = error
        self.details = details or {}

async def test_build_artifacts(dataset: str, method: str) -> TestResult:
    """Test building artifacts for a specific dataset and method combination."""
    test_name = f"BUILD_{dataset}_{method}"
    result = TestResult(test_name)
    
    try:
        print(f"🔨 Testing build: {dataset} + {method}")
        
        # Create config
        config_options = Config.parse(
            Path(f"Option/Method/{method}.yaml"), 
            dataset_name=dataset, 
            exp_name=dataset  # Use dataset name as exp_name
        )
        
        # Create GraphRAG instance
        graphrag_instance = GraphRAG(config=config_options)
        
        # Set document path
        docs_path = Path("Data") / dataset
        
        # Build artifacts
        build_result = await graphrag_instance.build_and_persist_artifacts(str(docs_path))
        
        if isinstance(build_result, dict) and "error" in build_result:
            result.mark_failure(f"Build error: {build_result['error']}")
        else:
            result.mark_success({"build_result": str(build_result)})
            print(f"✅ Build successful: {dataset} + {method}")
            
    except Exception as e:
        result.mark_failure(f"Exception during build: {str(e)}")
        print(f"❌ Build failed: {dataset} + {method} - {str(e)}")
        
    return result

async def test_query_processing(dataset: str, method: str, query: str) -> TestResult:
    """Test query processing for a specific combination."""
    test_name = f"QUERY_{dataset}_{method}"
    result = TestResult(test_name)
    
    try:
        print(f"🔍 Testing query: {dataset} + {method}")
        
        # Create config
        config_options = Config.parse(
            Path(f"Option/Method/{method}.yaml"), 
            dataset_name=dataset, 
            exp_name=dataset
        )
        
        # Create GraphRAG instance
        graphrag_instance = GraphRAG(config=config_options)
        
        # Setup for querying
        setup_success = await graphrag_instance.setup_for_querying()
        if not setup_success:
            result.mark_failure("Failed to setup for querying")
            return result
        
        # Execute query
        answer = await graphrag_instance.query(query)
        
        if answer and len(str(answer).strip()) > 0:
            result.mark_success({
                "query": query,
                "answer_length": len(str(answer)),
                "answer_preview": str(answer)[:200] + "..." if len(str(answer)) > 200 else str(answer)
            })
            print(f"✅ Query successful: {dataset} + {method}")
        else:
            result.mark_failure("Empty or no answer generated")
            
    except Exception as e:
        result.mark_failure(f"Exception during query: {str(e)}")
        print(f"❌ Query failed: {dataset} + {method} - {str(e)}")
        
    return result

async def check_existing_builds() -> Dict[str, List[str]]:
    """Check what builds already exist."""
    print("📊 Checking existing builds...")
    
    existing_builds = {}
    results_dir = Path("results")
    
    if results_dir.exists():
        for dataset_dir in results_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                existing_builds[dataset_name] = []
                
                for method_dir in dataset_dir.iterdir():
                    if method_dir.is_dir():
                        # Check if it has the essential files
                        graph_file = None
                        chunk_files = []
                        vdb_dir = None
                        
                        # Look for graph files
                        for file in method_dir.rglob("*.graphml"):
                            graph_file = file
                            break
                            
                        # Look for chunk files
                        for file in method_dir.glob("chunk_storage_*.pkl"):
                            chunk_files.append(file)
                            
                        # Look for VDB directories
                        for dir in method_dir.iterdir():
                            if dir.is_dir() and "vdb" in dir.name:
                                vdb_dir = dir
                                break
                        
                        if graph_file and chunk_files:
                            method_name = method_dir.name
                            existing_builds[dataset_name].append(method_name)
                            print(f"  ✅ Found: {dataset_name}/{method_name}")
    
    return existing_builds

async def fix_graph_file_locations():
    """Fix known graph file location issues."""
    print("🔧 Fixing graph file locations...")
    
    fixes_applied = []
    results_dir = Path("results")
    
    if results_dir.exists():
        for dataset_dir in results_dir.iterdir():
            if dataset_dir.is_dir():
                for method_dir in dataset_dir.iterdir():
                    if method_dir.is_dir():
                        # Look for incorrectly named graph files
                        graph_storage_file = method_dir / "graph_storage_nx_data.graphml"
                        graph_storage_dir = method_dir / "graph_storage"
                        expected_file = graph_storage_dir / "nx_data.graphml"
                        
                        if graph_storage_file.exists() and not expected_file.exists():
                            # Create directory and move file
                            graph_storage_dir.mkdir(exist_ok=True)
                            graph_storage_file.rename(expected_file)
                            fixes_applied.append(f"{dataset_dir.name}/{method_dir.name}")
                            print(f"  🔧 Fixed: {dataset_dir.name}/{method_dir.name}")
    
    return fixes_applied

async def run_comprehensive_tests():
    """Run comprehensive tests of all backend functionality."""
    
    print("🚀 Starting Comprehensive Backend Test Suite")
    print("=" * 60)
    
    # Check existing builds
    existing_builds = await check_existing_builds()
    
    # Fix graph file locations
    fixes = await fix_graph_file_locations()
    if fixes:
        print(f"Applied fixes to: {', '.join(fixes)}")
    
    # Test each dataset + method combination
    test_combinations = []
    
    for dataset in AVAILABLE_DATASETS:
        # Check if dataset exists
        dataset_path = Path("Data") / dataset
        if not dataset_path.exists():
            print(f"⚠️  Dataset not found: {dataset}")
            continue
            
        for method in AVAILABLE_METHODS:
            # Check if method config exists
            method_config = Path(f"Option/Method/{method}.yaml")
            if not method_config.exists():
                print(f"⚠️  Method config not found: {method}")
                continue
                
            test_combinations.append((dataset, method))
    
    print(f"📋 Testing {len(test_combinations)} combinations")
    print()
    
    # First, test querying with existing builds
    print("🔍 Testing Query Processing (existing builds)")
    print("-" * 40)
    
    query_results = []
    for dataset, method in test_combinations:
        # Check if build exists
        if dataset in existing_builds and any(build for build in existing_builds[dataset]):
            # Try querying with first available query
            test_query = TEST_QUERIES[0]
            result = await test_query_processing(dataset, method, test_query)
            query_results.append(result)
            test_results[result.test_name] = result
    
    print()
    
    # Test building for combinations that don't have builds
    print("🔨 Testing Build Process (missing builds)")
    print("-" * 40)
    
    build_results = []
    builds_to_test = [
        ("MySampleTexts", "LGraphRAG"),  # Known working combination
        ("Fictional_Test", "LGraphRAG"),  # Try with another dataset
        # Add more as needed
    ]
    
    for dataset, method in builds_to_test:
        if dataset in existing_builds and method not in existing_builds.get(dataset, []):
            result = await test_build_artifacts(dataset, method)
            build_results.append(result)
            test_results[result.test_name] = result
    
    print()
    
    # Summary Report
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    successful_queries = [r for r in query_results if r.success]
    failed_queries = [r for r in query_results if not r.success]
    
    successful_builds = [r for r in build_results if r.success]
    failed_builds = [r for r in build_results if not r.success]
    
    print(f"Query Tests: {len(successful_queries)}/{len(query_results)} successful")
    print(f"Build Tests: {len(successful_builds)}/{len(build_results)} successful")
    print()
    
    if successful_queries:
        print("✅ WORKING QUERY COMBINATIONS:")
        for result in successful_queries:
            parts = result.test_name.replace("QUERY_", "").split("_")
            dataset, method = parts[0], "_".join(parts[1:])
            print(f"  - {dataset} + {method}")
        print()
    
    if failed_queries:
        print("❌ FAILED QUERY COMBINATIONS:")
        for result in failed_queries:
            parts = result.test_name.replace("QUERY_", "").split("_")
            dataset, method = parts[0], "_".join(parts[1:])
            print(f"  - {dataset} + {method}: {result.error_message}")
        print()
    
    if successful_builds:
        print("✅ SUCCESSFUL BUILDS:")
        for result in successful_builds:
            parts = result.test_name.replace("BUILD_", "").split("_")
            dataset, method = parts[0], "_".join(parts[1:])
            print(f"  - {dataset} + {method}")
        print()
    
    if failed_builds:
        print("❌ FAILED BUILDS:")
        for result in failed_builds:
            parts = result.test_name.replace("BUILD_", "").split("_")
            dataset, method = parts[0], "_".join(parts[1:])
            print(f"  - {dataset} + {method}: {result.error_message}")
        print()
    
    # Recommendations
    print("🎯 RECOMMENDATIONS:")
    if successful_queries:
        best_combo = successful_queries[0]
        parts = best_combo.test_name.replace("QUERY_", "").split("_")
        dataset, method = parts[0], "_".join(parts[1:])
        print(f"  - Use {dataset} + {method} for frontend demo")
        print(f"  - Answer preview: {best_combo.details.get('answer_preview', 'N/A')}")
    else:
        print("  - No working combinations found. Need to debug build process.")
    
    return test_results

if __name__ == "__main__":
    # Set up environment
    os.chdir('/home/brian/digimon_cc')
    
    # Run tests
    results = asyncio.run(run_comprehensive_tests())
    
    print("\n🏁 Testing complete!")