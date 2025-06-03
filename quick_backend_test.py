#!/usr/bin/env python3
"""
Quick Backend Test for DIGIMON - Focus on Working Combinations
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.append('/home/brian/digimon_cc')

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config

# Known working combinations from previous test
WORKING_COMBINATIONS = [
    ("MySampleTexts", "LGraphRAG"),
    ("MySampleTexts", "KGP"),
]

# Additional combinations to test based on available builds
ADDITIONAL_TESTS = [
    ("MySampleTexts", "GGraphRAG"),
    ("MySampleTexts", "RAPTOR"),
    ("Fictional_Test", "LGraphRAG"),
]

TEST_QUERY = "who were some main people in the american revolution"

async def quick_test_query(dataset: str, method: str) -> dict:
    """Quick test of a specific combination."""
    try:
        print(f"🔍 Testing: {dataset} + {method}")
        
        config_options = Config.parse(
            Path(f"Option/Method/{method}.yaml"), 
            dataset_name=dataset, 
            exp_name=dataset
        )
        
        graphrag_instance = GraphRAG(config=config_options)
        
        # Setup for querying
        setup_success = await graphrag_instance.setup_for_querying()
        if not setup_success:
            print(f"❌ Setup failed: {dataset} + {method}")
            return {"success": False, "error": "Setup failed"}
        
        # Execute query
        answer = await graphrag_instance.query(TEST_QUERY)
        
        if answer and len(str(answer).strip()) > 0:
            print(f"✅ SUCCESS: {dataset} + {method}")
            return {
                "success": True, 
                "answer_length": len(str(answer)),
                "answer_preview": str(answer)[:150] + "..."
            }
        else:
            print(f"❌ Empty answer: {dataset} + {method}")
            return {"success": False, "error": "Empty answer"}
            
    except Exception as e:
        print(f"❌ Exception: {dataset} + {method} - {str(e)}")
        return {"success": False, "error": str(e)}

async def check_file_structure():
    """Check and fix file structure issues."""
    print("🔧 Checking file structure...")
    
    results_dir = Path("results")
    fixes = []
    
    if results_dir.exists():
        for dataset_dir in results_dir.iterdir():
            if dataset_dir.is_dir():
                for method_dir in dataset_dir.iterdir():
                    if method_dir.is_dir():
                        # Fix graph file location issue
                        graph_storage_file = method_dir / "graph_storage_nx_data.graphml"
                        graph_storage_dir = method_dir / "graph_storage"
                        expected_file = graph_storage_dir / "nx_data.graphml"
                        
                        if graph_storage_file.exists() and not expected_file.exists():
                            graph_storage_dir.mkdir(exist_ok=True)
                            graph_storage_file.rename(expected_file)
                            fixes.append(f"{dataset_dir.name}/{method_dir.name}")
    
    if fixes:
        print(f"Applied fixes to: {', '.join(fixes)}")
    return fixes

async def main():
    print("🚀 Quick Backend Test for DIGIMON")
    print("=" * 50)
    
    # Fix file structure
    await check_file_structure()
    
    # Test combinations
    working_combos = []
    failed_combos = []
    
    all_tests = WORKING_COMBINATIONS + ADDITIONAL_TESTS
    
    for dataset, method in all_tests:
        result = await quick_test_query(dataset, method)
        
        if result["success"]:
            working_combos.append((dataset, method, result))
        else:
            failed_combos.append((dataset, method, result["error"]))
    
    print("\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print("=" * 50)
    
    if working_combos:
        print(f"✅ WORKING COMBINATIONS ({len(working_combos)}):")
        for dataset, method, result in working_combos:
            print(f"  - {dataset} + {method} (Answer: {result['answer_length']} chars)")
        print()
        
        # Show preview of best answer
        best_combo = working_combos[0]
        print("🎯 BEST WORKING COMBINATION:")
        print(f"  Dataset: {best_combo[0]}")
        print(f"  Method: {best_combo[1]}")
        print(f"  Answer Preview: {best_combo[2]['answer_preview']}")
        print()
    
    if failed_combos:
        print(f"❌ FAILED COMBINATIONS ({len(failed_combos)}):")
        for dataset, method, error in failed_combos:
            print(f"  - {dataset} + {method}: {error}")
        print()
    
    print("🎯 FRONTEND RECOMMENDATIONS:")
    if working_combos:
        best = working_combos[0]
        print(f"  - Use {best[0]} + {best[1]} as default")
        print(f"  - Available working methods: {[combo[1] for combo in working_combos]}")
        print(f"  - Available working datasets: {list(set([combo[0] for combo in working_combos]))}")
    else:
        print("  - No working combinations found!")
    
    return working_combos

if __name__ == "__main__":
    os.chdir('/home/brian/digimon_cc')
    working = asyncio.run(main())