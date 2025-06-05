#!/usr/bin/env python3
"""Check status of graphs in Synthetic_Test"""

import os
import sys
sys.path.append('.')

def check_graph_status():
    print("GRAPH STATUS CHECK FOR SYNTHETIC_TEST")
    print("=" * 80)
    
    base_dir = "results/Synthetic_Test"
    
    # Define expected graph directories and their key files
    graph_checks = [
        ("ER Graph", "er_graph", ["nx_data.graphml", "graph_storage/nx_data.graphml"]),
        ("RK Graph", "rkg_graph", ["nx_data.graphml", "graph_storage/nx_data.graphml"]),
        ("Tree Graph", "tree_graph", ["tree_data.pkl", "graph_storage_tree_data.pkl"]),
        ("Balanced Tree Graph", "tree_graph_balanced", ["tree_data.pkl", "graph_storage_tree_data.pkl"]),
        ("Passage Graph", "passage_of_graph", ["nx_data.graphml", "graph_storage/nx_data.graphml"])
    ]
    
    results = {}
    
    for graph_name, dir_name, expected_files in graph_checks:
        graph_dir = os.path.join(base_dir, dir_name)
        exists = os.path.exists(graph_dir)
        has_files = False
        found_files = []
        
        if exists:
            # Check for expected files
            for expected_file in expected_files:
                file_path = os.path.join(graph_dir, expected_file)
                if os.path.exists(file_path):
                    has_files = True
                    found_files.append(expected_file)
            
            # Also list all files in the directory
            all_files = []
            for root, dirs, files in os.walk(graph_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), graph_dir)
                    all_files.append(rel_path)
        
        results[graph_name] = {
            "directory_exists": exists,
            "has_expected_files": has_files,
            "found_files": found_files,
            "all_files": all_files if exists else []
        }
    
    # Print results
    for graph_name, status in results.items():
        print(f"\n{graph_name}:")
        print(f"  Directory exists: {status['directory_exists']}")
        print(f"  Has expected files: {status['has_expected_files']}")
        if status['found_files']:
            print(f"  Found expected files: {', '.join(status['found_files'])}")
        if status['all_files']:
            print(f"  All files in directory:")
            for file in status['all_files']:
                print(f"    - {file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_graphs = len(results)
    built_graphs = sum(1 for status in results.values() if status['has_expected_files'])
    
    print(f"Graphs built: {built_graphs}/{total_graphs}")
    
    for graph_name, status in results.items():
        if status['has_expected_files']:
            print(f"  ✓ {graph_name}")
        else:
            print(f"  ✗ {graph_name}")
    
    return built_graphs

if __name__ == "__main__":
    check_graph_status()