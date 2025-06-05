#!/usr/bin/env python3
"""Check status of graph files without rebuilding"""

import os
import sys
sys.path.append('.')

def check_graph_files():
    dataset_name = "Synthetic_Test"
    
    print("DIGIMON GRAPH FILE STATUS")
    print("=" * 60)
    
    # Check corpus
    corpus_path = f"results/{dataset_name}/corpus/Corpus.json"
    if os.path.exists(corpus_path):
        print(f"✓ Corpus exists: {corpus_path}")
    else:
        print(f"✗ Corpus missing: {corpus_path}")
    
    # Define expected graph files
    graph_checks = [
        ("ER Graph", [
            f"results/{dataset_name}/er_graph/nx_data.graphml",
            f"results/{dataset_name}/er_graph/graph_storage/nx_data.graphml"
        ]),
        ("RK Graph", [
            f"results/{dataset_name}/rkg_graph/nx_data.graphml", 
            f"results/{dataset_name}/rkg_graph/graph_storage/nx_data.graphml",
            f"results/{dataset_name}/rk_graph/nx_data.graphml"
        ]),
        ("Tree Graph", [
            f"results/{dataset_name}/tree_graph/tree_data.pkl",
            f"results/{dataset_name}/tree_graph/tree_data_leaves.pkl",
            f"results/{dataset_name}/tree_graph/graph_storage/tree_data.pkl"
        ]),
        ("Tree Graph Balanced", [
            f"results/{dataset_name}/tree_graph_balanced/tree_data.pkl",
            f"results/{dataset_name}/tree_graph_balanced/tree_data_leaves.pkl",
            f"results/{dataset_name}/tree_graph_balanced/graph_storage/tree_data.pkl"
        ]),
        ("Passage Graph", [
            f"results/{dataset_name}/passage_graph/nx_data.graphml",
            f"results/{dataset_name}/passage_graph/graph_storage/nx_data.graphml",
            f"results/{dataset_name}/passage_of_graph/nx_data.graphml"
        ])
    ]
    
    results = {}
    
    for graph_name, possible_paths in graph_checks:
        found = False
        found_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                found = True
                found_path = path
                break
        
        results[graph_name] = found
        
        if found:
            print(f"\n✓ {graph_name}: Found at {found_path}")
            # Get file size
            size = os.path.getsize(found_path)
            print(f"  Size: {size:,} bytes")
        else:
            print(f"\n✗ {graph_name}: Not found")
            print(f"  Checked paths:")
            for path in possible_paths:
                print(f"    - {path}")
    
    # Check VDB files
    print("\n" + "-" * 60)
    print("VDB Files:")
    
    vdb_paths = [
        f"results/{dataset_name}/er_graph/vdb_entities",
        f"results/{dataset_name}/er_graph/vdb_relationships",
        f"results/{dataset_name}/rkg_graph/vdb_entities",
        f"results/{dataset_name}/vdb_entities",
        f"results/{dataset_name}/vdb_relationships"
    ]
    
    for vdb_path in vdb_paths:
        if os.path.exists(vdb_path):
            print(f"✓ VDB exists: {vdb_path}")
            # Check for index files
            index_files = []
            if os.path.exists(f"{vdb_path}/index.faiss"):
                index_files.append("index.faiss")
            if os.path.exists(f"{vdb_path}/index.pkl"):
                index_files.append("index.pkl")
            if index_files:
                print(f"  Contains: {', '.join(index_files)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Graphs found: {working}/{total}")
    
    for name, status in results.items():
        status_str = "✓" if status else "✗"
        print(f"{status_str} {name}")
    
    percentage = (working / total) * 100
    print(f"\nOverall: {percentage:.0f}% of graph types have files")

if __name__ == "__main__":
    check_graph_files()