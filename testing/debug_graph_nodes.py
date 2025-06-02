import asyncio
import json
import os
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Storage.NameSpace import NameSpace

async def debug_graph_and_vdb():
    """Debug script to check if VDB node_ids match graph node IDs"""
    
    # Load the ER graph
    dataset_name = "MyPipelineTestRun"
    results_dir = "/home/brian/digimon/results"
    
    # Create namespace for graph
    namespace = NameSpace(
        working_dir=results_dir,
        dataset=dataset_name,
        graph_type="er_graph"
    )
    
    # Load graph using NetworkXStorage
    nx_storage = NetworkXStorage(namespace=namespace, nx_graph_type="Graph")
    nx_graph = await nx_storage.load_graph(force_reload=False)
    
    print(f"Graph loaded: {nx_graph is not None}")
    print(f"Number of nodes in graph: {nx_graph.number_of_nodes()}")
    print(f"Number of edges in graph: {nx_graph.number_of_edges()}")
    
    # Get first 10 node IDs from graph
    graph_node_ids = list(nx_graph.nodes())[:10]
    print(f"\nFirst 10 node IDs in graph:")
    for nid in graph_node_ids:
        node_data = nx_graph.nodes[nid]
        print(f"  - {nid}: {node_data.get('entity_name', 'NO_NAME')}")
    
    # Test specific node IDs from VDB search results
    test_node_ids = [
        "93de506c05bf0bc71fa29a19afdc190e",
        "ebe9157deb59a51f8b1a1ed82fdca171", 
        "f716968123abceb69248910ac7ffb1f9",
        "23e227699ee48c16d4c983665ff8224d"
    ]
    
    print(f"\nChecking if VDB node_ids exist in graph:")
    for nid in test_node_ids:
        exists = nx_graph.has_node(nid)
        print(f"  - {nid}: {'EXISTS' if exists else 'NOT FOUND'}")
        
    # Also check with entity names
    print(f"\nChecking by entity names:")
    entity_names = ["american revolution", "the american revolution", "american revolutionary war"]
    for name in entity_names:
        # Find nodes by entity_name attribute
        found = False
        for node_id, data in nx_graph.nodes(data=True):
            if data.get('entity_name', '').lower() == name.lower():
                print(f"  - '{name}' found as node: {node_id}")
                found = True
                break
        if not found:
            print(f"  - '{name}' NOT FOUND")

if __name__ == "__main__":
    asyncio.run(debug_graph_and_vdb())
