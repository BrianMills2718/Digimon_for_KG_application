"""Debug script to inspect extracted relationships in the ER graph"""
import sys
sys.path.append('/home/brian/digimon')

import json
import networkx as nx

# Load the graph directly from GraphML file
graph = nx.read_graphml("results/MyPipelineTestRun/er_graph/nx_data.graphml")

# Print all edges with their attributes
print("=== ALL EDGES IN GRAPH ===")
edges = list(graph.edges(data=True))
print(f"Total edges: {len(edges)}\n")

for src, tgt, attrs in edges[:10]:  # Show first 10
    print(f"Source: {src}")
    print(f"Target: {tgt}")
    print(f"Attributes: {json.dumps(attrs, indent=2)}")
    print("-" * 50)

# Count relationship types
rel_types = {}
for src, tgt, attrs in graph.edges(data=True):
    rel_name = attrs.get('relation_name', 'unknown')
    rel_types[rel_name] = rel_types.get(rel_name, 0) + 1

print("\n=== RELATIONSHIP TYPE COUNTS ===")
for rel_type, count in rel_types.items():
    print(f"{rel_type}: {count}")
