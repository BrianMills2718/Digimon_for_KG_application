#!/usr/bin/env python3
"""Quick demonstration of DIGIMON-style graph analysis on COVID tweets"""

import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import re
import json

def quick_graph_analysis():
    """Quick demo of graph-based analysis"""
    print("DIGIMON-Style Graph Analysis - Quick Demo")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('COVID-19-conspiracy-theories-tweets.csv')
    print(f"Loaded {len(df)} tweets")
    
    # 1. Build a simple knowledge graph
    print("\n1. BUILDING KNOWLEDGE GRAPH")
    print("-" * 40)
    
    G = nx.Graph()
    
    # Extract entities and add nodes
    conspiracy_types = df['conspiracy_theory'].unique()
    stances = df['label'].unique()
    
    # Add conspiracy type nodes
    for ct in conspiracy_types:
        G.add_node(ct, node_type='conspiracy', color='red')
    
    # Add stance nodes  
    for stance in stances:
        G.add_node(stance, node_type='stance', color='blue')
    
    # Extract key narratives from tweets
    narratives = {
        'microchip': 'Vaccine contains microchips',
        'bioweapon': 'COVID is a bioweapon',
        '5g': '5G causes COVID',
        'gates': 'Bill Gates conspiracy',
        'control': 'Population control',
        'hoax': 'COVID is a hoax'
    }
    
    # Add narrative nodes and find connections
    narrative_counts = defaultdict(lambda: defaultdict(int))
    
    for idx, row in df.iterrows():
        if pd.isna(row['tweet']):
            continue
        tweet = str(row['tweet']).lower()
        ct_type = row['conspiracy_theory']
        stance = row['label']
        
        # Check for narratives
        for key, narrative in narratives.items():
            if key in tweet:
                if narrative not in G:
                    G.add_node(narrative, node_type='narrative', color='green')
                
                # Connect narrative to conspiracy type
                G.add_edge(narrative, ct_type, weight=G.get_edge_data(narrative, ct_type, {}).get('weight', 0) + 1)
                
                # Connect stance to narrative
                G.add_edge(stance, narrative, weight=G.get_edge_data(stance, narrative, {}).get('weight', 0) + 1)
                
                narrative_counts[narrative][ct_type] += 1
    
    # Extract top hashtags
    all_hashtags = []
    for tweet in df['tweet']:
        hashtags = re.findall(r'#(\w+)', str(tweet))
        all_hashtags.extend(hashtags)
    
    top_hashtags = [tag for tag, count in Counter(all_hashtags).most_common(10)]
    
    # Add hashtag nodes and connections
    for tag in top_hashtags:
        hashtag = f"#{tag}"
        G.add_node(hashtag, node_type='hashtag', color='purple')
        
        # Find which conspiracy types use this hashtag
        for idx, row in df[df['tweet'].str.contains(f'#{tag}', case=False, na=False)].iterrows():
            ct_type = row['conspiracy_theory']
            G.add_edge(hashtag, ct_type, weight=G.get_edge_data(hashtag, ct_type, {}).get('weight', 0) + 1)
    
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # 2. Graph Analysis
    print("\n2. GRAPH-BASED INSIGHTS")
    print("-" * 40)
    
    # Find most central nodes
    print("\na) Most Central Entities (by degree):")
    degree_cent = nx.degree_centrality(G)
    for node, cent in sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]:
        node_type = G.nodes[node].get('node_type', 'unknown')
        print(f"  {node} ({node_type}): {cent:.3f}")
    
    # Find influential nodes by PageRank
    print("\nb) Most Influential (by PageRank):")
    pagerank = nx.pagerank(G)
    for node, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]:
        node_type = G.nodes[node].get('node_type', 'unknown')
        print(f"  {node} ({node_type}): {score:.4f}")
    
    # Find communities
    print("\nc) Community Detection:")
    communities = list(nx.community.greedy_modularity_communities(G))
    print(f"Found {len(communities)} communities")
    for i, comm in enumerate(communities[:3]):
        print(f"\nCommunity {i+1} ({len(comm)} members):")
        members = list(comm)[:5]
        for member in members:
            node_type = G.nodes[member].get('node_type', 'unknown')
            print(f"  - {member} ({node_type})")
    
    # Path analysis
    print("\nd) Path Analysis - Narrative to Conspiracy Type:")
    microchip_node = 'Vaccine contains microchips'
    if microchip_node in G:
        # Find shortest paths to conspiracy types
        for ct in conspiracy_types[:3]:
            if nx.has_path(G, microchip_node, ct):
                path = nx.shortest_path(G, microchip_node, ct)
                print(f"\n'{microchip_node}' → '{ct}':")
                print(f"  Path: {' → '.join(path)}")
    
    # Pattern discovery
    print("\ne) Pattern Discovery - Stance-Narrative-Conspiracy:")
    patterns = []
    for stance in stances:
        for narrative in narratives.values():
            if narrative in G and G.has_edge(stance, narrative):
                for ct in conspiracy_types:
                    if G.has_edge(narrative, ct):
                        weight1 = G[stance][narrative]['weight']
                        weight2 = G[narrative][ct]['weight']
                        patterns.append({
                            'stance': stance,
                            'narrative': narrative,
                            'conspiracy': ct,
                            'strength': weight1 * weight2
                        })
    
    top_patterns = sorted(patterns, key=lambda x: x['strength'], reverse=True)[:5]
    print("\nTop patterns:")
    for p in top_patterns:
        print(f"  {p['stance']} → {p['narrative']} → {p['conspiracy']} (strength: {p['strength']})")
    
    # 3. Query Examples
    print("\n3. GRAPH-BASED QUERY EXAMPLES")
    print("-" * 40)
    
    # Query 1: Which narratives are most associated with 'support' stance?
    print("\nQuery 1: Which narratives do supporters believe?")
    support_narratives = []
    if 'support' in G:
        for neighbor in G.neighbors('support'):
            if G.nodes[neighbor].get('node_type') == 'narrative':
                weight = G['support'][neighbor]['weight']
                support_narratives.append((neighbor, weight))
    
    for narrative, weight in sorted(support_narratives, key=lambda x: x[1], reverse=True):
        print(f"  {narrative}: {weight} connections")
    
    # Query 2: Which conspiracy types are connected to multiple narratives?
    print("\nQuery 2: Which conspiracy types have multiple narratives?")
    for ct in conspiracy_types:
        narrative_neighbors = [n for n in G.neighbors(ct) if G.nodes[n].get('node_type') == 'narrative']
        if len(narrative_neighbors) > 1:
            print(f"  {ct}: connected to {len(narrative_neighbors)} narratives")
            for n in narrative_neighbors[:3]:
                print(f"    - {n}")
    
    # Save results
    print("\n" + "=" * 60)
    print("SUMMARY: This demonstrates DIGIMON's graph-based approach:")
    print("- Entities extracted and connected in knowledge graph")
    print("- Graph algorithms reveal hidden patterns")
    print("- Complex queries answered through graph traversal")
    print("- Communities and influence patterns discovered")
    
    # Save graph data
    graph_data = {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'top_influential': [(n, pagerank[n]) for n in sorted(pagerank, key=pagerank.get, reverse=True)[:5]],
        'communities': len(communities),
        'patterns_found': len(patterns)
    }
    
    with open('quick_graph_results.json', 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print("\n✓ Results saved to quick_graph_results.json")

if __name__ == "__main__":
    quick_graph_analysis()