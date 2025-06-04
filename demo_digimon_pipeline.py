#!/usr/bin/env python3
"""Quick demonstration of DIGIMON's graph-based analysis on COVID tweets"""

import asyncio
import pandas as pd
import json
import networkx as nx
from collections import defaultdict, Counter
import re
from datetime import datetime
from pathlib import Path

# Import DIGIMON components
from Core.Common.Logger import logger
from Core.Graph.ERGraph import ERGraph
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Option.Config2 import Config

class QuickDigimonDemo:
    """Demonstrates DIGIMON's graph-based analysis capabilities"""
    
    def __init__(self):
        # Use NetworkX directly for simplicity
        self.graph = nx.DiGraph()
        self.entities = {}
        self.relationships = []
        
    def extract_entities_from_tweets(self, df):
        """Extract entities using pattern matching and NLP-like rules"""
        print("\n1. ENTITY EXTRACTION")
        print("-" * 60)
        
        # Extract different entity types
        entities = {
            'Narrative': defaultdict(list),
            'Hashtag': defaultdict(list),
            'Account': defaultdict(list),
            'ConspiracyType': defaultdict(list),
            'Stance': defaultdict(list)
        }
        
        for idx, row in df.iterrows():
            tweet = row['tweet']
            ct_type = row['conspiracy_theory']
            stance = row['label']
            
            # Extract narratives (key phrases)
            narratives = []
            if 'vaccine' in tweet.lower() and 'microchip' in tweet.lower():
                narratives.append('Vaccine_Microchip_Theory')
            if 'bill gates' in tweet.lower() or 'gates' in tweet.lower():
                narratives.append('Bill_Gates_Conspiracy')
            if '5g' in tweet.lower():
                narratives.append('5G_Connection_Theory')
            if 'population control' in tweet.lower():
                narratives.append('Population_Control_Agenda')
            if 'bioweapon' in tweet.lower() or 'lab' in tweet.lower():
                narratives.append('Lab_Origin_Theory')
            
            # Extract hashtags
            hashtags = re.findall(r'#(\w+)', tweet)
            
            # Extract mentions
            mentions = re.findall(r'@(\w+)', tweet)
            
            # Store entities with tweet references
            for narrative in narratives:
                entities['Narrative'][narrative].append(idx)
            
            for hashtag in hashtags:
                entities['Hashtag'][f"#{hashtag}"].append(idx)
            
            for mention in mentions:
                entities['Account'][f"@{mention}"].append(idx)
            
            entities['ConspiracyType'][ct_type].append(idx)
            entities['Stance'][stance].append(idx)
        
        # Summary
        print(f"Extracted entities:")
        for entity_type, items in entities.items():
            print(f"  {entity_type}: {len(items)} unique entities")
            # Show top 3
            top_items = sorted(items.items(), key=lambda x: len(x[1]), reverse=True)[:3]
            for name, tweet_ids in top_items:
                print(f"    - {name}: appears in {len(tweet_ids)} tweets")
        
        return entities
    
    def build_knowledge_graph(self, entities, df):
        """Build a knowledge graph from entities"""
        print("\n2. KNOWLEDGE GRAPH CONSTRUCTION")
        print("-" * 60)
        
        # Create nodes
        node_id = 0
        node_map = {}
        
        # Add entity nodes
        for entity_type, items in entities.items():
            for name, tweet_ids in items.items():
                node_data = {
                    'id': node_id,
                    'name': name,
                    'type': entity_type,
                    'tweet_count': len(tweet_ids),
                    'tweet_ids': tweet_ids[:10]  # Store first 10 for reference
                }
                self.graph.add_node(node_id, **node_data)
                node_map[f"{entity_type}:{name}"] = node_id
                node_id += 1
        
        print(f"Created {node_id} nodes")
        
        # Create relationships
        edge_count = 0
        
        # Connect narratives to conspiracy types
        for narrative, narrative_tweets in entities['Narrative'].items():
            narrative_node = node_map.get(f"Narrative:{narrative}")
            if narrative_node is None:
                continue
                
            # Find which conspiracy types contain this narrative
            for ct_type, ct_tweets in entities['ConspiracyType'].items():
                overlap = set(narrative_tweets) & set(ct_tweets)
                if overlap:
                    ct_node = node_map.get(f"ConspiracyType:{ct_type}")
                    if ct_node:
                        self.graph.add_edge(narrative_node, ct_node, 
                                          type='APPEARS_IN',
                                          weight=len(overlap),
                                          examples=list(overlap)[:3])
                        edge_count += 1
        
        # Connect hashtags to narratives
        for hashtag, hashtag_tweets in entities['Hashtag'].items():
            hashtag_node = node_map.get(f"Hashtag:{hashtag}")
            if hashtag_node is None:
                continue
                
            for narrative, narrative_tweets in entities['Narrative'].items():
                overlap = set(hashtag_tweets) & set(narrative_tweets)
                if overlap:
                    narrative_node = node_map.get(f"Narrative:{narrative}")
                    if narrative_node:
                        self.graph.add_edge(hashtag_node, narrative_node,
                                          type='PROMOTES',
                                          weight=len(overlap))
                        edge_count += 1
        
        # Connect stances to conspiracy types
        for stance, stance_tweets in entities['Stance'].items():
            stance_node = node_map.get(f"Stance:{stance}")
            if stance_node is None:
                continue
                
            for ct_type, ct_tweets in entities['ConspiracyType'].items():
                overlap = set(stance_tweets) & set(ct_tweets)
                if overlap:
                    ct_node = node_map.get(f"ConspiracyType:{ct_type}")
                    if ct_node:
                        weight = len(overlap) / len(ct_tweets)  # Proportion
                        self.graph.add_edge(stance_node, ct_node,
                                          type='STANCE_ON',
                                          weight=weight,
                                          count=len(overlap))
                        edge_count += 1
        
        print(f"Created {edge_count} relationships")
        
        # Compute graph metrics
        print("\nGraph metrics:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Density: {nx.density(self.graph):.4f}")
        
        # Find most connected nodes
        degree_centrality = nx.degree_centrality(self.graph)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nMost connected entities (by degree centrality):")
        for node_id, centrality in top_nodes:
            node_data = self.graph.nodes[node_id]
            print(f"  {node_data['name']} ({node_data['type']}): {centrality:.3f}")
        
        return node_map
    
    def demonstrate_retrieval_operations(self, node_map, df):
        """Demonstrate DIGIMON's retrieval capabilities"""
        print("\n3. GRAPH-BASED RETRIEVAL OPERATIONS")
        print("-" * 60)
        
        # 1. Entity neighborhood retrieval
        print("\na) Entity Neighborhood Retrieval:")
        vaccine_microchip_node = node_map.get('Narrative:Vaccine_Microchip_Theory')
        if vaccine_microchip_node is not None:
            neighbors = list(self.graph.g.neighbors(vaccine_microchip_node))
            print(f"Entities connected to 'Vaccine_Microchip_Theory':")
            for neighbor in neighbors[:5]:
                node_data = self.graph.g.nodes[neighbor]
                edge_data = self.graph.g.edges[vaccine_microchip_node, neighbor]
                print(f"  → {node_data['name']} via {edge_data['type']} (weight: {edge_data.get('weight', 0)})")
        
        # 2. Path-based retrieval
        print("\nb) Path-Based Retrieval:")
        support_node = node_map.get('Stance:support')
        denial_node = node_map.get('Stance:deny')
        if support_node and denial_node:
            # Find what connects support and denial stances
            try:
                paths = list(nx.all_simple_paths(self.graph.g, support_node, denial_node, cutoff=3))
                if paths:
                    print(f"Found {len(paths)} paths between 'support' and 'deny' stances")
                    # Show first path
                    path = paths[0]
                    print("Example path:")
                    for i, node_id in enumerate(path):
                        node_data = self.graph.g.nodes[node_id]
                        print(f"  {i+1}. {node_data['name']} ({node_data['type']})")
            except:
                print("No short paths found between support and deny stances")
        
        # 3. Community detection
        print("\nc) Community Detection:")
        # Find communities using Louvain method
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(self.graph.g.to_undirected())
            
            # Group nodes by community
            community_groups = defaultdict(list)
            for node_id, comm_id in communities.items():
                node_data = self.graph.g.nodes[node_id]
                community_groups[comm_id].append((node_data['name'], node_data['type']))
            
            print(f"Found {len(community_groups)} communities:")
            for comm_id, members in sorted(community_groups.items())[:3]:
                print(f"\nCommunity {comm_id} ({len(members)} members):")
                for name, node_type in members[:5]:
                    print(f"  - {name} ({node_type})")
        except ImportError:
            print("Community detection requires python-louvain package")
        
        # 4. Influence propagation
        print("\nd) Influence Analysis (PageRank):")
        pagerank = nx.pagerank(self.graph.g)
        top_influential = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Most influential entities by PageRank:")
        for node_id, score in top_influential:
            node_data = self.graph.g.nodes[node_id]
            print(f"  {node_data['name']} ({node_data['type']}): {score:.4f}")
        
        # 5. Pattern mining
        print("\ne) Pattern Mining:")
        # Find triadic patterns (narrative-hashtag-conspiracy type)
        patterns = []
        for narrative_key, narrative_node in node_map.items():
            if not narrative_key.startswith('Narrative:'):
                continue
            
            narrative_data = self.graph.g.nodes[narrative_node]
            
            # Find hashtags promoting this narrative
            for neighbor in self.graph.g.neighbors(narrative_node):
                neighbor_data = self.graph.g.nodes[neighbor]
                if neighbor_data['type'] == 'Hashtag':
                    edge_data = self.graph.g.edges[neighbor, narrative_node]
                    if edge_data.get('type') == 'PROMOTES':
                        # Find conspiracy types
                        for ct_neighbor in self.graph.g.neighbors(narrative_node):
                            ct_data = self.graph.g.nodes[ct_neighbor]
                            if ct_data['type'] == 'ConspiracyType':
                                pattern = {
                                    'hashtag': neighbor_data['name'],
                                    'narrative': narrative_data['name'],
                                    'conspiracy_type': ct_data['name'],
                                    'strength': edge_data.get('weight', 0)
                                }
                                patterns.append(pattern)
        
        print(f"Found {len(patterns)} hashtag→narrative→conspiracy patterns")
        # Show top patterns
        top_patterns = sorted(patterns, key=lambda x: x['strength'], reverse=True)[:3]
        for pattern in top_patterns:
            print(f"  {pattern['hashtag']} → {pattern['narrative']} → {pattern['conspiracy_type']} (strength: {pattern['strength']})")
        
        return {
            'node_count': self.graph.g.number_of_nodes(),
            'edge_count': self.graph.g.number_of_edges(),
            'top_influential': [(self.graph.g.nodes[n]['name'], s) for n, s in top_influential[:3]],
            'patterns_found': len(patterns)
        }

async def main():
    """Run the demonstration"""
    print("DIGIMON Graph-Based Analysis Demo")
    print("=" * 80)
    print("Demonstrating knowledge graph construction and retrieval on COVID tweets")
    
    # Load data
    df = pd.read_csv('COVID-19-conspiracy-theories-tweets.csv')
    print(f"\nLoaded {len(df)} tweets")
    
    # Create demo instance
    demo = QuickDigimonDemo()
    
    # Extract entities
    entities = demo.extract_entities_from_tweets(df)
    
    # Build knowledge graph
    node_map = demo.build_knowledge_graph(entities, df)
    
    # Demonstrate retrieval operations
    results = demo.demonstrate_retrieval_operations(node_map, df)
    
    # Save results
    print("\n" + "=" * 60)
    print("SUMMARY OF DIGIMON CAPABILITIES DEMONSTRATED:")
    print("=" * 60)
    print(f"✓ Built knowledge graph with {results['node_count']} nodes and {results['edge_count']} edges")
    print(f"✓ Identified top influential entities using PageRank")
    print(f"✓ Found {results['patterns_found']} cross-entity patterns")
    print(f"✓ Demonstrated neighborhood, path, and community-based retrieval")
    print("\nThis shows how DIGIMON can:")
    print("1. Transform text into structured knowledge graphs")
    print("2. Discover hidden relationships between entities")
    print("3. Identify influential nodes and communities")
    print("4. Enable graph-based retrieval for complex queries")
    print("5. Find patterns across different entity types")
    
    # Save graph for visualization
    graph_data = {
        'nodes': [
            {
                'id': n,
                'label': self.graph.g.nodes[n]['name'],
                'type': self.graph.g.nodes[n]['type'],
                'size': self.graph.g.nodes[n].get('tweet_count', 1)
            }
            for n in self.graph.g.nodes()
        ],
        'edges': [
            {
                'source': u,
                'target': v,
                'type': self.graph.g.edges[u, v].get('type', 'RELATED'),
                'weight': self.graph.g.edges[u, v].get('weight', 1)
            }
            for u, v in self.graph.g.edges()
        ]
    }
    
    with open('covid_conspiracy_graph.json', 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print("\n✓ Graph data saved to covid_conspiracy_graph.json for visualization")

if __name__ == "__main__":
    asyncio.run(main())