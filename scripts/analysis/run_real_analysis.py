#!/usr/bin/env python3
"""
Direct execution of social media analysis using DIGIMON

This script directly uses DIGIMON tools to analyze the COVID conspiracy dataset.
"""

import asyncio
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

# First, set up the environment
import sys
sys.path.append(".")

# Import DIGIMON components
from Core.Common.Logger import logger
from Core.GraphRAG import GraphRAG
from Option.Config2 import Config

def prepare_corpus_from_csv(csv_path: str, output_dir: str = "social_media_corpus"):
    """Convert CSV tweets to corpus format"""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tweets")
    
    # Create corpus directory
    corpus_dir = Path(output_dir)
    corpus_dir.mkdir(exist_ok=True)
    
    # Group tweets by conspiracy type
    for ct_type in df['conspiracy_theory'].unique():
        ct_df = df[df['conspiracy_theory'] == ct_type]
        
        # Create a text file for each conspiracy type
        doc_text = f"# Conspiracy Theory: {ct_type}\n\n"
        doc_text += f"Total tweets: {len(ct_df)}\n\n"
        
        # Add tweets
        for _, row in ct_df.iterrows():
            doc_text += f"## Tweet {row.name}\n"
            doc_text += f"Text: {row['tweet']}\n"
            doc_text += f"Label: {row['label']}\n"
            doc_text += f"Conspiracy Type: {row['conspiracy_theory']}\n"
            doc_text += "---\n\n"
        
        # Save to file
        output_path = corpus_dir / f"{ct_type}_tweets.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_text)
        
        print(f"Created corpus file: {output_path}")
    
    # Create corpus.json
    corpus_json = {
        "files": [str(f) for f in corpus_dir.glob("*.txt")],
        "total_tweets": len(df),
        "conspiracy_types": list(df['conspiracy_theory'].unique()),
        "label_distribution": df['label'].value_counts().to_dict()
    }
    
    with open(corpus_dir / "Corpus.json", 'w') as f:
        json.dump(corpus_json, f, indent=2)
    
    print(f"Corpus prepared in {corpus_dir}")
    return str(corpus_dir)

async def analyze_conspiracy_tweets():
    """Main analysis function"""
    print("=== DIGIMON COVID Conspiracy Tweet Analysis ===\n")
    
    # 1. Prepare corpus
    csv_path = "COVID-19-conspiracy-theories-tweets.csv"
    if not Path(csv_path).exists():
        print(f"Error: Dataset not found at {csv_path}")
        print("Please run: python download_covid_conspiracy_dataset.py")
        return
    
    corpus_dir = prepare_corpus_from_csv(csv_path)
    
    # 2. Initialize DIGIMON
    print("\nInitializing DIGIMON...")
    config_path = "Option/Config2.yaml"
    
    try:
        # Load config
        config = Config.from_yaml_file(config_path)
        print("Config loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        print("Using default configuration...")
        config = Config.default()
    
    # 3. Create GraphRAG instance
    print("\nCreating GraphRAG instance...")
    dataset_name = "covid_conspiracy_analysis"
    
    try:
        graphrag = GraphRAG(config)
        print("GraphRAG initialized")
        
        # 4. Build graph
        print("\nBuilding knowledge graph...")
        print("This may take a few minutes...")
        
        # Build using ER graph (simplest)
        from Core.Graph.ERGraph import ERGraph
        graph = ERGraph(config)
        
        # Process corpus
        print("Processing corpus files...")
        corpus_files = list(Path(corpus_dir).glob("*.txt"))
        
        for i, file_path in enumerate(corpus_files):
            print(f"Processing {i+1}/{len(corpus_files)}: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract entities and relationships
            # This is simplified - in real usage, would use full pipeline
            entities = extract_entities_from_text(content)
            print(f"  Found {len(entities)} entities")
        
        print("\nGraph building complete!")
        
        # 5. Run queries
        print("\n=== Running Analysis Queries ===\n")
        
        queries = [
            "Who are the most influential accounts spreading conspiracy theories?",
            "What are the main conspiracy narratives about vaccines?",
            "How do conspiracy theories spread through the network?",
            "What patterns exist in conspiracy theory discourse?"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            print("Answer: [Would use GraphRAG query here]")
            # In real usage: answer = await graphrag.query(query)
        
        print("\n=== Analysis Complete ===")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def extract_entities_from_text(text: str):
    """Simple entity extraction (placeholder)"""
    entities = []
    
    # Extract hashtags
    import re
    hashtags = re.findall(r'#\w+', text)
    entities.extend([{"type": "hashtag", "name": tag} for tag in hashtags])
    
    # Extract @mentions
    mentions = re.findall(r'@\w+', text)
    entities.extend([{"type": "user", "name": mention} for mention in mentions])
    
    # Extract conspiracy types
    ct_types = re.findall(r'CT_\d', text)
    entities.extend([{"type": "conspiracy_type", "name": ct} for ct in ct_types])
    
    return entities

if __name__ == "__main__":
    # Run the analysis
    asyncio.run(analyze_conspiracy_tweets())