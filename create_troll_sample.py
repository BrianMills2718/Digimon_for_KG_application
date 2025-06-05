#!/usr/bin/env python3
"""Create a small sample of Russian troll tweets for quick DIGIMON testing"""

import pandas as pd
import json
import os
from pathlib import Path

# Create output directory
output_dir = Path("Data/Russian_Troll_Sample")
output_dir.mkdir(parents=True, exist_ok=True)

# Read first 100 tweets from first CSV file
print("Reading sample tweets...")
df = pd.read_csv("russian-troll-tweets/IRAhandle_tweets_1.csv", nrows=100)

# Group tweets by account category for diversity
categories = df['account_category'].value_counts()
print(f"\nAccount categories in sample: {categories.to_dict()}")

# Create text files grouped by category (for better graph structure)
corpus_data = []

for category in df['account_category'].unique():
    if pd.notna(category):
        category_tweets = df[df['account_category'] == category]
        
        # Create a document for each category
        doc_content = f"Category: {category}\n\n"
        doc_content += f"Number of tweets: {len(category_tweets)}\n\n"
        
        for _, tweet in category_tweets.iterrows():
            doc_content += f"Author: @{tweet['author']}\n"
            doc_content += f"Date: {tweet['publish_date']}\n"
            doc_content += f"Tweet: {tweet['content']}\n"
            doc_content += f"Followers: {tweet['followers']}\n"
            doc_content += "-" * 50 + "\n\n"
        
        # Save as text file
        filename = f"{category.replace(' ', '_').lower()}_tweets.txt"
        filepath = output_dir / filename
        filepath.write_text(doc_content, encoding='utf-8')
        
        corpus_data.append({
            "doc_id": filename.replace('.txt', ''),
            "content": doc_content,
            "metadata": {
                "category": category,
                "tweet_count": len(category_tweets),
                "source": "russian-troll-tweets"
            }
        })

# Create Corpus.json
corpus_json = output_dir / "Corpus.json"
with open(corpus_json, 'w', encoding='utf-8') as f:
    json.dump(corpus_data, f, indent=2)

print(f"\nCreated sample dataset in {output_dir}")
print(f"- {len(corpus_data)} documents")
print(f"- Total tweets: {len(df)}")

# Create sample questions
questions = [
    {
        "question": "What are the main categories of Russian troll accounts?",
        "answer": "Based on the sample"
    },
    {
        "question": "What topics did RightTroll accounts focus on?", 
        "answer": "To be determined"
    },
    {
        "question": "How did different troll categories differ in their messaging?",
        "answer": "To be determined"
    }
]

question_file = output_dir / "Question.json"
with open(question_file, 'w', encoding='utf-8') as f:
    json.dump(questions, f, indent=2)

print(f"\nQuick test commands:")
print(f"1. Build graph: python main.py build -opt Option/Method/LGraphRAG.yaml -dataset_name Russian_Troll_Sample")
print(f"2. Query: python main.py query -opt Option/Method/LGraphRAG.yaml -dataset_name Russian_Troll_Sample -question \"What are the main themes in Russian troll tweets?\"")
print(f"3. Or use agent: python digimon_cli.py -c Data/Russian_Troll_Sample -q \"What messaging strategies did Russian trolls use?\"")