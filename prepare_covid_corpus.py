#!/usr/bin/env python3
"""
Prepare COVID conspiracy tweets for DIGIMON analysis
Converts CSV to the expected corpus format
"""

import pandas as pd
import json
from pathlib import Path

def prepare_corpus():
    """Convert CSV to DIGIMON corpus format"""
    # Load CSV
    csv_path = "COVID-19-conspiracy-theories-tweets.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tweets")
    
    # Create output directory
    output_dir = Path("Data/COVID_Conspiracy")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by conspiracy type and create text files
    corpus_files = []
    
    for ct_type in sorted(df['conspiracy_theory'].unique()):
        ct_df = df[df['conspiracy_theory'] == ct_type]
        
        # Create meaningful filename
        ct_names = {
            'CT_1': 'economic_instability',
            'CT_2': 'public_misinformation',
            'CT_3': 'bioweapon_theory',
            'CT_4': 'government_disinformation',
            'CT_5': 'chinese_intentional_spread',
            'CT_6': 'vaccine_population_control'
        }
        
        filename = f"{ct_names.get(ct_type, ct_type)}.txt"
        filepath = output_dir / filename
        
        # Prepare content
        content = f"# Conspiracy Theory: {ct_type}\n"
        content += f"# Topic: {ct_names.get(ct_type, 'Unknown')}\n"
        content += f"# Number of tweets: {len(ct_df)}\n\n"
        
        # Add tweets grouped by stance
        for label in ['support', 'deny', 'neutral']:
            label_df = ct_df[ct_df['label'] == label]
            if len(label_df) > 0:
                content += f"\n## {label.upper()} ({len(label_df)} tweets)\n\n"
                
                for idx, row in label_df.iterrows():
                    content += f"{row['tweet']}\n\n"
                    # Add some context/metadata as natural text
                    if idx < 10:  # Limit to avoid too large files
                        content += f"[This tweet {label}s the theory that {ct_names.get(ct_type, 'conspiracy theories exist')}]\n\n"
                content += "---\n"
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        corpus_files.append(str(filepath))
        print(f"Created: {filepath} ({len(ct_df)} tweets)")
    
    # Create Corpus.json
    corpus_data = {
        "files": [f.name for f in output_dir.glob("*.txt")],
        "metadata": {
            "total_tweets": len(df),
            "conspiracy_types": list(df['conspiracy_theory'].unique()),
            "label_distribution": df['label'].value_counts().to_dict(),
            "source": "COVID-19-conspiracy-theories-tweets.csv"
        }
    }
    
    corpus_json_path = output_dir / "Corpus.json"
    with open(corpus_json_path, 'w') as f:
        json.dump(corpus_data, f, indent=2)
    
    print(f"\nCorpus.json created at: {corpus_json_path}")
    print(f"Total files created: {len(corpus_files)}")
    
    # Create questions for testing
    questions = [
        "What are the main conspiracy theories about COVID-19?",
        "How do people justify vaccine conspiracy theories?",
        "What arguments are used to deny conspiracy theories?",
        "Which conspiracy theory has the most support?",
        "What is the relationship between economic instability and COVID conspiracies?",
        "How do conspiracy theorists view government responses to COVID?",
        "What evidence do people cite for bioweapon theories?",
        "How are vaccines portrayed in conspiracy discourse?"
    ]
    
    questions_data = {
        "questions": questions,
        "metadata": {
            "created_for": "COVID conspiracy analysis",
            "num_questions": len(questions)
        }
    }
    
    questions_path = output_dir / "Question.json"
    with open(questions_path, 'w') as f:
        json.dump(questions_data, f, indent=2)
    
    print(f"Question.json created at: {questions_path}")
    print("\nCorpus preparation complete!")
    print(f"\nYou can now run:")
    print(f"  python main.py build -opt Option/Method/LGraphRAG.yaml -dataset_name COVID_Conspiracy")
    print(f"  python main.py query -opt Option/Method/LGraphRAG.yaml -dataset_name COVID_Conspiracy")

if __name__ == "__main__":
    prepare_corpus()