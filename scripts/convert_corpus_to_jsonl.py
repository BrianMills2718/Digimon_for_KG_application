#!/usr/bin/env python3
"""Convert text files to JSONL corpus format"""

import json
from pathlib import Path

def convert_to_jsonl():
    corpus_dir = Path("Data/COVID_Conspiracy")
    output_lines = []
    
    # Process each text file
    for doc_id, txt_file in enumerate(sorted(corpus_dir.glob("*.txt"))):
        print(f"Processing {txt_file.name}...")
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create corpus entry
        corpus_entry = {
            "title": txt_file.stem,
            "content": content,
            "doc_id": doc_id
        }
        
        # Add as JSON line
        output_lines.append(json.dumps(corpus_entry))
    
    # Write JSONL file
    corpus_jsonl = corpus_dir / "Corpus.json"
    with open(corpus_jsonl, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nCreated {corpus_jsonl} with {len(output_lines)} documents")

if __name__ == "__main__":
    convert_to_jsonl()