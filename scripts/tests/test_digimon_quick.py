#!/usr/bin/env python3
"""Quick test of DIGIMON on Russian troll sample"""

import subprocess
import sys

print("Testing DIGIMON on Russian troll sample dataset...")
print("=" * 60)

# Test 1: Direct query mode (simpler)
print("\n1. Testing direct query mode:")
cmd = [
    "python", "digimon_cli.py",
    "-c", "Data/Russian_Troll_Sample",
    "-q", "What are the main themes in these tweets?"
]
print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)

# Test 2: Check if corpus exists
print("\n2. Checking corpus file:")
import os
corpus_path = "Data/Russian_Troll_Sample/Corpus.json"
if os.path.exists(corpus_path):
    print(f"✓ Corpus exists at {corpus_path}")
    import json
    with open(corpus_path) as f:
        data = json.load(f)
    print(f"  Contains {len(data)} documents")
else:
    print(f"✗ Corpus not found at {corpus_path}")

# Test 3: Alternative - use main.py directly
print("\n3. Alternative test using main.py:")
print("To build graph: python main.py build -opt Option/Method/LGraphRAG.yaml -dataset_name Russian_Troll_Sample")
print("To query: python main.py query -opt Option/Method/LGraphRAG.yaml -dataset_name Russian_Troll_Sample -question 'What themes appear?'")