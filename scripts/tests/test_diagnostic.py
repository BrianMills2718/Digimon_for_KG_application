#!/usr/bin/env python3
"""Diagnostic test to see what's happening with orchestrator outputs"""

import subprocess
import json

# Run a simple corpus prep + graph build
cmd = [
    "python", "digimon_cli.py",
    "-c", "Data/Russian_Troll_Sample",
    "-q", "First prepare the corpus, then build an ER graph"
]

print("Running diagnostic test...")
print("Command:", " ".join(cmd))
print("-" * 60)

result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

# Extract all the relevant log lines
print("\nRelevant log lines:")
for line in result.stdout.split('\n') + result.stderr.split('\n'):
    if any(keyword in line for keyword in [
        "Stored output",
        "Available:",
        "step_1_prepare_corpus",
        "prepared_corpus_name", 
        "corpus_json_path",
        "BuildERGraphInputs",
        "from_step_id",
        "named_output_key"
    ]):
        print(line)

# Also check for the actual error
print("\n\nERROR LINES:")
for line in result.stderr.split('\n'):
    if "ERROR" in line:
        print(line)