#!/usr/bin/env python3
"""Test DIGIMON right now to see the actual error"""

import subprocess
import sys

cmd = [
    "python", "digimon_cli.py",
    "-c", "Data/Russian_Troll_Sample",
    "-q", "Analyze this dataset"
]

print("Testing DIGIMON with improved logging...")
print("Command:", " ".join(cmd))
print("-" * 60)

# Run and capture both stdout and stderr
result = subprocess.run(cmd, capture_output=True, text=True)

# Look for the raw LLM response in the output
for line in result.stdout.split('\n'):
    if "Raw LLM response" in line:
        print(">>> FOUND:", line)
    if "Error" in line and "natural language plan" in line:
        print(">>> ERROR:", line)
    if "JSON decode error" in line:
        print(">>> JSON ERROR:", line)

# Check stderr too
if result.stderr:
    print("\nSTDERR:")
    for line in result.stderr.split('\n'):
        if "Error" in line or "error" in line:
            print(">>>", line)

print("\nExit code:", result.returncode)