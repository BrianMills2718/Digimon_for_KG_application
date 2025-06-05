#!/usr/bin/env python3
"""Debug DIGIMON execution to see what's happening"""

import subprocess
import sys

print("DIGIMON Debug Test - Verbose Output")
print("=" * 60)

# Run with direct output streaming to see what's happening
cmd = [
    "python", "digimon_cli.py",
    "-c", "Data/Russian_Troll_Sample", 
    "-q", "First prepare the corpus, then build an ER graph, and tell me what you find"
]

print(f"Command: {' '.join(cmd)}")
print("\nRunning with full output...")
print("-" * 60)

# Run without capturing to see full output
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Stream output line by line
for line in iter(process.stdout.readline, ''):
    if line:
        # Highlight key lines
        if any(keyword in line for keyword in ["FAILED", "Error", "failure", "success", "built", "ReAct"]):
            print(f">>> {line.strip()}")
        else:
            print(line.strip())

process.wait()
print("-" * 60)
print(f"Exit code: {process.returncode}")