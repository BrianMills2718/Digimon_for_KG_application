#!/usr/bin/env python3
"""Final test after all fixes"""

import subprocess
import time

print("DIGIMON Final Test - Build and Query")
print("=" * 60)

# Simple direct test
cmd = [
    "python", "digimon_cli.py",
    "-c", "Data/Russian_Troll_Sample",
    "-q", "Build an ER graph of the Russian troll tweets and tell me what themes and entities you find"
]

print("Running DIGIMON agent...")
start = time.time()

# Run and capture output
result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
elapsed = time.time() - start

# Extract key information
output = result.stdout

# Check for successful operations
successes = []
failures = []

if "Successfully created Corpus.json" in output:
    successes.append("✓ Corpus preparation")
else:
    failures.append("✗ Corpus preparation")

if "ERGraph built successfully" in output or "graph_id='Russian_Troll_Sample_ERGraph'" in output:
    successes.append("✓ ER Graph building")
else:
    failures.append("✗ ER Graph building")

if "Entity.VDB.Build executed" in output:
    successes.append("✓ Vector DB creation attempted")
    
if "Answer:" in output:
    answer_start = output.find("Answer:")
    answer_end = output.find("Context Retrieved:", answer_start) if "Context Retrieved:" in output else len(output)
    answer = output[answer_start:answer_end].strip()
    
print("\nResults:")
for s in successes:
    print(s)
for f in failures:
    print(f)

print(f"\nCompleted in {elapsed:.1f}s")

if "Answer:" in output:
    print("\nAgent's Answer:")
    print("-" * 40)
    print(answer)
    
# Also check for any errors
if "FAILED" in output or "Error" in output:
    print("\nDetected issues:")
    for line in output.split('\n'):
        if "FAILED" in line or "Error" in line:
            print(f"  - {line.strip()[:100]}...")