#!/usr/bin/env python3
"""Test fix #1: Orchestrator state preservation in ReAct mode"""

import subprocess
import time

print("Testing Fix #1: Orchestrator state preservation")
print("=" * 60)

cmd = [
    "python", "digimon_cli.py",
    "-c", "Data/Russian_Troll_Sample",
    "-i", "--react"
]

# Start the process
process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Send a query
query = "Build an ER graph and show me the entities\n"
stdout, stderr = process.communicate(input=query, timeout=120)

# Check for success
print("\nChecking results...")

if "ERGraph built successfully" in stdout:
    print("✓ ER Graph built successfully!")
else:
    print("✗ ER Graph build failed")

if "entity–relationship graph never actually got built" in stdout:
    print("✗ FAIL: Agent still thinks graph wasn't built")
else:
    print("✓ Agent recognizes graph was built")

# Look for entities in the output
if "entities" in stdout.lower() and "Answer:" in stdout:
    answer_start = stdout.find("Answer:")
    answer_section = stdout[answer_start:answer_start+500]
    print(f"\nAnswer preview:\n{answer_section}")
    
# Check for the specific error
if "Available: []" in stderr:
    print("\n✗ ERROR: Orchestrator still losing state between iterations")
    # Show the error context
    for line in stderr.split('\n'):
        if "Available:" in line:
            print(f"  {line}")
else:
    print("\n✓ No orchestrator state loss detected")