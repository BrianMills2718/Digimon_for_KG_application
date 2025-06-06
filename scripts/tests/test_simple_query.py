#!/usr/bin/env python3
"""Test a very simple query"""

import subprocess
import time

cmd = [
    "python", "digimon_cli.py",
    "-c", "Data/Russian_Troll_Sample",
    "-q", "What data is available?"
]

print("Testing simple query...")
print("This should generate a natural language plan")
print("-" * 60)

start = time.time()
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
elapsed = time.time() - start

# Extract relevant parts
output_lines = result.stdout.split('\n')
stderr_lines = result.stderr.split('\n')

print(f"\nCompleted in {elapsed:.1f}s\n")

# Look for key information
found_plan = False
for i, line in enumerate(output_lines):
    if "natural language plan" in line.lower():
        print(f"Plan line {i}: {line}")
        found_plan = True
    if "steps:" in line:
        print(f"Steps line {i}: {line}")
    if "Answer:" in line:
        # Print answer section
        for j in range(i, min(i+10, len(output_lines))):
            print(output_lines[j])
        break

if not found_plan:
    print("\nNo natural language plan found in output")
    
# Check for errors
print("\nErrors found:")
for line in stderr_lines:
    if "error" in line.lower() and "natural" in line.lower():
        print("STDERR:", line)

# Also check stdout for errors
for line in output_lines:
    if "error" in line.lower() and "natural" in line.lower():
        print("STDOUT:", line)