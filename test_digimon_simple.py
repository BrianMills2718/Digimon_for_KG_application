#!/usr/bin/env python3
"""Simple DIGIMON test - let the agent handle everything"""

import subprocess
import time

print("DIGIMON Quick Test - Russian Troll Tweets")
print("=" * 60)
print("\nAsking DIGIMON to analyze the dataset...")
print("This demonstrates the agent's ability to:")
print("1. Automatically prepare the corpus")
print("2. Build appropriate graphs")
print("3. Run analysis")
print("4. Answer questions\n")

# Single command - let the agent figure it out
cmd = [
    "python", "digimon_cli.py",
    "-c", "Data/Russian_Troll_Sample",
    "-q", "Build a graph of this data and tell me: What are the main themes and entities in these Russian troll tweets? Focus on the most frequently mentioned topics."
]

print(f"Command: {' '.join(cmd)}")
print("\nRunning... (this may take 30-60 seconds)")
print("-" * 60)

start_time = time.time()
result = subprocess.run(cmd, capture_output=True, text=True)
elapsed = time.time() - start_time

# Extract just the answer section
output = result.stdout
if "Answer:" in output:
    answer_start = output.find("Answer:")
    answer_section = output[answer_start:]
    print(answer_section)
else:
    print("Full output:", output[-1000:])

if result.stderr:
    print("\nErrors:", result.stderr[-500:])

print(f"\nCompleted in {elapsed:.1f} seconds")
print("\nKey observations:")
print("- DIGIMON can work with natural language commands")
print("- It automatically determines what tools to use")
print("- The agent builds graphs on-demand")
print("- Answers are based on the actual data analyzed")