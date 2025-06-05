#!/usr/bin/env python3
"""Test Stage 2: Verify agent detects and handles failures"""

import subprocess
import re
import time

# Run a command that will trigger PassageGraph failure
cmd = [
    "python", "digimon_cli.py", 
    "-c", "Data/Russian_Troll_Sample",
    "-q", "Build a PassageGraph then analyze discourse patterns",
    "--react"
]

print("Running DIGIMON with a query that will cause PassageGraph to fail...")
print("Command:", " ".join(cmd))
print("\nLooking for agent's handling of failures...\n")

try:
    # Run the command with a reasonable timeout
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180  # 3 minute timeout
    )
    elapsed = time.time() - start_time
    
    output = result.stdout + result.stderr
    
    # Look for key indicators
    print(f"\nExecution completed in {elapsed:.1f} seconds")
    
    # Check if failure was detected
    if "FAILED -" in output:
        print("\n✓ Agent detected failure (found 'FAILED -' in observations)")
    else:
        print("\n✗ No failure detection found")
    
    # Check if agent adapted to failure
    failure_adaptations = [
        "skip", "Skip", "depend", "failed", "alternative", 
        "cannot", "unable", "insufficient", "No input chunks"
    ]
    
    adapted = False
    for term in failure_adaptations:
        if term in output:
            adapted = True
            # Find context around the term
            idx = output.find(term)
            context = output[max(0, idx-100):idx+100]
            print(f"\n✓ Found adaptation indicator '{term}':")
            print(f"   Context: ...{context}...")
            break
    
    if not adapted:
        print("\n⚠ No clear adaptation to failure found")
    
    # Check final answer acknowledges limitation
    answer_start = output.find("Answer:")
    if answer_start > 0:
        answer = output[answer_start:answer_start+500]
        print(f"\nFinal answer excerpt:")
        print(answer)
        
        if any(term in answer.lower() for term in ["could not", "unable", "failed", "error", "no "]):
            print("\n✓ Final answer acknowledges limitations")
        else:
            print("\n⚠ Final answer may not acknowledge failures")
    
    # Summary
    print("\n" + "="*60)
    print("STAGE 2 VERIFICATION:")
    if "FAILED -" in output and (adapted or "could not" in output.lower()):
        print("✓ PASSED: Agent detects failures and adapts behavior")
    else:
        print("✗ FAILED: Agent does not properly handle failures")
        print("\nDebugging info:")
        # Show some ReAct reasoning
        reasoning_matches = re.findall(r"ReAct REASONING: (.+)", output)
        if reasoning_matches:
            print("\nAgent reasoning samples:")
            for r in reasoning_matches[-3:]:
                print(f"  - {r}")
        
except subprocess.TimeoutExpired:
    print("\n✗ Command timed out after 3 minutes")
except Exception as e:
    print(f"\n✗ Error: {e}")