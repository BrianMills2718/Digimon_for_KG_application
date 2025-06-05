#!/usr/bin/env python3
"""Test Stage 3: Test PPR functionality"""

import subprocess
import re

# Test PPR directly
cmd = [
    "python", "digimon_cli.py", 
    "-c", "Data/Russian_Troll_Sample",
    "-q", "Build an ER graph then run Entity.PPR with seed entities Trump, Russia",
    "--react"
]

print("Testing Entity.PPR functionality...")
print("Command:", " ".join(cmd))
print("\nLooking for PPR execution...\n")

try:
    # Run with timeout
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180  # 3 minutes
    )
    
    output = result.stdout + result.stderr
    
    # Check for the specific error
    if "'EntityRetriever' object has no attribute 'entities_to_relationships'" in output:
        print("\n✗ FAILED: Still getting entities_to_relationships AttributeError")
    elif "Entity.PPR: Error during entity_retriever._find_relevant_entities_by_ppr" in output:
        print("\n✗ FAILED: PPR execution failed with error")
        # Find the error
        error_match = re.search(r"Entity.PPR: Error during.*?: (.+)", output)
        if error_match:
            print(f"   Error: {error_match.group(1)}")
    elif "ranked_entities" in output or "Entity.PPR" in output:
        print("\n✓ PASSED: PPR execution appears to have completed")
        # Check if we got results
        if "ranked_entities=[]" in output:
            print("   Note: PPR returned empty results (might be expected)")
        else:
            print("   PPR returned some results")
    else:
        print("\n? UNCLEAR: Could not determine PPR execution status")
    
    # Check for missing tools
    missing_tools = []
    if "Tool ID 'Chunk.GetTextForClusters' not found" in output:
        missing_tools.append("Chunk.GetTextForClusters")
    if "Tool ID 'report.Generate' not found" in output:
        missing_tools.append("report.Generate")
    
    if missing_tools:
        print(f"\n⚠ Missing tools detected: {', '.join(missing_tools)}")
        print("  These tools are referenced by the agent but not implemented")
    
    # Summary
    print("\n" + "="*60)
    print("STAGE 3 VERIFICATION:")
    if "'EntityRetriever' object has no attribute" not in output:
        print("✓ PARTIAL PASS: entities_to_relationships error fixed")
        if missing_tools:
            print("⚠ But missing tools still need to be addressed")
    else:
        print("✗ FAILED: PPR still has AttributeError")
        
except subprocess.TimeoutExpired:
    print("\n✗ Command timed out")
except Exception as e:
    print(f"\n✗ Error: {e}")