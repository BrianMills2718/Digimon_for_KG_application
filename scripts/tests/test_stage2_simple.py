#!/usr/bin/env python3
"""
Stage 2: Tool Hallucination Prevention (Simplified Test)
Goal: Run digimon_cli and check what tools it tries to use
"""
import subprocess
import re

def test_tool_hallucination():
    """Test for tool hallucinations using digimon_cli"""
    print("\n" + "="*80)
    print("STAGE 2: Tool Hallucination Prevention")
    print("="*80)
    
    # Run digimon_cli with a query that might trigger hallucinations
    query = "Build an ER graph for Social_Discourse_Test and create a vector database index"
    
    print(f"\nRunning query: {query}")
    print("\nExecuting digimon_cli.py...")
    
    # Capture output
    result = subprocess.run(
        ["python", "digimon_cli.py", "-c", "Data/Social_Discourse_Test", "-q", query],
        capture_output=True,
        text=True
    )
    
    output = result.stdout + result.stderr
    
    # Extract tool references from output
    # Look for patterns like "Tool ID 'X' not found" or "tool_id": "X"
    tool_not_found_pattern = r"Tool ID '([^']+)' not found"
    tool_id_pattern = r'"tool_id":\s*"([^"]+)"'
    
    # Find all tool references
    all_tools = set(re.findall(tool_id_pattern, output))
    missing_tools = set(re.findall(tool_not_found_pattern, output))
    
    print("\nTools referenced in plan:")
    for tool in sorted(all_tools):
        if tool in missing_tools:
            print(f"  ❌ {tool} (NOT FOUND)")
        else:
            print(f"  ✓ {tool}")
    
    # Check for specific hallucinated tools we know about
    known_hallucinations = [
        "vector_db.CreateIndex",
        "vector_db.QueryIndex", 
        "graph.GetClusters",
        "Chunk.GetTextForClusters",
        "report.Generate"
    ]
    
    found_hallucinations = []
    for tool in known_hallucinations:
        if tool in all_tools or tool in output:
            found_hallucinations.append(tool)
    
    # Look for execution errors
    error_count = output.count("Tool ID") and output.count("not found")
    failure_count = output.count('"status": "failure"')
    
    print("\n" + "="*80)
    print("EVIDENCE SUMMARY:")
    print(f"- total_tools_used: {len(all_tools)}")
    print(f"- missing_tools: {len(missing_tools)} tools not found")
    print(f"- known_hallucinations: {len(found_hallucinations)} detected")
    print(f"- execution_errors: {error_count + failure_count}")
    
    if missing_tools:
        print(f"\nMissing tools:")
        for tool in sorted(missing_tools):
            print(f"  - {tool}")
    
    if found_hallucinations:
        print(f"\nKnown hallucinations found:")
        for tool in found_hallucinations:
            print(f"  - {tool}")
    
    # Check if the query was successful despite hallucinations
    success_indicators = ["Successfully built", "nodes:", "edges:", "Graph built"]
    query_succeeded = any(indicator in output for indicator in success_indicators)
    
    print(f"\nQuery success: {'YES' if query_succeeded else 'NO'}")
    
    if not missing_tools and not found_hallucinations:
        print("\n✅ STAGE 2: PASSED - No tool hallucinations!")
        return True
    else:
        print("\n❌ STAGE 2: FAILED - Tool hallucinations detected")
        
        # Save output for analysis
        with open("stage2_output.txt", "w") as f:
            f.write(output)
        print("\nFull output saved to stage2_output.txt")
        
        return False

if __name__ == "__main__":
    success = test_tool_hallucination()