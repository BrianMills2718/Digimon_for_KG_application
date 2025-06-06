#!/usr/bin/env python3
"""
Stage 4: Graph Registration & Context Management (Simple Test)
Goal: Check if graphs persist in context between tool calls
"""
import subprocess

def test_graph_registration():
    """Test graph registration using digimon_cli"""
    print("\n" + "="*80)
    print("STAGE 4: Graph Registration & Context Management")
    print("="*80)
    
    # Run a multi-step query that requires graph persistence
    query = """First build an ER graph for Social_Discourse_Test dataset. 
    Then build an entity VDB using that graph. 
    Finally search for entities about 'tech'."""
    
    print(f"\nRunning multi-step query:")
    print(query)
    print("\nExecuting...")
    
    # Run digimon_cli
    result = subprocess.run(
        ["python", "digimon_cli.py", "-c", "Data/Social_Discourse_Test", "-q", query],
        capture_output=True,
        text=True
    )
    
    output = result.stdout + result.stderr
    
    # Check for successful graph building
    graph_built = False
    graph_id = None
    if "Successfully built" in output and "graph" in output:
        graph_built = True
        # Try to extract graph ID
        import re
        match = re.search(r"graph_id['\"]:\s*['\"]([^'\"]+)['\"]", output)
        if match:
            graph_id = match.group(1)
    
    # Check for successful VDB building
    vdb_built = False
    entities_indexed = 0
    if "VDB built successfully" in output or "entities indexed" in output:
        vdb_built = True
        # Try to extract entity count
        match = re.search(r"(\d+)\s*entities\s*indexed", output, re.IGNORECASE)
        if match:
            entities_indexed = int(match.group(1))
    
    # Check for search results
    search_succeeded = False
    if "Found" in output and "entities" in output:
        search_succeeded = True
    
    # Check for graph registration errors
    registration_errors = []
    if "not found in graph_registry" in output:
        registration_errors.append("Graph not found in registry")
    if "GraphRAGContext has no graph" in output:
        registration_errors.append("Context missing graph")
    if "No graph instance available" in output:
        registration_errors.append("Graph instance not accessible")
    
    # Print results
    print("\n" + "-"*40)
    print("ANALYSIS:")
    print(f"- Graph built: {'YES' if graph_built else 'NO'}")
    print(f"- Graph ID: {graph_id or 'NOT FOUND'}")
    print(f"- VDB built: {'YES' if vdb_built else 'NO'}")
    print(f"- Entities indexed: {entities_indexed}")
    print(f"- Search succeeded: {'YES' if search_succeeded else 'NO'}")
    
    if registration_errors:
        print(f"\nRegistration errors detected:")
        for error in registration_errors:
            print(f"  ❌ {error}")
    
    # Check for specific error patterns
    if "status\": \"failure\"" in output:
        print(f"\nFailures detected in output:")
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if '"status": "failure"' in line:
                # Print context around failure
                start = max(0, i-2)
                end = min(len(lines), i+3)
                for j in range(start, end):
                    if j == i:
                        print(f"  >>> {lines[j]}")
                    else:
                        print(f"      {lines[j]}")
    
    # Final verdict
    print("\n" + "="*80)
    print("EVIDENCE SUMMARY:")
    print(f"- graph_built: {graph_id or 'FAILED'}")
    print(f"- vdb_built_from_graph: {'success' if vdb_built else 'failed'}")
    print(f"- entities_indexed: {entities_indexed}")
    print(f"- registration_errors: {len(registration_errors)}")
    
    # Save output for analysis
    with open("stage4_output.txt", "w") as f:
        f.write(output)
    print("\nFull output saved to stage4_output.txt")
    
    if graph_built and vdb_built and entities_indexed > 0 and not registration_errors:
        print("\n✅ STAGE 4: PASSED - Graph registration working!")
        return True
    else:
        print("\n❌ STAGE 4: FAILED - Graph registration issues detected")
        return False

if __name__ == "__main__":
    success = test_graph_registration()