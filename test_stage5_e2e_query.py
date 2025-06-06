#!/usr/bin/env python3
"""
Stage 5: End-to-End Query Success
Goal: Complete full pipeline successfully with real data in answer
"""
import subprocess
import re

def test_e2e_query():
    """Test complete end-to-end query pipeline"""
    print("\n" + "="*80)
    print("STAGE 5: End-to-End Query Success")
    print("="*80)
    
    # Test with a query that should return real data
    query = "Tell me about the tech community and AI researchers in the social network"
    
    print(f"\nRunning query: {query}")
    print("\nExecuting full pipeline...")
    
    # Run digimon_cli
    result = subprocess.run(
        ["python", "digimon_cli.py", "-c", "Data/Social_Discourse_Test", "-q", query],
        capture_output=True,
        text=True
    )
    
    output = result.stdout + result.stderr
    
    # Track pipeline stages
    corpus_docs = 0
    graph_nodes = 0
    graph_edges = 0
    vdb_entities = 0
    search_results = []
    retrieved_text = []
    final_answer = ""
    errors = []
    
    # Extract metrics from output
    # Corpus loading
    match = re.search(r"Successfully loaded (\d+) documents", output)
    if match:
        corpus_docs = int(match.group(1))
    
    # Graph building
    match = re.search(r"node_count=(\d+)", output)
    if match:
        graph_nodes = int(match.group(1))
    match = re.search(r"edge_count=(\d+)", output)
    if match:
        graph_edges = int(match.group(1))
    
    # VDB building
    match = re.search(r"(\d+)\s*entities indexed", output)
    if match:
        vdb_entities = int(match.group(1))
    
    # Search results
    matches = re.findall(r"Found entity: ([^\n]+)", output)
    search_results.extend(matches)
    
    # Retrieved text
    text_matches = re.findall(r"Retrieved text[^:]*:\s*([^\n]+)", output)
    retrieved_text.extend(text_matches)
    
    # Extract final answer
    if "## Answer:" in output:
        answer_start = output.find("## Answer:")
        if answer_start != -1:
            answer_section = output[answer_start:].split("\n", 2)
            if len(answer_section) > 1:
                final_answer = answer_section[1].strip()
    
    # Check for errors
    if '"status": "failure"' in output:
        errors.append("Tool failures detected")
    if "Error:" in output:
        error_matches = re.findall(r"Error: ([^\n]+)", output)
        errors.extend(error_matches)
    
    # Analyze answer quality
    answer_has_data = False
    expected_entities = ["tech", "AI", "researcher", "alex chen", "sarah johnson", 
                        "techvisionary", "airesearcher", "community"]
    
    found_in_answer = []
    for entity in expected_entities:
        if entity.lower() in final_answer.lower():
            found_in_answer.append(entity)
            answer_has_data = True
    
    # Print analysis
    print("\n" + "-"*40)
    print("PIPELINE ANALYSIS:")
    print(f"1. Corpus loading: {corpus_docs} documents")
    print(f"2. Graph building: {graph_nodes} nodes, {graph_edges} edges")
    print(f"3. VDB indexing: {vdb_entities} entities")
    print(f"4. Search results: {len(search_results)} entities found")
    print(f"5. Text retrieval: {len(retrieved_text)} chunks retrieved")
    print(f"6. Final answer: {'YES' if final_answer else 'NO'} ({len(final_answer)} chars)")
    
    if search_results:
        print(f"\nSample search results:")
        for result in search_results[:5]:
            print(f"  - {result}")
    
    if found_in_answer:
        print(f"\nEntities found in answer:")
        for entity in found_in_answer:
            print(f"  ✓ {entity}")
    
    if errors:
        print(f"\nErrors detected:")
        for error in errors[:5]:
            print(f"  ❌ {error}")
    
    # Show answer preview
    if final_answer:
        print(f"\nAnswer preview:")
        print(f"  {final_answer[:200]}...")
    
    # Final verdict
    print("\n" + "="*80)
    print("EVIDENCE SUMMARY:")
    print(f"- corpus_docs: {corpus_docs}")
    print(f"- graph_nodes: {graph_nodes}")
    print(f"- vdb_entities: {vdb_entities}")
    print(f"- search_results: {search_results[:3] if search_results else []}")
    print(f"- retrieved_text: {'YES' if retrieved_text else 'NO'}")
    print(f"- final_answer: {'Contains real data' if answer_has_data else 'No real data'}")
    
    # Success criteria
    success = (
        corpus_docs > 0 and
        graph_nodes > 0 and
        vdb_entities > 0 and
        answer_has_data and
        len(errors) == 0
    )
    
    if success:
        print("\n✅ STAGE 5: PASSED - Full pipeline works end-to-end!")
        print(f"   Answer contains {len(found_in_answer)} expected entities")
    else:
        print("\n❌ STAGE 5: FAILED - Pipeline issues detected")
        if not answer_has_data:
            print("   Answer does not contain real data from corpus")
    
    # Save output
    with open("stage5_output.txt", "w") as f:
        f.write(output)
    print("\nFull output saved to stage5_output.txt")
    
    return success

if __name__ == "__main__":
    success = test_e2e_query()