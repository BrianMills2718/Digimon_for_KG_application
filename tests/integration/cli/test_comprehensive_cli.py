#!/usr/bin/env python3
"""
Comprehensive test of DIGIMON CLI functionality
Tests multiple queries against the Fictional_Test corpus
"""

import subprocess
import sys
import json
from pathlib import Path

def run_cli_query(corpus, query):
    """Run a CLI query and return the output"""
    cmd = [
        sys.executable,
        "digimon_cli.py",
        "-c", corpus,
        "-q", query
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr

def extract_answer(output):
    """Extract the answer from CLI output"""
    lines = output.split('\n')
    in_answer = False
    answer_lines = []
    
    for line in lines:
        if line.strip() == "Answer:":
            in_answer = True
            continue
        elif line.strip() == "Context Retrieved:":
            break
        elif in_answer:
            answer_lines.append(line)
    
    return '\n'.join(answer_lines).strip()

def main():
    """Run comprehensive tests"""
    test_queries = [
        {
            "query": "What is crystal technology?",
            "expected_keywords": ["levitite", "432 Hz", "anti-gravity", "Zorathian"],
            "description": "Basic informational query about crystal tech"
        },
        {
            "query": "Tell me about the Crystal Plague",
            "expected_keywords": ["1850 BCE", "black veins", "levitite crystals", "fall"],
            "description": "Query about the Crystal Plague"
        },
        {
            "query": "What caused the fall of the Zorathian Empire?",
            "expected_keywords": ["Crystal Plague", "floating cities", "crashed"],
            "description": "Causal query about empire's downfall"
        },
        {
            "query": "Who was Emperor Zorthak III?",
            "expected_keywords": ["Emperor", "cure", "unsuccessful"],
            "description": "Query about historical figure"
        },
        {
            "query": "Describe the city of Aerophantis",
            "expected_keywords": ["floating", "500 meters", "levitite", "capital"],
            "description": "Query about specific location"
        }
    ]
    
    corpus = "Data/Fictional_Test"
    results = []
    
    print("DIGIMON CLI Comprehensive Test")
    print("=" * 80)
    print(f"Testing corpus: {corpus}")
    print("=" * 80)
    
    for i, test in enumerate(test_queries, 1):
        print(f"\nTest {i}/{len(test_queries)}: {test['description']}")
        print(f"Query: {test['query']}")
        print("-" * 40)
        
        # Run the query
        stdout, stderr = run_cli_query(corpus, test['query'])
        
        # Extract answer
        answer = extract_answer(stdout)
        
        # Check for expected keywords
        found_keywords = []
        missing_keywords = []
        
        for keyword in test['expected_keywords']:
            if keyword.lower() in answer.lower():
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        # Determine success
        success = len(found_keywords) >= len(test['expected_keywords']) * 0.5  # At least 50% of keywords
        
        # Store result
        result = {
            "query": test['query'],
            "answer": answer,
            "success": success,
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "answer_length": len(answer)
        }
        results.append(result)
        
        # Print result summary
        print(f"Answer ({len(answer)} chars):")
        if len(answer) > 300:
            print(answer[:300] + "...")
        else:
            print(answer)
        
        print(f"\nKeyword Analysis:")
        print(f"  Found: {found_keywords}")
        print(f"  Missing: {missing_keywords}")
        print(f"  Status: {'✅ PASS' if success else '❌ FAIL'}")
        
        # Check for errors
        if "Error" in answer or "error" in answer.lower():
            print(f"  ⚠️  WARNING: Answer contains error message")
        
        if stderr:
            print(f"\nErrors in stderr:")
            print(stderr[:500])
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Total queries: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    print("\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✅" if result['success'] else "❌"
        print(f"{status} Query {i}: {result['query'][:50]}...")
        print(f"   Answer length: {result['answer_length']} chars")
        print(f"   Keywords found: {len(result['found_keywords'])}/{len(result['found_keywords']) + len(result['missing_keywords'])}")
    
    # Save results
    with open('cli_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: cli_test_results.json")

if __name__ == "__main__":
    main()