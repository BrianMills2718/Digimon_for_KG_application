#!/usr/bin/env python3
"""
Test the Social Discourse dataset with DIGIMON
"""
import subprocess
import sys

def test_query(query, description):
    """Run a test query on the social discourse dataset"""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"QUERY: {query}")
    print('='*80)
    
    cmd = [
        sys.executable,
        "digimon_cli.py",
        "-c", "Data/Social_Discourse_Test",
        "-q", query
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for complex queries
        )
        
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("Query timed out after 5 minutes")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run tests on the social discourse dataset"""
    print("TESTING SOCIAL DISCOURSE DATASET WITH DIGIMON")
    print("=" * 80)
    print("This dataset is optimized for:")
    print("- Dense social networks with clear communities")
    print("- Rich discourse patterns and opinion evolution")
    print("- Explicit relationships via mentions")
    print("- Temporal dynamics across 5 phases")
    print("- Multiple stakeholder perspectives")
    
    # Test queries designed to exercise different capabilities
    test_queries = [
        # Basic entity extraction
        ("List the main actors and their roles", "Basic entity extraction"),
        
        # Community detection
        ("Identify the different communities and their key members", "Community detection"),
        
        # Relationship analysis
        ("Who mentions whom in this discourse? Show the social network connections", "Social network analysis"),
        
        # Discourse analysis
        ("Analyze how opinions evolve from announcement to resolution phase", "Temporal discourse analysis"),
        
        # Influence analysis
        ("Who are the most influential actors based on mentions and engagement?", "Influence analysis"),
        
        # Theme extraction
        ("What are the main themes and how do different communities respond to them?", "Theme and perspective analysis"),
        
        # Bridge identification
        ("Which actors connect different communities? Who facilitates dialogue?", "Bridge actor identification"),
        
        # Complex multi-step
        ("Build a social network graph, identify communities, analyze discourse patterns, and explain how the conflict was resolved", "Complex integrated analysis")
    ]
    
    # Run a subset of queries (full set might take too long)
    print("\nRunning key test queries...")
    
    # Start with simpler queries
    test_query(test_queries[0][0], test_queries[0][1])  # Basic entities
    test_query(test_queries[2][0], test_queries[2][1])  # Social network
    test_query(test_queries[5][0], test_queries[5][1])  # Themes
    
    print("\n" + "="*80)
    print("DATASET READY FOR TESTING")
    print("The Social_Discourse_Test dataset provides:")
    print("- Clear network structure for relationship analysis")
    print("- Temporal evolution for discourse tracking")
    print("- Multiple viewpoints for perspective analysis")
    print("- Rich mention network for influence measurement")
    print("- Explicit communities for clustering validation")

if __name__ == "__main__":
    main()