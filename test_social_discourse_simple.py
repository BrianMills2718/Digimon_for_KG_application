#!/usr/bin/env python3
"""
Very simple test - just prepare corpus for Social Discourse dataset
"""
import asyncio
from pathlib import Path
from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs

async def main():
    """Prepare corpus for Social Discourse dataset"""
    print("\nPreparing Social Discourse Test corpus...")
    print("=" * 80)
    
    # Check if already exists
    corpus_path = Path("results/Social_Discourse_Test/corpus/Corpus.json")
    if corpus_path.exists():
        print(f"Corpus already exists at: {corpus_path}")
        
        # Read and show summary
        import json
        with open(corpus_path) as f:
            data = [json.loads(line) for line in f]
        
        print(f"\nCorpus contains {len(data)} documents:")
        for i, doc in enumerate(data[:3]):
            print(f"\nDocument {i+1}:")
            print(f"  Title: {doc.get('title', 'N/A')}")
            content = doc.get('content', '')
            print(f"  Content preview: {content[:200]}...")
            print(f"  Length: {len(content)} chars")
    else:
        # Prepare corpus
        inputs = PrepareCorpusInputs(
            input_directory_path="Data/Social_Discourse_Test",
            output_directory_path="results/Social_Discourse_Test/corpus",
            target_corpus_name="Social_Discourse_Test"
        )
        
        result = await prepare_corpus_from_directory(inputs)
        print(f"\nStatus: {result.status}")
        print(f"Message: {result.message}")
        print(f"Documents: {result.document_count}")
        print(f"Output: {result.corpus_json_path}")
    
    # Now run a simple query using the CLI
    print("\n" + "=" * 80)
    print("Testing with DIGIMON CLI...")
    print("Query: 'List all the Twitter accounts mentioned'")
    print("=" * 80 + "\n")
    
    import subprocess
    import sys
    
    cmd = [
        sys.executable,
        "digimon_cli.py",
        "-c", "Social_Discourse_Test",
        "-q", "List all the Twitter accounts mentioned",
        "--timeout", "60"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("Query timed out after 120 seconds")
    except Exception as e:
        print(f"Error running query: {e}")

if __name__ == "__main__":
    asyncio.run(main())