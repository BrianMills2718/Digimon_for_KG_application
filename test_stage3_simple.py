#!/usr/bin/env python3
"""
Stage 3: Corpus Path Standardization (Simple Test)
Goal: Check where corpus files are created vs where they're expected
"""
import os
import subprocess
from pathlib import Path

def test_corpus_paths():
    """Test corpus path handling"""
    print("\n" + "="*80)
    print("STAGE 3: Corpus Path Standardization")
    print("="*80)
    
    # Use the existing Social_Discourse_Test dataset
    dataset = "Social_Discourse_Test"
    
    # Check where corpus files exist
    print(f"\nChecking corpus locations for dataset: {dataset}")
    
    locations = {
        "Tool creates at": Path(f"results/{dataset}/corpus/Corpus.json"),
        "ChunkFactory primary": Path(f"results/{dataset}/Corpus.json"),
        "ChunkFactory secondary": Path(f"Data/{dataset}/Corpus.json"),
    }
    
    corpus_found = []
    for desc, path in locations.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        print(f"\n{desc}:")
        print(f"  Path: {path}")
        print(f"  Exists: {'✓ YES' if exists else '✗ NO'}")
        if exists:
            print(f"  Size: {size} bytes")
            corpus_found.append((desc, path))
    
    # Test if ChunkFactory can find the corpus
    print("\n" + "-"*40)
    print("Testing ChunkFactory access...")
    
    test_code = f"""
import asyncio
from Core.Chunk.ChunkFactory import ChunkFactory
from Option.Config2 import Config

async def test():
    config = Config.default()
    chunk_factory = ChunkFactory(config)
    chunks = await chunk_factory.get_chunks_for_dataset("{dataset}")
    return len(chunks)

print(f"Chunks loaded: {{asyncio.run(test())}}")
"""
    
    result = subprocess.run(
        ["python", "-c", test_code],
        capture_output=True,
        text=True
    )
    
    chunks_loaded = 0
    if "Chunks loaded:" in result.stdout:
        try:
            chunks_loaded = int(result.stdout.split("Chunks loaded:")[1].strip())
        except:
            pass
    
    print(f"\nChunkFactory result: {chunks_loaded} chunks loaded")
    
    if result.stderr and "not found" in result.stderr:
        print("❌ ChunkFactory could not find corpus")
    
    # Check if we need to copy/move files
    need_fix = False
    if corpus_found:
        actual_path = corpus_found[0][1]
        expected_path = Path(f"results/{dataset}/Corpus.json")
        
        if actual_path != expected_path:
            need_fix = True
            print(f"\n⚠️ PATH MISMATCH DETECTED!")
            print(f"  Found at: {actual_path}")
            print(f"  Expected: {expected_path}")
            
            # Test if copying fixes it
            if actual_path.exists() and not expected_path.exists():
                print(f"\nTesting fix: copy {actual_path} -> {expected_path}")
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(actual_path, expected_path)
                
                # Re-test
                result2 = subprocess.run(
                    ["python", "-c", test_code],
                    capture_output=True,
                    text=True
                )
                
                if "Chunks loaded:" in result2.stdout:
                    try:
                        chunks_after_fix = int(result2.stdout.split("Chunks loaded:")[1].strip())
                        if chunks_after_fix > 0:
                            print(f"✓ Fix successful! {chunks_after_fix} chunks loaded after copy")
                            need_fix = False
                    except:
                        pass
    
    # Final verdict
    print("\n" + "="*80)
    print("EVIDENCE SUMMARY:")
    
    if corpus_found:
        print(f"- corpus_created_at: {corpus_found[0][1]}")
    else:
        print(f"- corpus_created_at: NOT FOUND")
        
    print(f"- corpus_expected_at: results/{dataset}/Corpus.json")
    print(f"- chunks_loaded: {chunks_loaded}")
    print(f"- path_mismatch: {'YES' if need_fix else 'NO'}")
    
    if chunks_loaded > 0 and not need_fix:
        print("\n✅ STAGE 3: PASSED - Corpus paths are properly handled!")
        return True
    else:
        print("\n❌ STAGE 3: FAILED - Corpus path issues detected")
        print("\nRecommended fix:")
        print(f"1. Update PrepareCorpusFromDirectory to create at results/{{dataset}}/Corpus.json")
        print(f"2. OR update ChunkFactory to check results/{{dataset}}/corpus/Corpus.json")
        return False

if __name__ == "__main__":
    success = test_corpus_paths()