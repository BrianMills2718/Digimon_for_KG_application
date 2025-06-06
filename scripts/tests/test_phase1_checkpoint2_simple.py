#!/usr/bin/env python3
"""
Simple Phase 1.2 Test - Focus on core functionality
"""

import asyncio
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from social_media_discourse_executor import DiscourseEnhancedSocialMediaExecutor
from Core.AgentTools.discourse_enhanced_planner import DiscourseEnhancedPlanner


async def test_simple():
    print("=== SIMPLE PHASE 1.2 TEST ===\n")
    
    tests_passed = 0
    
    # Test 1: COVID dataset exists
    print("Test 1: COVID dataset check...")
    covid_path = Path("./COVID-19-conspiracy-theories-tweets.csv")
    if covid_path.exists():
        df = pd.read_csv(covid_path)
        print(f"✓ Found dataset with {len(df)} tweets")
        tests_passed += 1
    else:
        print("✗ Dataset not found")
        return False
    
    # Test 2: Basic corpus creation
    print("\nTest 2: Basic corpus creation...")
    try:
        executor = DiscourseEnhancedSocialMediaExecutor()
        await executor.initialize()
        
        # Use small sample
        sample_df = df.head(100)
        sample_path = Path("./test_sample.csv")
        sample_df.to_csv(sample_path, index=False)
        
        result = await executor.prepare_dataset(str(sample_path), "test_simple")
        
        corpus_path = Path("./results/test_simple/Corpus.json")
        if corpus_path.exists():
            print("✓ Corpus created successfully")
            tests_passed += 1
            
            # Quick peek at content
            with open(Path("./results/test_simple/discourse_chunk_0.txt")) as f:
                content = f.read()
                print("\nSample content preview:")
                print(content[:300] + "...")
        else:
            print("✗ Corpus not created")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Entity extraction setup
    print("\nTest 3: Entity extraction setup...")
    try:
        planner = DiscourseEnhancedPlanner()
        views = planner.generate_discourse_views("COVID analysis", ["Who"])
        if views:
            print(f"✓ Entity extraction configured: {views[0].entities}")
            tests_passed += 1
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Cleanup
    import shutil
    for path in ["./results/test_simple", "./test_sample.csv"]:
        if Path(path).exists():
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                Path(path).unlink()
    
    print(f"\nTests passed: {tests_passed}/3")
    return tests_passed >= 2


if __name__ == "__main__":
    success = asyncio.run(test_simple())
    sys.exit(0 if success else 1)