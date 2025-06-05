"""Minimal test for discourse executor functionality"""

import asyncio
import pandas as pd
from pathlib import Path
from social_media_discourse_executor import DiscourseEnhancedSocialMediaExecutor

async def test_minimal():
    print("=== MINIMAL DISCOURSE TEST ===\n")
    
    # Create tiny test dataset
    print("1. Creating test dataset...")
    test_data = {
        'tweet_id': [1, 2, 3, 4, 5],
        'tweet': [
            '@user1 The vaccine is a bioweapon #conspiracy',
            'Bill Gates created virus #plandemic @influencer1',
            'Population control through vaccines #truth',
            '@user2 Wake up people #covid19 #conspiracy',
            'Government hiding the truth #vaccine #bioweapon'
        ],
        'conspiracy_theory': ['vaccine_control', 'gates_conspiracy', 'population_control', 'general', 'bioweapon'],
        'label': ['support', 'support', 'support', 'neutral', 'support']
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv('test_minimal.csv', index=False)
    print("✓ Test dataset created")
    
    # Initialize executor
    print("\n2. Initializing discourse executor...")
    executor = DiscourseEnhancedSocialMediaExecutor()
    
    try:
        success = await executor.initialize()
        if not success:
            print("✗ Failed to initialize executor")
            return False
        print("✓ Executor initialized successfully")
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test dataset preparation
    print("\n3. Testing dataset preparation...")
    try:
        success = await executor.prepare_dataset('test_minimal.csv', 'test_minimal')
        if not success:
            print("✗ Failed to prepare dataset")
            return False
        print("✓ Dataset prepared successfully")
        
        # Check if corpus directory was created
        corpus_dir = Path("./discourse_corpus_test_minimal")
        if corpus_dir.exists():
            files = list(corpus_dir.glob("discourse_chunk_*.txt"))
            print(f"  - Created {len(files)} discourse chunk files")
    except Exception as e:
        print(f"✗ Dataset preparation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ ALL MINIMAL TESTS PASSED")
    return True

# Clean up function
def cleanup():
    """Remove test files"""
    test_files = [
        'test_minimal.csv',
        './discourse_corpus_test_minimal'
    ]
    for file in test_files:
        path = Path(file)
        if path.exists():
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
            else:
                path.unlink()

if __name__ == "__main__":
    try:
        success = asyncio.run(test_minimal())
        if success:
            print("\n=== TEST COMPLETED SUCCESSFULLY ===")
        else:
            print("\n=== TEST FAILED ===")
    finally:
        cleanup()
        print("\nTest files cleaned up")