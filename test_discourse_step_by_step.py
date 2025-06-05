"""Step-by-step test to identify exact failure point"""

import asyncio
import pandas as pd
from pathlib import Path
import traceback

async def test_step_by_step():
    print("=== STEP-BY-STEP DISCOURSE TEST ===\n")
    
    # Step 1: Import and create executor
    print("Step 1: Importing modules...")
    try:
        from social_media_discourse_executor import DiscourseEnhancedSocialMediaExecutor
        executor = DiscourseEnhancedSocialMediaExecutor()
        print("✓ Executor created")
    except Exception as e:
        print(f"✗ Failed to create executor: {e}")
        traceback.print_exc()
        return
    
    # Step 2: Initialize
    print("\nStep 2: Initializing executor...")
    try:
        success = await executor.initialize()
        if success:
            print("✓ Initialization successful")
        else:
            print("✗ Initialization returned False")
            return
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        traceback.print_exc()
        return
    
    # Step 3: Create test data
    print("\nStep 3: Creating test data...")
    try:
        test_data = {
            'tweet_id': [1, 2, 3],
            'tweet': ['test 1', 'test 2', 'test 3'],
            'conspiracy_theory': ['test', 'test', 'test'],
            'label': ['support', 'deny', 'neutral']
        }
        pd.DataFrame(test_data).to_csv('test_step.csv', index=False)
        print("✓ Test CSV created")
    except Exception as e:
        print(f"✗ Failed to create test data: {e}")
        traceback.print_exc()
        return
    
    # Step 4: Create corpus directory manually
    print("\nStep 4: Creating corpus directory...")
    try:
        corpus_dir = Path("./discourse_corpus_test_step")
        corpus_dir.mkdir(exist_ok=True)
        
        # Create a simple text file
        test_file = corpus_dir / "test_doc.txt"
        test_file.write_text("Test document content\nTweet: test tweet\nNarrative: test")
        
        print(f"✓ Created corpus directory with {len(list(corpus_dir.glob('*.txt')))} files")
    except Exception as e:
        print(f"✗ Failed to create corpus directory: {e}")
        traceback.print_exc()
        return
    
    # Step 5: Try corpus preparation directly
    print("\nStep 5: Testing corpus preparation tool...")
    try:
        from Core.AgentTools.corpus_tools import PrepareCorpusFromDirectory
        from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs
        
        tool = PrepareCorpusFromDirectory(
            main_config=executor.config,
            graphrag_context=executor.context
        )
        
        inputs = PrepareCorpusInputs(
            directory_path=str(corpus_dir),
            corpus_name="test_step",
            output_path=str(corpus_dir)
        )
        
        result = tool.run(inputs)
        print(f"✓ Corpus tool result: {result}")
    except Exception as e:
        print(f"✗ Corpus tool error: {e}")
        traceback.print_exc()
        
        # Try with absolute path
        print("\nTrying with absolute path...")
        try:
            abs_corpus_dir = corpus_dir.absolute()
            inputs = PrepareCorpusInputs(
                directory_path=str(abs_corpus_dir),
                corpus_name="test_step",
                output_path=str(abs_corpus_dir)
            )
            result = tool.run(inputs)
            print(f"✓ Corpus tool with absolute path: {result}")
        except Exception as e2:
            print(f"✗ Still failed with absolute path: {e2}")
    
    # Step 6: Test prepare_dataset method
    print("\nStep 6: Testing prepare_dataset method...")
    try:
        success = await executor.prepare_dataset('test_step.csv', 'test_step')
        if success:
            print("✓ prepare_dataset successful")
        else:
            print("✗ prepare_dataset returned False")
    except Exception as e:
        print(f"✗ prepare_dataset error: {e}")
        traceback.print_exc()
    
    print("\n=== END OF STEP-BY-STEP TEST ===")

# Cleanup
def cleanup():
    for path in ['test_step.csv', './discourse_corpus_test_step']:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
            else:
                p.unlink()

if __name__ == "__main__":
    try:
        asyncio.run(test_step_by_step())
    finally:
        cleanup()
        print("\nCleaned up test files")