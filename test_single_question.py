"""Test single policy question analysis"""

import asyncio
import json
from pathlib import Path
from social_media_discourse_executor import DiscourseEnhancedSocialMediaExecutor

async def test_single_question():
    print("=== SINGLE QUESTION TEST ===\n")
    
    # Check if we have the real dataset
    dataset_path = Path("COVID-19-conspiracy-theories-tweets.csv")
    if not dataset_path.exists():
        print("Creating small test dataset...")
        import pandas as pd
        test_data = {
            'tweet_id': range(1, 51),
            'tweet': [
                f"@user{i % 5} The vaccine is a bioweapon for population control #conspiracy #covid19"
                if i % 3 == 0 else
                f"Bill Gates created the virus to sell vaccines #plandemic @influencer{i % 3}"
                if i % 3 == 1 else
                f"Government hiding the truth about vaccine deaths #truth #wakeup"
                for i in range(1, 51)
            ],
            'conspiracy_theory': [
                ['vaccine_control', 'bioweapon', 'gates_conspiracy'][i % 3]
                for i in range(50)
            ],
            'label': [['support', 'support', 'deny'][i % 3] for i in range(50)]
        }
        pd.DataFrame(test_data).to_csv('test_single_question.csv', index=False)
        dataset_path = Path('test_single_question.csv')
        print("✓ Test dataset created")
    else:
        print("✓ Using real COVID conspiracy dataset")
    
    # Create executor with trace
    def trace_callback(event_type, data):
        if event_type in ["policy_question_start", "policy_question_complete", "init_complete", "dataset_prep_complete"]:
            print(f"[TRACE] {event_type}: {data}")
    
    executor = DiscourseEnhancedSocialMediaExecutor(trace_callback=trace_callback)
    
    # Test simple question first
    question = "Who are the super-spreaders of COVID conspiracy theories?"
    
    print(f"\nAnalyzing question: {question}")
    print("-" * 60)
    
    try:
        results = await executor.analyze_policy_question(question, str(dataset_path))
        
        if "error" in results:
            print(f"✗ Analysis failed: {results['error']}")
            return False
        
        print(f"✓ Analysis completed successfully!")
        print(f"  - Insights generated: {len(results.get('insights', []))}")
        print(f"  - Discourse patterns: {len(results.get('discourse_patterns', {}))}")
        print(f"  - Execution time: {results.get('analysis_time', 'N/A')} seconds")
        
        # Show sample insights
        if results.get('insights'):
            print("\nSample insights:")
            for i, insight in enumerate(results['insights'][:3]):
                print(f"  {i+1}. {insight.get('type', 'Unknown')}: {insight.get('description', 'N/A')[:80]}...")
        
        # Save results
        output_file = Path("test_single_question_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Results saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Exception during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup():
    """Clean up test files"""
    for path in ['test_single_question.csv', 'test_single_question_results.json']:
        p = Path(path)
        if p.exists():
            p.unlink()
    
    # Clean up discourse corpus directories
    import shutil
    for dir_pattern in Path('.').glob('discourse_corpus_*'):
        if dir_pattern.is_dir():
            shutil.rmtree(dir_pattern)

if __name__ == "__main__":
    success = asyncio.run(test_single_question())
    
    if success:
        print("\n=== SINGLE QUESTION TEST PASSED ===")
    else:
        print("\n=== SINGLE QUESTION TEST FAILED ===")
    
    # Optionally cleanup
    # cleanup()
    # print("Test files cleaned up")