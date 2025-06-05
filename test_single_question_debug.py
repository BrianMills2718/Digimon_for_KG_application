"""Debug single question analysis"""

import asyncio
import traceback
from pathlib import Path
from social_media_discourse_executor import DiscourseEnhancedSocialMediaExecutor

async def test_with_debug():
    print("=== DEBUG SINGLE QUESTION ===\n")
    
    # Create minimal test data
    import pandas as pd
    test_data = {
        'tweet_id': [1, 2, 3],
        'tweet': ['vaccine bioweapon', 'gates conspiracy', 'population control'],
        'conspiracy_theory': ['vaccine', 'gates', 'control'],
        'label': ['support', 'support', 'support']
    }
    pd.DataFrame(test_data).to_csv('debug_test.csv', index=False)
    
    executor = DiscourseEnhancedSocialMediaExecutor()
    
    # Wrap analyze_policy_question to catch the exact error
    try:
        # First test planner directly
        print("1. Testing discourse planner...")
        from Core.AgentTools.discourse_enhanced_planner import DiscourseEnhancedPlanner
        planner = DiscourseEnhancedPlanner()
        
        try:
            scenarios = planner.generate_scenarios(
                ["Who are the super-spreaders?"], 
                "COVID-19 conspiracy theories"
            )
            print(f"✓ Generated {len(scenarios)} scenarios")
        except Exception as e:
            print(f"✗ Planner error: {e}")
            traceback.print_exc()
            return
        
        # Now test full analysis
        print("\n2. Testing full analysis...")
        results = await executor.analyze_policy_question(
            "Who are the super-spreaders?",
            "debug_test.csv"
        )
        
        if "error" in results:
            print(f"✗ Analysis error: {results['error']}")
        else:
            print("✓ Analysis succeeded")
            
    except Exception as e:
        print(f"✗ Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup
        Path('debug_test.csv').unlink(missing_ok=True)

if __name__ == "__main__":
    asyncio.run(test_with_debug())