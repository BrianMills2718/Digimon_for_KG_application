# Fix and Test Plan for Discourse Analysis System

## Current Failures
1. **GraphRAGContext initialization error**: Missing required fields `target_dataset_name` and `main_config`
2. **SyntaxError in test framework**: Missing closing bracket in list comprehension

## Checkpoint and Test Plan

### Checkpoint 1: Fix Syntax Errors ✓
**Goal**: Ensure all Python files have valid syntax

**Tests**:
```bash
python -m py_compile test_discourse_analysis_framework.py
python -m py_compile social_media_discourse_executor.py
python -m py_compile demo_policy_discourse_analysis.py
```

**Success Criteria**: All files compile without syntax errors

### Checkpoint 2: Fix GraphRAGContext Initialization ✓
**Goal**: Correct the GraphRAGContext initialization to use proper field names

**Issue**: The error shows:
- Expected: `target_dataset_name` and `main_config`
- Provided: `dataset_name` and `config`

**Tests**:
```python
# minimal_context_test.py
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config

config = Config.from_yaml_file("Option/Config2.yaml")
context = GraphRAGContext(
    target_dataset_name="test_dataset",
    main_config=config
)
print("✓ Context created successfully")
```

**Success Criteria**: Context object creates without errors

### Checkpoint 3: Create Minimal Working Test ✓
**Goal**: Verify core functionality works before full demo

**Tests**:
```python
# minimal_discourse_test.py
import asyncio
from social_media_discourse_executor import DiscourseEnhancedSocialMediaExecutor

async def test_minimal():
    executor = DiscourseEnhancedSocialMediaExecutor()
    
    # Test 1: Initialize
    success = await executor.initialize()
    assert success, "Failed to initialize"
    print("✓ Initialization successful")
    
    # Test 2: Prepare small dataset
    # Create tiny test CSV
    import pandas as pd
    test_data = {
        'tweet_id': [1, 2, 3],
        'tweet': ['test tweet 1', 'test tweet 2', 'test tweet 3'],
        'conspiracy_theory': ['test', 'test', 'test'],
        'label': ['support', 'deny', 'neutral']
    }
    pd.DataFrame(test_data).to_csv('test_minimal.csv', index=False)
    
    success = await executor.prepare_dataset('test_minimal.csv', 'test_minimal')
    assert success, "Failed to prepare dataset"
    print("✓ Dataset preparation successful")
    
    print("\n✓ ALL MINIMAL TESTS PASSED")

asyncio.run(test_minimal())
```

**Success Criteria**: All assertions pass

### Checkpoint 4: Test Single Question Analysis ✓
**Goal**: Ensure one question can be analyzed before attempting all 5

**Tests**:
```python
# test_single_question.py
import asyncio
from social_media_discourse_executor import DiscourseEnhancedSocialMediaExecutor

async def test_single_question():
    executor = DiscourseEnhancedSocialMediaExecutor()
    
    question = "Who are the super-spreaders?"
    results = await executor.analyze_policy_question(question, 'test_minimal.csv')
    
    assert "error" not in results, f"Analysis failed: {results.get('error')}"
    assert "insights" in results, "No insights generated"
    print(f"✓ Generated {len(results['insights'])} insights")
    print("✓ SINGLE QUESTION TEST PASSED")

asyncio.run(test_single_question())
```

**Success Criteria**: Analysis completes without errors

### Checkpoint 5: Test Full Demo with Real Data ✓
**Goal**: Ensure full demonstration works with actual dataset

**Tests**:
1. Run test framework first
2. Run demonstration
3. Verify outputs created

**Success Criteria**: 
- All 5 questions analyzed
- Results saved to files
- No Python errors

## Iteration Process

For each checkpoint:
1. **Make the fix**
2. **Run the test**
3. **If test fails**:
   - Read error message carefully
   - Fix the specific issue
   - Re-run test
   - Repeat until test passes
4. **Only proceed to next checkpoint after current one passes**

## Fixes to Implement

### Fix 1: SyntaxError in test_discourse_analysis_framework.py
Line 149 is missing a closing bracket:
```python
# WRONG:
'label': ['support', 'deny', 'neutral'][i % 3] for i in range(100)

# CORRECT:
'label': [['support', 'deny', 'neutral'][i % 3] for i in range(100)]
```

### Fix 2: GraphRAGContext initialization
Change all occurrences from:
```python
# WRONG:
self.context = GraphRAGContext(
    dataset_name="social_media_discourse_analysis",
    config=self.config
)

# CORRECT:
self.context = GraphRAGContext(
    target_dataset_name="social_media_discourse_analysis",
    main_config=self.config
)
```

### Fix 3: Import errors
Ensure all imports are correct and modules exist

## Execution Order

1. Fix syntax error in test framework
2. Test syntax fix
3. Fix GraphRAGContext initialization
4. Test context fix with minimal example
5. Test single question analysis
6. Test full framework
7. Run complete demonstration

## Validation

After all fixes:
```bash
# Run all tests
python minimal_context_test.py
python minimal_discourse_test.py
python test_single_question.py
python test_discourse_analysis_framework.py
python demo_policy_discourse_analysis.py
```

All must complete without errors before considering the system working.