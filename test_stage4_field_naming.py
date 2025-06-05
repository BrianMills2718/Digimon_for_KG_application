#!/usr/bin/env python3
"""
Test Stage 4: Field Naming Fixes
Tests that the agent correctly handles tool input/output field names
"""
import asyncio
from pathlib import Path
import json

from Option.Config2 import Config
from Core.GraphRAG import GraphRAG
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger

async def test_field_naming():
    """Test that field naming and tool validation work correctly"""
    print("\n=== Stage 4: Field Naming Test ===")
    
    # Initialize system
    config = Config.default()
    config.retriever.type = "naive"
    
    graphrag = GraphRAG(config=config)
    
    # Create test query that would trigger field naming issues
    test_query = "Prepare the corpus and build a graph for russian troll tweets"
    corpus_name = "Russian_Troll_Sample"
    
    # Ensure test data exists
    test_data_path = Path(f"Data/{corpus_name}")
    if not test_data_path.exists():
        print(f"Creating test data at {test_data_path}")
        test_data_path.mkdir(parents=True, exist_ok=True)
        (test_data_path / "test.txt").write_text("Test Russian troll tweet content")
    
    # Test with the agent
    try:
        result = await graphrag.agent_query(test_query, corpus_name=corpus_name)
        
        print("\nQuery Result:")
        if isinstance(result, dict):
            # Check if we got proper results
            if "generated_answer" in result:
                print(f"Answer: {result['generated_answer']}")
            
            # Check for errors
            if "error" in result:
                print(f"Error occurred: {result['error']}")
                return False
            
            # Check for tool validation warnings in logs
            if "retrieved_context" in result:
                context = result["retrieved_context"]
                print(f"\nContext keys: {list(context.keys())}")
                
                # Check if corpus was prepared
                corpus_prepared = any("corpus" in str(k).lower() and "success" in str(v).get("status", "").lower() 
                                    for k, v in context.items() if isinstance(v, dict))
                print(f"Corpus prepared: {corpus_prepared}")
                
                # Check if graph was built  
                graph_built = any("graph" in str(k).lower() and "success" in str(v).get("status", "").lower() 
                                for k, v in context.items() if isinstance(v, dict))
                print(f"Graph built: {graph_built}")
                
                return corpus_prepared or graph_built
        else:
            print(f"Unexpected result type: {type(result)}")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_tool_validation():
    """Test that invalid tools are caught"""
    print("\n=== Stage 4: Tool Validation Test ===")
    
    config = Config.default()
    graphrag = GraphRAG(config=config)
    
    # Create a planning agent
    agent = graphrag.planning_agent
    
    # Test validation with a plan that has invalid tools
    from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, DynamicToolChainConfig
    
    invalid_plan = ExecutionPlan(
        plan_id="test_invalid",
        plan_description="Test plan with invalid tools",
        target_dataset_name="test",
        plan_inputs={"main_query": "test"},
        steps=[
            ExecutionStep(
                step_id="step1",
                description="Use non-existent tool",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Chunk.GetTextForClusters",  # This tool doesn't exist
                            inputs={"test": "value"}
                        )
                    ]
                )
            ),
            ExecutionStep(
                step_id="step2", 
                description="Use another non-existent tool",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="report.Generate",  # This tool doesn't exist either
                            inputs={"test": "value"}
                        )
                    ]
                )
            )
        ]
    )
    
    # Validate the plan
    errors = agent._validate_plan_tools(invalid_plan)
    
    print(f"\nValidation found {len(errors)} errors:")
    for error in errors:
        print(f"  - {error}")
    
    # Test should pass if we found the invalid tools
    return len(errors) == 2

async def main():
    """Run all Stage 4 tests"""
    print("STAGE 4 TESTS: Field Naming and Tool Validation")
    print("=" * 60)
    
    # Test 1: Field naming with real query
    test1_passed = await test_field_naming()
    print(f"\n✓ Field naming test: {'PASSED' if test1_passed else 'FAILED'}")
    
    # Test 2: Tool validation  
    test2_passed = await test_tool_validation()
    print(f"✓ Tool validation test: {'PASSED' if test2_passed else 'FAILED'}")
    
    # Summary
    all_passed = test1_passed and test2_passed
    print(f"\n{'='*60}")
    print(f"STAGE 4 RESULT: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print(f"{'='*60}")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)