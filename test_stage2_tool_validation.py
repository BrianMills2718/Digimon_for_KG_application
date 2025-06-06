#!/usr/bin/env python3
"""
Stage 2: Tool Hallucination Prevention
Goal: Ensure agent only uses tools that actually exist
"""
import asyncio
import json
from pathlib import Path
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentTools.tool_registry import DynamicToolRegistry
from Option.Config2 import Config
from Core.Common.Logger import logger
from Core.Provider.LiteLLMProvider import LiteLLMProvider

async def test_tool_validation():
    """Test that agent only uses registered tools"""
    print("\n" + "="*80)
    print("STAGE 2: Tool Hallucination Prevention")
    print("="*80)
    
    # Setup
    config = Config.default()
    llm = LiteLLMProvider(config.llm)
    
    # Get registered tools
    registry = DynamicToolRegistry()
    all_tools = registry.list_all_tools()
    registered_tools = [tool['tool_id'] for tool in all_tools]
    print(f"\nRegistered tools ({len(registered_tools)}):")
    for tool in sorted(registered_tools):
        print(f"  - {tool}")
    
    # Create planning agent
    agent_brain = PlanningAgent(config=config)
    
    # Test query that might trigger tool hallucinations
    query = """Build an ER graph for the Social_Discourse_Test dataset, 
    create a vector database index for entities, 
    search for social network actors, 
    and analyze the community clusters."""
    
    print(f"\nTest query: {query}")
    print("\nGenerating plan...")
    
    try:
        # Generate initial plan
        plan = await agent_brain.plan(query)
        
        # Extract tool IDs from plan
        plan_tools = []
        if plan and hasattr(plan, 'steps'):
            for step in plan.steps:
                if hasattr(step, 'tool_id'):
                    plan_tools.append(step.tool_id)
        
        print(f"\nTools in generated plan ({len(plan_tools)}):")
        for tool in plan_tools:
            print(f"  - {tool}")
        
        # Check for hallucinated tools
        hallucinated_tools = []
        for tool in plan_tools:
            if tool not in registered_tools:
                hallucinated_tools.append(tool)
                print(f"  ❌ HALLUCINATED: {tool}")
        
        # Create orchestrator to test execution
        orchestrator = AgentOrchestrator(registry)
        
        # Try executing the plan to see if we get "Tool not found" errors
        print("\nTesting plan execution...")
        execution_errors = []
        
        if plan and hasattr(plan, 'steps'):
            for step in plan.steps[:3]:  # Test first 3 steps only
                try:
                    print(f"\nExecuting step: {step.step_id} - {step.tool_id}")
                    output = await orchestrator.execute_step(step, plan.context)
                    
                    # Check for tool not found errors
                    if hasattr(output, 'status') and output.status == 'failure':
                        if 'not found' in str(output.message).lower():
                            execution_errors.append(f"{step.tool_id}: {output.message}")
                            print(f"  ❌ Tool not found error: {output.message}")
                    
                except Exception as e:
                    if 'not found' in str(e).lower():
                        execution_errors.append(f"{step.tool_id}: {str(e)}")
                        print(f"  ❌ Exception: {str(e)}")
        
        # Final verdict
        print("\n" + "="*80)
        print("EVIDENCE SUMMARY:")
        print(f"- registered_tools: {len(registered_tools)} tools available")
        print(f"- plan_tools: {plan_tools}")
        print(f"- validation: {'FAILED' if hallucinated_tools else 'PASSED'} - {len(hallucinated_tools)} hallucinated tools")
        print(f"- no_errors: {'FAILED' if execution_errors else 'PASSED'} - {len(execution_errors)} execution errors")
        
        if hallucinated_tools:
            print(f"\nHallucinated tools found:")
            for tool in hallucinated_tools:
                print(f"  - {tool}")
        
        if execution_errors:
            print(f"\nExecution errors:")
            for error in execution_errors:
                print(f"  - {error}")
        
        if not hallucinated_tools and not execution_errors:
            print("\n✅ STAGE 2: PASSED - No tool hallucinations!")
            return True
        else:
            print("\n❌ STAGE 2: FAILED - Tool hallucinations detected")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tool_validation())