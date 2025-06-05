#!/usr/bin/env python3
"""Test ReAct mode specifically to diagnose the issue"""

import asyncio
import sys
sys.path.append('.')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LiteLLMProvider import LiteLLMProvider

async def test_react():
    # Initialize components
    config = Config.default()
    llm = LiteLLMProvider(config.llm)
    context = GraphRAGContext(main_config=config, target_dataset_name="Russian_Troll_Sample")
    orchestrator = AgentOrchestrator(main_config=config, graphrag_context=context)
    
    agent = PlanningAgent(
        llm_provider=llm,
        orchestrator=orchestrator,
        graphrag_context=context
    )
    
    # Test query
    query = "Build an ER graph and list entities"
    
    print("Testing ReAct mode...")
    result = await agent.process_query_react(query, actual_corpus_name="Russian_Troll_Sample")
    
    print("\nResult:")
    print(f"Error: {result.get('error', 'None')}")
    print(f"Answer: {result.get('final_answer', result.get('generated_answer', 'No answer'))[:200]}...")
    
    # Check orchestrator state
    print("\nOrchestrator step outputs:")
    for step_id, outputs in orchestrator.step_outputs.items():
        print(f"  {step_id}: {list(outputs.keys())}")

if __name__ == "__main__":
    asyncio.run(test_react())