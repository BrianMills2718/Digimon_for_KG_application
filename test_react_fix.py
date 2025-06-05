#!/usr/bin/env python3
"""Test that ReAct mode works with the orchestrator state preservation fix"""

import asyncio
import sys
sys.path.append('.')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory

async def test_react_mode():
    """Test ReAct mode with a simple query"""
    
    # Initialize components
    config = Config.default()
    llm = create_llm_instance(config.llm)
    
    # Create embedding instance
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    
    chunk_factory = ChunkFactory(config)
    context = GraphRAGContext(
        main_config=config,
        target_dataset_name="Russian_Troll_Sample",
        llm_provider=llm,
        embedding_provider=encoder,
        chunk_storage_manager=chunk_factory
    )
    
    orchestrator = AgentOrchestrator(
        main_config=config,
        llm_instance=llm,
        encoder_instance=encoder,
        chunk_factory=chunk_factory,
        graphrag_context=context
    )
    
    agent = PlanningAgent(
        config=config,
        graphrag_context=context
    )
    
    print("Testing ReAct mode with orchestrator state preservation fix...")
    print("=" * 60)
    
    # Test a query that requires multiple steps
    query = "Build an ER graph for Russian_Troll_Sample and then find entities related to Trump"
    
    try:
        result = await agent.process_query_react(query, "Russian_Troll_Sample")
        
        print("\nReAct Test Results:")
        print(f"- Iterations: {result.get('iterations', 0)}")
        print(f"- Steps executed: {len(result.get('executed_steps', []))}")
        print(f"- Answer generated: {'Yes' if result.get('generated_answer') else 'No'}")
        
        # Check if the orchestrator maintained state
        if hasattr(orchestrator, 'step_outputs'):
            print(f"\nOrchestrator state preservation:")
            print(f"- Total steps tracked: {len(orchestrator.step_outputs)}")
            for step_id in list(orchestrator.step_outputs.keys())[:5]:  # Show first 5
                print(f"  - {step_id}: {list(orchestrator.step_outputs[step_id].keys())}")
                
        # Print the answer
        answer = result.get('generated_answer', 'No answer generated')
        print(f"\nGenerated Answer:")
        print(answer[:500] + "..." if len(answer) > 500 else answer)
        
        # Success criteria
        if result.get('iterations', 0) > 0 and len(result.get('executed_steps', [])) > 0:
            print("\n✓ SUCCESS: ReAct mode is working!")
        else:
            print("\n✗ FAILURE: ReAct mode did not execute properly")
            
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_react_mode())