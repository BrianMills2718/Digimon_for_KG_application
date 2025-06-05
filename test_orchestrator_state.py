#!/usr/bin/env python3
"""Test orchestrator state preservation between ReAct iterations"""

import asyncio
import sys
sys.path.append('.')

from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, DynamicToolChainConfig
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Common.Logger import logger
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory

async def test_state_preservation():
    """Test that orchestrator preserves step_outputs between plan executions"""
    
    # Initialize components
    config = Config.default()
    # Use the config as-is from Config2.yaml
    
    llm = create_llm_instance(config.llm)
    # Create embedding instance
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    context = GraphRAGContext(main_config=config, target_dataset_name="Russian_Troll_Sample")
    
    orchestrator = AgentOrchestrator(
        main_config=config,
        llm_instance=llm,
        encoder_instance=encoder,
        chunk_factory=chunk_factory,
        graphrag_context=context
    )
    
    print("Testing orchestrator state preservation...")
    print("=" * 60)
    
    # Plan 1: Prepare corpus
    plan1 = ExecutionPlan(
        plan_id="test_plan_1",
        plan_description="Prepare corpus",
        target_dataset_name="Russian_Troll_Sample",
        plan_inputs={"main_query": "test"},
        steps=[
            ExecutionStep(
                step_id="step_1_prepare_corpus",
                description="Prepare corpus",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="corpus.PrepareFromDirectory",
                            inputs={
                                "input_directory_path": "Data/Russian_Troll_Sample",
                                "output_directory_path": "results/Russian_Troll_Sample/corpus",
                                "target_corpus_name": "Russian_Troll_Sample"
                            },
                            named_outputs={
                                "prepared_corpus_name": "corpus_json_path"
                            }
                        )
                    ]
                )
            )
        ]
    )
    
    # Execute plan 1
    print("\nExecuting Plan 1 (Prepare Corpus)...")
    results1 = await orchestrator.execute_plan(plan1)
    
    print(f"\nStep outputs after Plan 1: {list(orchestrator.step_outputs.keys())}")
    for step_id, outputs in orchestrator.step_outputs.items():
        print(f"  {step_id}: {list(outputs.keys())}")
    
    # Plan 2: Build ER graph (references output from plan 1)
    plan2 = ExecutionPlan(
        plan_id="test_plan_2",
        plan_description="Build ER graph",
        target_dataset_name="Russian_Troll_Sample",
        plan_inputs={"main_query": "test"},
        steps=[
            ExecutionStep(
                step_id="step_2_build_er_graph",
                description="Build ER graph",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="graph.BuildERGraph",
                            inputs={
                                "target_dataset_name": "Russian_Troll_Sample",
                                "corpus_ref": {"from_step_id": "step_1_prepare_corpus", "named_output_key": "prepared_corpus_name"}
                            },
                            named_outputs={
                                "er_graph_id": "graph_id"
                            }
                        )
                    ]
                )
            )
        ]
    )
    
    # Execute plan 2
    print("\nExecuting Plan 2 (Build ER Graph)...")
    
    # Check if step_outputs are preserved
    initial_outputs = dict(orchestrator.step_outputs)
    results2 = await orchestrator.execute_plan(plan2)
    
    print(f"\nStep outputs after Plan 2: {list(orchestrator.step_outputs.keys())}")
    for step_id, outputs in orchestrator.step_outputs.items():
        print(f"  {step_id}: {list(outputs.keys())}")
    
    # Verify state preservation
    if "step_1_prepare_corpus" in orchestrator.step_outputs:
        print("\n✓ SUCCESS: Orchestrator preserved state from Plan 1!")
        print(f"  - Plan 1 outputs still available: {list(orchestrator.step_outputs['step_1_prepare_corpus'].keys())}")
    else:
        print("\n✗ FAILURE: Orchestrator lost state from Plan 1!")
        
    # Check if Plan 2 could reference Plan 1's outputs
    if "step_2_build_er_graph" in orchestrator.step_outputs:
        step2_outputs = orchestrator.step_outputs["step_2_build_er_graph"]
        if "er_graph_id" in step2_outputs or "graph_id" in step2_outputs:
            print("✓ Plan 2 executed successfully with reference to Plan 1's output")
        elif "error" in str(step2_outputs):
            print(f"✗ Plan 2 failed: {step2_outputs}")
    
    return orchestrator.step_outputs

if __name__ == "__main__":
    results = asyncio.run(test_state_preservation())
    print("\nFinal state:")
    print(f"Total steps tracked: {len(results)}")