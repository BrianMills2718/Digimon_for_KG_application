#!/usr/bin/env python3
"""Test a single graph build to verify the fix"""

import asyncio
import sys
sys.path.append('.')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory

async def test_graph_build():
    config = Config.default()
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    context = GraphRAGContext(
        main_config=config,
        target_dataset_name="Synthetic_Test",
        llm_provider=llm,
        embedding_provider=encoder,
        chunk_storage_manager=chunk_factory
    )
    
    agent = PlanningAgent(config=config, graphrag_context=context)
    
    print("Testing graph build with corpus preparation...")
    
    # First ensure corpus exists
    import os
    corpus_path = "results/Synthetic_Test/Corpus.json"
    if not os.path.exists(corpus_path):
        print("Corpus not found at expected location, preparing...")
        # Prepare corpus first
        prep_result = await agent.process_query(
            "Prepare the corpus from Data/Synthetic_Test directory",
            "Synthetic_Test"
        )
        print(f"Corpus preparation: {prep_result.get('generated_answer', '')[:100]}")
    
    # Now try to build graph with explicit force_rebuild
    query = """Build an ER graph for Synthetic_Test. 
    IMPORTANT: The corpus is already prepared at results/Synthetic_Test/corpus/Corpus.json.
    Use force_rebuild=true to ensure a fresh build."""
    
    result = await agent.process_query(query, "Synthetic_Test")
    
    answer = result.get("generated_answer", "")
    context = result.get("retrieved_context", {})
    
    print("\nResult:")
    print(f"Answer: {answer[:200]}...")
    
    # Check if graph was built
    if "success" in answer.lower() or "built" in answer.lower():
        print("\n✓ SUCCESS: Graph built!")
        
        # Check graph files
        graph_path = "results/Synthetic_Test/er_graph"
        if os.path.exists(graph_path):
            files = os.listdir(graph_path)
            print(f"Graph files created: {files}")
    else:
        print("\n✗ FAILURE: Graph build failed")
        
        # Debug info
        for step_id, outputs in context.items():
            if isinstance(outputs, dict) and "error" in str(outputs):
                print(f"\nError in {step_id}: {outputs}")

if __name__ == "__main__":
    asyncio.run(test_graph_build())