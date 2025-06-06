#!/usr/bin/env python3
"""Test only RK Graph building"""

import asyncio
import sys
import os
sys.path.append('.')

from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.AgentTools.graph_construction_tools import build_rk_graph
from Core.AgentSchema.graph_construction_tool_contracts import BuildRKGraphInputs

async def test_rk_graph():
    print("Testing RK Graph Build")
    print("=" * 50)
    
    # Setup
    config = Config.default()
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    # Clean up any existing RK graph
    import shutil
    rk_path = "results/Synthetic_Test/rkg_graph"
    if os.path.exists(rk_path):
        shutil.rmtree(rk_path)
        print(f"Removed existing RK graph at {rk_path}")
    
    # Build RK Graph
    print("\nBuilding RK Graph...")
    rk_input = BuildRKGraphInputs(target_dataset_name="Synthetic_Test", force_rebuild=True)
    
    try:
        rk_result = await build_rk_graph(
            tool_input=rk_input,
            main_config=config,
            llm_instance=llm,
            encoder_instance=encoder,
            chunk_factory=chunk_factory
        )
        
        print(f"Status: {rk_result.status}")
        print(f"Message: {rk_result.message}")
        print(f"Graph ID: {rk_result.graph_id}")
        print(f"Artifact Path: {rk_result.artifact_path}")
        print(f"Node Count: {rk_result.node_count}")
        print(f"Edge Count: {rk_result.edge_count}")
        
        # Check if files were created
        if rk_result.artifact_path and os.path.exists(rk_result.artifact_path):
            print(f"\nFiles in {rk_result.artifact_path}:")
            for item in os.listdir(rk_result.artifact_path):
                print(f"  - {item}")
        
        return rk_result.status == "success"
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_rk_graph())
    exit(0 if success else 1)