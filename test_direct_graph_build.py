#!/usr/bin/env python3
"""Direct test of graph building without agent"""

import asyncio
import sys
sys.path.append('.')

from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.AgentTools.graph_construction_tools import build_rk_graph, build_tree_graph
from Core.AgentSchema.graph_construction_tool_contracts import BuildRKGraphInputs, BuildTreeGraphInputs

async def test_direct_graph_build():
    print("DIRECT GRAPH BUILD TEST")
    print("=" * 80)
    
    # Setup
    config = Config.default()
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    # Test RK Graph
    print("\n1. Building RK Graph directly...")
    rk_input = BuildRKGraphInputs(target_dataset_name="Synthetic_Test")
    rk_result = await build_rk_graph(
        tool_input=rk_input,
        main_config=config,
        llm_instance=llm,
        encoder_instance=encoder,
        chunk_factory=chunk_factory
    )
    print(f"   Status: {rk_result.status}")
    print(f"   Message: {rk_result.message}")
    print(f"   Graph ID: {rk_result.graph_id}")
    print(f"   Artifact Path: {rk_result.artifact_path}")
    print(f"   Node Count: {rk_result.node_count}")
    print(f"   Edge Count: {rk_result.edge_count}")
    
    # Test Tree Graph
    print("\n2. Building Tree Graph directly...")
    tree_input = BuildTreeGraphInputs(target_dataset_name="Synthetic_Test")
    tree_result = await build_tree_graph(
        tool_input=tree_input,
        main_config=config,
        llm_instance=llm,
        encoder_instance=encoder,
        chunk_factory=chunk_factory
    )
    print(f"   Status: {tree_result.status}")
    print(f"   Message: {tree_result.message}")
    print(f"   Graph ID: {tree_result.graph_id}")
    print(f"   Artifact Path: {tree_result.artifact_path}")
    print(f"   Node Count: {tree_result.node_count}")
    print(f"   Layer Count: {tree_result.layer_count}")
    
    return rk_result.status == "success" and tree_result.status == "success"

if __name__ == "__main__":
    success = asyncio.run(test_direct_graph_build())
    exit(0 if success else 1)