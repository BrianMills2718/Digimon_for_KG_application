#!/usr/bin/env python3
"""Test RK graph building specifically"""

import asyncio
import sys
import os
sys.path.append('.')

from Core.Graph.GraphFactory import get_graph
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Common.Logger import logger

async def test_rk_graph():
    # Setup
    config = Config.default()
    
    # Use a faster model for testing
    config.llm.model = "o4-mini"
    
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    dataset_name = "Synthetic_Test"
    
    # Get chunks
    chunks = await chunk_factory.get_chunks_for_dataset(dataset_name)
    print(f"Loaded {len(chunks)} chunks for {dataset_name}")
    
    # Test RK graph
    print("\nTesting RK Graph building...")
    print("=" * 60)
    
    try:
        # Create config for RK graph
        temp_config = config.model_copy(deep=True)
        temp_config.graph.type = "rkg_graph"
        
        # Create graph instance
        graph_instance = get_graph(
            config=temp_config,
            llm=llm,
            encoder=encoder
        )
        
        # Set namespace
        if hasattr(graph_instance._graph, 'namespace'):
            graph_instance._graph.namespace = chunk_factory.get_namespace(dataset_name, graph_type="rkg_graph")
            print(f"Set namespace to: {graph_instance._graph.namespace.path}")
        
        # Build graph with only first 2 chunks to speed up
        print(f"Building RK graph with first 2 chunks...")
        small_chunks = chunks[:2]
        success = await graph_instance.build_graph(chunks=small_chunks, force=True)
        
        if success:
            print(f"✓ RK graph built successfully!")
            
            # Check if file was created
            expected_paths = [
                f"results/{dataset_name}/rkg_graph/nx_data.graphml",
                f"results/{dataset_name}/rkg_graph/graph_storage/nx_data.graphml"
            ]
            
            for path in expected_paths:
                if os.path.exists(path):
                    print(f"✓ Found file: {path}")
                    size = os.path.getsize(path)
                    print(f"  Size: {size:,} bytes")
        else:
            print(f"✗ RK graph build returned False")
            
    except Exception as e:
        print(f"✗ Error building RK graph: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rk_graph())