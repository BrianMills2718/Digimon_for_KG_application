#!/usr/bin/env python3
"""Test basic graph building without agent"""

import asyncio
from pathlib import Path
import sys
sys.path.append('.')

from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Graph.GraphFactory import get_graph
from Core.Chunk.ChunkFactory import ChunkFactory
from Option.Config2 import Config

async def test_basic_graph():
    print("Testing basic graph building...")
    
    # Load config using default method
    config = Config.default()
    config.graph.type = "er_graph"
    
    # Initialize components
    llm = LiteLLMProvider(config.llm)
    encoder = get_rag_embedding(config.embedding.api_type, config)
    
    # Initialize ChunkFactory first
    chunk_factory = ChunkFactory(config=config)
    
    # Create graph instance
    print("\n1. Creating ER graph instance...")
    try:
        graph = get_graph(
            config=config,
            llm=llm,
            encoder=encoder
        )
        # Set namespace for the graph using ChunkFactory
        if hasattr(graph._graph, 'namespace'):
            graph._graph.namespace = chunk_factory.get_namespace("Russian_Troll_Sample", "er_graph")
        print("✓ Graph instance created successfully")
    except Exception as e:
        print(f"✗ Failed to create graph: {e}")
        return
    
    # Load chunks
    print("\n2. Loading chunks...")
    chunks = await chunk_factory.get_chunks_for_dataset("Russian_Troll_Sample")
    
    if chunks:
        print(f"✓ Loaded {len(chunks)} chunks")
        # Check if chunks are tuples or objects
        if isinstance(chunks[0], tuple):
            print(f"  First chunk preview: {str(chunks[0])[:100]}...")
        else:
            print(f"  First chunk preview: {chunks[0].content[:100]}...")
    else:
        print("✗ No chunks found")
        return
    
    # Build graph
    print("\n3. Building graph...")
    try:
        success = await graph.build_graph(chunks=chunks)
        if success:
            print("✓ Graph built successfully!")
            # Get some stats
            if hasattr(graph, '_graph'):
                if hasattr(graph._graph, 'number_of_nodes'):
                    print(f"  Nodes: {graph._graph.number_of_nodes()}")
                    print(f"  Edges: {graph._graph.number_of_edges()}")
        else:
            print("✗ Graph build returned False")
    except Exception as e:
        print(f"✗ Graph build failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_basic_graph())