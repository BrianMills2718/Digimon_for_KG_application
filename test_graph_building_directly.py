#!/usr/bin/env python3
"""Test graph building directly to diagnose issues"""

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

async def test_graph_building():
    # Setup
    config = Config.default()
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    dataset_name = "Synthetic_Test"
    
    # Get chunks
    chunks = await chunk_factory.get_chunks_for_dataset(dataset_name)
    print(f"Loaded {len(chunks)} chunks for {dataset_name}")
    
    # Test each graph type
    graph_types = ["er_graph", "rkg_graph", "tree_graph", "tree_graph_balanced", "passage_graph"]
    
    for graph_type in graph_types:
        print(f"\n{'='*60}")
        print(f"Testing {graph_type}")
        print('='*60)
        
        try:
            # Create config for this graph type
            temp_config = config.model_copy(deep=True)
            temp_config.graph.type = graph_type
            
            # Create graph instance
            graph_instance = get_graph(
                config=temp_config,
                llm=llm,
                encoder=encoder
            )
            
            # Set namespace
            if hasattr(graph_instance._graph, 'namespace'):
                graph_instance._graph.namespace = chunk_factory.get_namespace(dataset_name, graph_type=graph_type)
                print(f"Set namespace to: {graph_instance._graph.namespace.path}")
            
            # Build graph
            print(f"Building {graph_type}...")
            success = await graph_instance.build_graph(chunks=chunks, force=True)
            
            if success:
                print(f"✓ {graph_type} built successfully!")
                
                # Get counts
                node_count = None
                edge_count = None
                if hasattr(graph_instance, 'node_num'):
                    node_count = graph_instance.node_num
                    if callable(node_count):
                        node_count = node_count()
                elif hasattr(graph_instance._graph, 'get_node_num'):
                    node_count = graph_instance._graph.get_node_num()
                    
                if hasattr(graph_instance, 'edge_num'):
                    edge_count = graph_instance.edge_num
                    if callable(edge_count):
                        edge_count = edge_count()
                elif hasattr(graph_instance._graph, 'get_edge_num'):
                    edge_count = graph_instance._graph.get_edge_num()
                
                print(f"  Nodes: {node_count}, Edges: {edge_count}")
                
                # Check file existence
                expected_paths = [
                    f"results/{dataset_name}/{graph_type}/nx_data.graphml",
                    f"results/{dataset_name}/{graph_type}/graph_storage/nx_data.graphml",
                    f"results/{dataset_name}/{graph_type}/tree_data.pkl",
                    f"results/{dataset_name}/{graph_type}/tree_data_leaves.pkl"
                ]
                
                for path in expected_paths:
                    if os.path.exists(path):
                        print(f"  ✓ Found file: {path}")
                        
            else:
                print(f"✗ {graph_type} build failed!")
                
        except Exception as e:
            print(f"✗ Error building {graph_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("GRAPH BUILDING TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_graph_building())