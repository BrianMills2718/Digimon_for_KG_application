#!/usr/bin/env python3
"""
Stage 3: Corpus Path Standardization
Goal: Ensure corpus files are found regardless of creation method
"""
import asyncio
import json
import shutil
from pathlib import Path
from Core.AgentTools.corpus_tools import PrepareCorpusFromDirectoryTool
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Graph.GraphFactory import GraphFactory
from Option.Config2 import Config
from Core.Common.Logger import logger
from Core.Provider.LiteLLMProvider import LiteLLMProvider

async def test_corpus_paths():
    """Test corpus path handling between tools"""
    print("\n" + "="*80)
    print("STAGE 3: Corpus Path Standardization")
    print("="*80)
    
    # Setup
    config = Config.default()
    llm = LiteLLMProvider(config.llm)
    
    # Test dataset name
    test_dataset = "TestCorpusPath"
    test_dir = Path(f"Data/{test_dataset}")
    
    # Clean up any existing test data
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    
    # Create test text files
    (test_dir / "test1.txt").write_text("This is test document 1 about AI and technology.")
    (test_dir / "test2.txt").write_text("This is test document 2 about social networks.")
    
    print(f"\nTest dataset: {test_dataset}")
    print(f"Created test files in: {test_dir}")
    
    # Step 1: Use corpus preparation tool
    print("\n1. Testing corpus preparation tool...")
    corpus_tool = PrepareCorpusFromDirectoryTool()
    
    try:
        corpus_result = await corpus_tool.run(
            corpus_directory=str(test_dir),
            output_name=test_dataset
        )
        
        print(f"Corpus tool status: {corpus_result.status}")
        print(f"Corpus tool message: {corpus_result.message}")
        
        # Find where corpus was actually created
        possible_paths = [
            Path(f"results/{test_dataset}/corpus/Corpus.json"),
            Path(f"results/{test_dataset}/Corpus.json"),
            Path(f"Data/{test_dataset}/Corpus.json"),
            Path(corpus_result.corpus_json_path) if hasattr(corpus_result, 'corpus_json_path') else None
        ]
        
        corpus_created_at = None
        for path in possible_paths:
            if path and path.exists():
                corpus_created_at = path
                break
        
        print(f"\nCorpus created at: {corpus_created_at}")
        
    except Exception as e:
        print(f"❌ Corpus tool error: {e}")
        corpus_created_at = None
    
    # Step 2: Test ChunkFactory
    print("\n2. Testing ChunkFactory...")
    chunk_factory = ChunkFactory(config)
    
    # Check where ChunkFactory looks for corpus
    expected_paths = [
        Path(config.working_dir) / test_dataset / "Corpus.json",
        Path(config.data_root) / test_dataset / "Corpus.json"
    ]
    
    print(f"\nChunkFactory expects corpus at:")
    for path in expected_paths:
        exists = "✓ EXISTS" if path.exists() else "✗ NOT FOUND"
        print(f"  - {path} [{exists}]")
    
    # Try loading chunks
    try:
        chunks = await chunk_factory.get_chunks_for_dataset(test_dataset)
        chunks_loaded = len(chunks)
        print(f"\nChunks loaded: {chunks_loaded}")
        
        if chunks:
            print("Sample chunk:")
            chunk_id, chunk_content = chunks[0]
            print(f"  ID: {chunk_id}")
            print(f"  Content: {chunk_content.content[:100]}...")
    except Exception as e:
        print(f"❌ ChunkFactory error: {e}")
        chunks_loaded = 0
    
    # Step 3: Test graph building
    print("\n3. Testing graph building with chunks...")
    graph_built = False
    
    if chunks_loaded > 0:
        try:
            # Try building a simple ER graph
            graph_factory = GraphFactory(config)
            graph_factory.llm = llm
            
            namespace = chunk_factory.get_namespace(test_dataset, "er_graph")
            graph = await graph_factory.build(
                chunks=chunks,
                namespace=namespace,
                method_config={'graph': {'type': 'er_graph'}}
            )
            
            if graph:
                nodes = await graph.get_nodes()
                edges = await graph.get_edges()
                print(f"Graph built successfully!")
                print(f"  Nodes: {len(nodes)}")
                print(f"  Edges: {len(edges)}")
                graph_built = True
        except Exception as e:
            print(f"❌ Graph building error: {e}")
    
    # Final verdict
    print("\n" + "="*80)
    print("EVIDENCE SUMMARY:")
    print(f"- corpus_created_at: {corpus_created_at}")
    print(f"- corpus_expected_at: {expected_paths[0]} (primary)")
    print(f"- chunks_loaded: {chunks_loaded}")
    print(f"- graph_built: {'success' if graph_built else 'failed'}")
    
    # Check if paths match
    path_mismatch = False
    if corpus_created_at and not any(corpus_created_at == path for path in expected_paths):
        path_mismatch = True
        print(f"\n⚠️ PATH MISMATCH DETECTED!")
        print(f"  Tool creates at: {corpus_created_at}")
        print(f"  ChunkFactory expects: {expected_paths}")
    
    # Clean up test data
    if test_dir.exists():
        shutil.rmtree(test_dir)
    results_dir = Path(f"results/{test_dataset}")
    if results_dir.exists():
        shutil.rmtree(results_dir)
    
    if chunks_loaded > 0 and not path_mismatch:
        print("\n✅ STAGE 3: PASSED - Corpus paths are properly handled!")
        return True
    else:
        print("\n❌ STAGE 3: FAILED - Corpus path issues detected")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_corpus_paths())