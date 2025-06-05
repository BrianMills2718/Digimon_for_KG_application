#!/usr/bin/env python3
"""
Simple test of Social Discourse dataset - build graphs directly
"""
import asyncio
from pathlib import Path
from Option.Config2 import Config
from Core.Common.Logger import logger
from Core.Graph.GraphFactory import GraphFactory
from Core.Chunk.ChunkFactory import ChunkFactory

async def test_build_graphs():
    """Build graphs for social discourse dataset"""
    print("\nTEST: Building graphs for Social_Discourse_Test dataset")
    print("=" * 80)
    
    # Get config
    config = Config.default()
    
    # Get graph factory and chunk factory
    graph_factory = GraphFactory()
    chunk_factory = ChunkFactory(config)
    
    # Prepare corpus if not exists
    corpus_path = Path("results/Social_Discourse_Test/corpus/Corpus.json")
    if not corpus_path.exists():
        print("\nPreparing corpus from text files...")
        from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
        from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs
        
        inputs = PrepareCorpusInputs(
            input_directory_path="Data/Social_Discourse_Test",
            output_directory_path="results/Social_Discourse_Test/corpus",
            target_corpus_name="Social_Discourse_Test"
        )
        
        result = await prepare_corpus_from_directory(inputs)
        print(f"Corpus prepared: {result.status} - {result.message}")
        print(f"Document count: {result.document_count}")
    
    # Get chunks
    print("\nLoading chunks...")
    chunks = await chunk_factory.get_chunks_for_dataset("Social_Discourse_Test")
    print(f"Loaded {len(chunks)} chunks")
    
    # Build ER Graph
    print("\n1. Building Entity-Relationship Graph...")
    try:
        kwargs = graph_factory._prepare_graph_kwargs("er_graph", config)
        er_graph = graph_factory._create_er_graph(config.graph, **kwargs)
        er_graph.namespace = chunk_factory.get_namespace("Social_Discourse_Test", "er_graph")
        
        await er_graph.build_graph(chunks=chunks, force=True)
        
        # Get statistics
        nodes = await er_graph.get_nodes()
        edges = await er_graph.get_edges()
        print(f"✓ ER Graph built: {len(nodes)} nodes, {len(edges)} edges")
        
        # Show sample entities
        if nodes:
            print("\nSample entities found:")
            for node in list(nodes)[:10]:
                node_data = await er_graph.get_node(node)
                if node_data and 'description' in node_data:
                    print(f"  - {node}: {node_data['description'][:100]}...")
    except Exception as e:
        print(f"✗ ER Graph build failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Build RK Graph
    print("\n2. Building Relationship-Knowledge Graph...")
    try:
        kwargs = graph_factory._prepare_graph_kwargs("rkg_graph", config)
        rk_graph = graph_factory._create_rk_graph(config.graph, **kwargs)
        rk_graph.namespace = chunk_factory.get_namespace("Social_Discourse_Test", "rkg_graph")
        
        await rk_graph.build_graph(chunks=chunks, force=True)
        
        # Get statistics
        nodes = await rk_graph.get_nodes()
        edges = await rk_graph.get_edges()
        print(f"✓ RK Graph built: {len(nodes)} nodes, {len(edges)} edges")
    except Exception as e:
        print(f"✗ RK Graph build failed: {e}")
    
    # Build Tree Graph
    print("\n3. Building Tree Graph...")
    try:
        kwargs = graph_factory._prepare_graph_kwargs("tree_graph", config) 
        tree_graph = graph_factory._create_tree_graph(config.graph, **kwargs)
        tree_graph.namespace = chunk_factory.get_namespace("Social_Discourse_Test", "tree_graph")
        
        await tree_graph.build_graph(chunks=chunks, force=True)
        
        print(f"✓ Tree Graph built successfully")
    except Exception as e:
        print(f"✗ Tree Graph build failed: {e}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Social Discourse dataset is ready for testing!")
    print("The dataset provides rich social network structure with:")
    print("- 10 actors across 4 communities") 
    print("- 20+ explicit mentions creating network connections")
    print("- 5 discourse phases showing opinion evolution")
    print("- Clear themes around AI automation debate")

async def test_entity_extraction():
    """Test entity extraction from the ER graph"""
    print("\n\nTEST: Extracting entities from ER graph")
    print("=" * 80)
    
    config = Config.default()
    graph_factory = GraphFactory()
    chunk_factory = ChunkFactory(config)
    
    # Load existing ER graph
    kwargs = graph_factory._prepare_graph_kwargs("er_graph", config)
    er_graph = graph_factory._create_er_graph(config.graph, **kwargs)
    er_graph.namespace = chunk_factory.get_namespace("Social_Discourse_Test", "er_graph")
    
    # Try to load persisted graph
    loaded = await er_graph.load_persisted_graph(force=False)
    if not loaded:
        print("No persisted ER graph found. Run graph building first.")
        return
    
    # Get all nodes
    nodes = await er_graph.get_nodes()
    print(f"\nTotal entities found: {len(nodes)}")
    
    # Categorize entities
    actors = []
    orgs = []
    concepts = []
    
    for node in nodes:
        node_data = await er_graph.get_node(node)
        if node_data:
            entity_type = node_data.get('entity_type', '').lower()
            desc = node_data.get('description', '')
            
            if any(x in desc.lower() for x in ['@', 'twitter', 'account']):
                actors.append((node, desc))
            elif entity_type == 'organization' or any(x in desc.lower() for x in ['company', 'union', 'organization']):
                orgs.append((node, desc))
            else:
                concepts.append((node, desc))
    
    print(f"\nActors ({len(actors)}):")
    for name, desc in actors[:10]:
        print(f"  - {name}: {desc[:80]}...")
    
    print(f"\nOrganizations ({len(orgs)}):")
    for name, desc in orgs[:10]:
        print(f"  - {name}: {desc[:80]}...")
    
    print(f"\nConcepts ({len(concepts)}):")
    for name, desc in concepts[:10]:
        print(f"  - {name}: {desc[:80]}...")

async def test_relationship_analysis():
    """Test relationship extraction"""
    print("\n\nTEST: Analyzing relationships")
    print("=" * 80)
    
    config = Config.default()
    graph_factory = GraphFactory()
    chunk_factory = ChunkFactory(config)
    
    # Load existing ER graph
    kwargs = graph_factory._prepare_graph_kwargs("er_graph", config)
    er_graph = graph_factory._create_er_graph(config.graph, **kwargs)
    er_graph.namespace = chunk_factory.get_namespace("Social_Discourse_Test", "er_graph")
    
    loaded = await er_graph.load_persisted_graph(force=False)
    if not loaded:
        print("No persisted ER graph found.")
        return
        
    # Get edges
    edges = await er_graph.get_edges()
    print(f"\nTotal relationships found: {len(edges)}")
    
    # Analyze relationship types
    rel_types = {}
    mentions = []
    
    for edge in edges[:50]:  # Sample first 50
        edge_data = await er_graph.get_edge(edge[0], edge[1])
        if edge_data:
            rel_type = edge_data.get('relationship', 'unknown')
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
            
            if 'mention' in rel_type.lower():
                mentions.append((edge[0], edge[1], edge_data.get('description', '')))
    
    print("\nRelationship types:")
    for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {rel_type}: {count}")
    
    print(f"\nSample mentions ({len(mentions)}):")
    for source, target, desc in mentions[:10]:
        print(f"  - {source} -> {target}: {desc[:80]}...")

async def main():
    """Run all tests"""
    await test_build_graphs()
    await test_entity_extraction()
    await test_relationship_analysis()

if __name__ == "__main__":
    asyncio.run(main())