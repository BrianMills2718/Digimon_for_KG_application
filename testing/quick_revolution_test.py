#!/usr/bin/env python3

"""
Quick test script to run revolution documents through DIGIMON pipeline
Uses the working pipeline architecture from test_agent_corpus_to_graph_pipeline.py
"""

import os
import sys
import asyncio
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Option.Config2 import Config
from Core.AgentSchema.context import GraphRAGContext
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory

# Import tools
from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
from Core.AgentTools.graph_construction_tools import build_er_graph
from Core.AgentTools.entity_vdb_tools import entity_vdb_build_tool
from Core.AgentTools.entity_tools import entity_vdb_search_tool
from Core.AgentTools.entity_onehop_tools import entity_onehop_neighbors_tool

# Import contracts
from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs
from Core.AgentSchema.graph_construction_tool_contracts import BuildERGraphInputs
from Core.AgentSchema.tool_contracts import EntityVDBBuildInputs, EntityVDBSearchInputs, EntityOneHopInput

async def test_revolution_pipeline():
    """Test the revolution documents through the full pipeline"""
    
    print("🚀 Testing Revolution Documents with DIGIMON GraphRAG")
    print("=" * 60)
    
    # Load config and initialize context
    config = Config.default()
    llm_instance = LiteLLMProvider(config.llm)
    encoder_instance = get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    # Create GraphRAG context
    context = GraphRAGContext(
        target_dataset_name="revolution_test",
        main_config=config,
        llm_provider=llm_instance,
        embedding_provider=encoder_instance,
        chunk_storage_manager=chunk_factory
    )
    
    try:
        # Step 1: Prepare corpus
        print("\n📚 Step 1: Preparing corpus from revolution documents...")
        corpus_result = await prepare_corpus_from_directory(
            PrepareCorpusInputs(
                input_directory_path="/home/brian/digimon/Data/Revolutions_Small",
                output_directory_path="./results/revolution_test",
                target_corpus_name="revolution_test"
            ),
            context
        )
        print(f"✅ Corpus prepared: {corpus_result.corpus_json_path}")
        print(f"   Documents processed: {corpus_result.document_count}")
        
        # Step 2: Build ER Graph
        print("\n🕸️  Step 2: Building entity-relationship graph...")
        graph_result = await build_er_graph(
            BuildERGraphInputs(
                target_dataset_name="revolution_test",
                force_rebuild=True
            ),
            config,
            llm_instance,
            encoder_instance,
            chunk_factory
        )
        print(f"✅ ER Graph built: {graph_result.graph_id}")
        print(f"   Status: {graph_result.status}")
        
        # Load the graph from disk and register it
        from Core.Graph.GraphFactory import get_graph as get_graph_factory_instance
        from Core.Storage.NetworkXStorage import NetworkXStorage
        from Core.Storage.NameSpace import Workspace, NameSpace
        
        # Create a temporary config for loading the graph
        temp_config_for_load = config.model_copy(deep=True)
        temp_config_for_load.graph.type = "er_graph"
        
        # Create graph instance for loading
        graph_instance = get_graph_factory_instance(
            config=temp_config_for_load,
            llm=llm_instance,
            encoder=encoder_instance,
            storage_instance=NetworkXStorage()
        )
        
        # Configure storage namespace for loading
        if hasattr(graph_instance._graph, 'namespace') and isinstance(graph_instance._graph, NetworkXStorage):
            ws = Workspace(working_dir=str(config.working_dir), exp_name="revolution_test")
            ns = NameSpace(workspace=ws, namespace="er_graph")
            graph_instance._graph.namespace = ns
            graph_instance._graph.name = "nx_data.graphml"
        
        # Load the persisted graph data
        await graph_instance._load_graph()
        print(f"   📊 Loaded graph with {graph_instance.node_num} nodes, {graph_instance.edge_num} edges")
        
        # Register the loaded graph in context
        context.add_graph_instance("revolution_test_ERGraph", graph_instance)
        
        # Step 3: Build Entity VDB  
        print("\n🔍 Step 3: Building entity vector database...")
        
        # Get the graph instance from context
        graph_instance = context.get_graph_instance("revolution_test_ERGraph")
        if graph_instance:
            # Create VDB with proper configuration
            from Core.Index.FaissIndex import FaissIndex
            from Core.Storage.PickleBlobStorage import PickleBlobStorage
            from Core.Index.Schema import FAISSIndexConfig
            import uuid
            from pathlib import Path
            
            # Setup VDB storage
            vdb_path = Path("results/revolution_test/vdb")
            vdb_path.mkdir(parents=True, exist_ok=True)
            
            faiss_ws = Workspace(working_dir=str(vdb_path), exp_name="revolution_entities")
            faiss_ns = NameSpace(workspace=faiss_ws, namespace="")
            
            faiss_config = FAISSIndexConfig(
                collection_name="revolution_entities",
                path_suffix="",
                persist_path=str(vdb_path),
                embed_model=encoder_instance
            )
            
            entity_vdb = FaissIndex(config=faiss_config)
            entity_vdb.storage_instance = PickleBlobStorage(namespace=faiss_ns, name="revolution_entities_vdb")
            
            # Get node data from graph
            nodes_data = await graph_instance.nodes_data()
            print(f"   🔍 Found {len(nodes_data)} nodes in graph")
            
            # Debug: Check node structure
            if nodes_data:
                print(f"   🔍 Sample node keys: {list(nodes_data[0].keys())}")
                for i, node in enumerate(nodes_data[:3]):  # Show first 3 nodes
                    print(f"   🔍 Node {i}: {node}")
            
            elements = [
                {"id": str(node.get("entity_name", node.get("id", uuid.uuid4().hex))), 
                 "content": node.get("description") or node.get("content") or str(node.get("entity_name",""))
                } for node in nodes_data if node.get("description") or node.get("content") or node.get("entity_name")
            ]
            
            if elements:
                print(f"   📊 Indexing {len(elements)} entities in VDB")
                await entity_vdb.build_index(
                    elements=elements,
                    meta_data=["id", "content", "name"],
                    force=True
                )
                
                # Register VDB in context
                vdb_id = "revolution_entities_vdb"
                context.add_vdb_instance(vdb_id, entity_vdb)
                print(f"✅ Entity VDB built: {vdb_id}")
                print(f"   Status: Success - {len(elements)} entities indexed")
            else:
                print("❌ No entities found with content in graph")
                vdb_id = ""
        else:
            print("❌ Graph instance not found in context")
            vdb_id = ""
        
        # Step 4: Test semantic search
        print("\n🎯 Step 4: Testing semantic search...")
        search_result = await entity_vdb_search_tool(
            EntityVDBSearchInputs(
                query_text="King Louis XVI taxation revolution",
                vdb_reference_id=vdb_id,
                top_k_results=5
            ),
            context
        )
        print(f"✅ Found {len(search_result.similar_entities)} entities:")
        for i, entity in enumerate(search_result.similar_entities, 1):
            print(f"   {i}. {entity.entity_name} (score: {entity.score:.3f})")
            
        # Step 5: Test one-hop neighbors on first entity
        if search_result.similar_entities:
            print("\n🔗 Step 5: Testing one-hop neighbors...")
            first_entity = search_result.similar_entities[0]
            neighbors_result = await entity_onehop_neighbors_tool(
                {
                    "entity_ids": [first_entity.entity_name],
                    "graph_id": "revolution_test_ERGraph"
                },
                context
            )
            print(f"✅ Found {len(neighbors_result.get('relationships', []))} relationships:")
            for i, rel in enumerate(neighbors_result.get('relationships', [])[:3], 1):  # Show first 3
                print(f"   {i}. {rel.get('source_entity', 'Unknown')} -> {rel.get('target_entity', 'Unknown')} ({rel.get('relationship_type', 'Unknown')})")
                
        print("\n🎉 Revolution pipeline test completed successfully!")
        print(f"   📊 {corpus_result.document_count} documents processed")
        print(f"   🕸️  Graph: {graph_result.graph_id}")
        print(f"   🔍 VDB: {vdb_id}")
        print(f"   🎯 {len(search_result.similar_entities)} entities found for query")
        
    except Exception as e:
        logger.error(f"Error in revolution pipeline test: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(test_revolution_pipeline())
