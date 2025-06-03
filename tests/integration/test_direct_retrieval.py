#!/usr/bin/env python3
# test_direct_retrieval.py - Direct testing of retrieval tools with context sharing

import os
import sys
import asyncio
import uuid
import json
from pathlib import Path
import networkx as nx
import numpy as np
import faiss
from loguru import logger

# Add the project root to Python path to fix imports
project_root = Path("/home/brian/digimon")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the necessary components from project modules
from Config import ChunkConfig
from Option.Config2 import LLMConfig, Config, EmbeddingConfig
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.AgentSchema.context import GraphRAGContext
from Core.Graph.ERGraph import ERGraph
from Core.AgentTools.entity_tools import entity_vdb_search_tool
from Core.AgentTools.relationship_tools import relationship_one_hop_neighbors_tool
from Core.AgentSchema.tool_contracts import EntityVDBSearchInputs, RelationshipOneHopNeighborsInputs
from Core.Storage.NameSpace import Workspace


async def test_direct_retrieval():
    """Test direct retrieval from graph and VDB without agent orchestration"""
    # Initialize providers without direct Config dependency
    # Create a minimal config with LLM and embedding settings
    
    # Create minimal configs
    llm_config = LLMConfig(model="openai/o4-mini-2025-04-16", api_key="sk-dummy-key-for-testing-only", base_url="", api_type="openai", calc_usage=False)
    emb_config = EmbeddingConfig(api_type="openai", model="text-embedding-ada-002", api_key="sk-dummy-key-for-testing-only")
    
    # Create a minimal overall config
    minimal_config = Config(llm=llm_config, embedding=emb_config, exp_name="test_direct_retrieval")
    
    # Initialize providers
    llm_instance = LiteLLMProvider(config=llm_config)
    encoder_instance = get_rag_embedding(config=minimal_config)
    chunk_factory = ChunkFactory(config=minimal_config)
    
    # Create a GraphRAGContext - we'll use this directly for our test
    graphrag_context = GraphRAGContext(
        request_id="test_direct_retrieval",
        session_id="test_session",
        tool_calls_made=[],
        model_name="test-model",
        current_datetime="2025-06-01T21:45:00-07:00",
        target_dataset_name="MySampleTexts_Agent_Corpus"
    )
    
    logger.info(f"Created GraphRAGContext with ID {id(graphrag_context)}")
    
    # Now load the graph and register it in the context
    try:
        # Try to load existing graph or create a new test graph
        try:
            graph_data_path = "/home/brian/digimon/results/MySampleTexts_Agent_Corpus/er_graph/er_graph/nx_data"
            logger.info(f"Loading graph from {graph_data_path}")
            # Create storage instance with default initialization
            storage = NetworkXStorage()
            # Set the storage's namespace explicitly if needed
            storage.namespace = Workspace("MySampleTexts_Agent_Corpus")
            storage.name = "er_graph/nx_data"
            # Get the actual networkx graph object, not the return status
            # Check if load_graph is an async function and await it if it is
            if asyncio.iscoroutinefunction(storage.load_graph):
                result = await storage.load_graph(force=False)
            else:
                result = storage.load_graph(force=False)
                
            # Check if the graph was loaded
            if not hasattr(storage, '_graph') or storage._graph is None:
                raise ValueError("Graph failed to load into storage._graph")
            nx_graph = storage._graph
            logger.info(f"Successfully loaded NetworkX graph from {graph_data_path} with {len(nx_graph.nodes)} nodes")
            
            # If the graph is empty, raise an exception to create our test graph
            if len(nx_graph.nodes) == 0:
                raise ValueError("Graph has 0 nodes, creating a test graph instead")
        except Exception as e:
            logger.warning(f"Could not load existing graph: {e}. Creating a simple test graph.")
            nx_graph = nx.DiGraph()
            
            # Add our Digimon test data
            nx_graph.add_node("e1", name="Digimon", type="franchise", description="Digital Monsters anime and game franchise")
            nx_graph.add_node("e2", name="Agumon", type="character", description="Dinosaur-like Digimon partner of Tai")
            nx_graph.add_node("e3", name="Gabumon", type="character", description="Wolf-like Digimon partner of Matt")
            nx_graph.add_edge("e2", "e1", relation="part_of")
            nx_graph.add_edge("e3", "e1", relation="part_of")
            
            storage = NetworkXStorage()
            storage.namespace = Workspace("test_direct_retrieval")
            storage.name = "test_graph"
            storage._graph = nx_graph
        
        # Extract entities from graph nodes
        entities = []
        for node_id, node_data in nx_graph.nodes(data=True):
            entities.append({
                'id': node_id,
                'name': node_data.get('name', ''),
                'description': node_data.get('description', f"Entity {node_id}")
            })
        
        logger.info(f"Extracted {len(entities)} entities from graph")
        
        # Create a wrapper class for the graph to match the structure expected by tools
        class GraphWrapper:
            def __init__(self, graph):
                self.graph = graph
        
        # Create the ERGraph instance with the correct structure
        er_graph = ERGraph(config=minimal_config, llm=llm_instance, encoder=encoder_instance, storage_instance=storage)
        er_graph._graph = GraphWrapper(nx_graph)
        
        # Register the graph in our context
        graphrag_context.graph_instance = er_graph
        logger.info(f"Registered graph in context with ID {id(graphrag_context)}")
        logger.info(f"Context has graph_instance: {hasattr(graphrag_context, 'graph_instance')}")
        
        # Now create and register the entity VDB
        logger.info("Creating entity VDB for search")
        
        if entities and encoder_instance:
            try:
                # Generate embeddings for entity descriptions
                texts = [f"{e['name']}: {e['description']}" for e in entities]
                logger.info(f"Generating embeddings for {len(texts)} entities")
                
                # Use embedding generation
                embeddings = []
                for text in texts:
                    # Try different embedding methods
                    try:
                        # Try async method if available
                        if hasattr(encoder_instance, 'aembed_query'):
                            embedding = await encoder_instance.aembed_query(text)
                        # Try embed_query method
                        elif hasattr(encoder_instance, 'embed_query'):
                            embedding = encoder_instance.embed_query(text)
                        # Try embed method
                        elif hasattr(encoder_instance, 'embed'):
                            embedding = encoder_instance.embed(text)
                        # Try encode method
                        elif hasattr(encoder_instance, 'encode'):
                            embedding = encoder_instance.encode(text)
                        else:
                            raise AttributeError(f"No embedding method found on {type(encoder_instance).__name__}")
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Error generating embedding for text: {e}")
                        raise
                
                # Create FAISS index
                dimension = len(embeddings[0])
                index = faiss.IndexFlatL2(dimension)
                index.add(np.array(embeddings).astype('float32'))
                
                # Create VDB instance with the structure expected by the search tool
                entity_vdb_instance = {
                    "index": index,
                    "entities": entities,
                    "embeddings": embeddings,
                    "entity_ids": [entity['id'] for entity in entities]
                }
                
                # Register VDB in context directly
                graphrag_context.entities_vdb_instance = entity_vdb_instance
                logger.info(f"Successfully registered entity VDB with {len(entities)} entities in context {id(graphrag_context)}")
                logger.info(f"Context has entities_vdb_instance: {hasattr(graphrag_context, 'entities_vdb_instance')}")
                
                # Now directly test the entity_vdb_search_tool
                # Create search parameters
                search_params = EntityVDBSearchInputs(
                    query="digimon",
                    top_k=3
                )
                
                # Call the tool directly with our context
                logger.info("\n\n======= TESTING ENTITY VDB SEARCH TOOL =======\n")
                logger.info(f"Searching for 'digimon' in entity VDB")
                search_results = await entity_vdb_search_tool(search_params, graphrag_context)
                
                logger.info(f"Search results: {search_results}")
                
                # Now test the one-hop neighbors tool
                # Create one-hop parameters - use the first entity found in the search
                one_hop_params = RelationshipOneHopNeighborsInputs(
                    entity_id="e2",  # Agumon's ID
                    relationship_type=None  # Get all relationships
                )
                
                # Call the one-hop tool directly with our context
                logger.info("\n\n======= TESTING ONE-HOP NEIGHBORS TOOL =======\n")
                logger.info(f"Finding neighbors for entity 'e2' (Agumon)")
                neighbors_results = await relationship_one_hop_neighbors_tool(one_hop_params, graphrag_context)
                
                logger.info(f"Neighbors results: {neighbors_results}")
                
            except Exception as e:
                logger.error(f"Error during VDB creation or search: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.error("No entities found or no encoder available for VDB creation")
            
    except Exception as e:
        logger.error(f"Error during graph loading/creation: {e}")
        import traceback
        logger.error(traceback.format_exc())


# Run the test when script is executed directly
if __name__ == "__main__":
    asyncio.run(test_direct_retrieval())
