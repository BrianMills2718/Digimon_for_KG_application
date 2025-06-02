# Core/AgentTools/entity_vdb_tools.py
"""
Entity VDB Build Tool
"""

import uuid
from typing import List, Optional
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    EntityVDBBuildInputs,
    EntityVDBBuildOutputs
)
from Core.Index.FaissIndex import FaissIndex
from Core.Common.Logger import logger
from Core.Index.Schema import FAISSIndexConfig
from Core.Storage.NameSpace import Workspace, NameSpace
from Core.Storage.PickleBlobStorage import PickleBlobStorage

# Dummy config for VDB 
class MockIndexConfig:
    def __init__(self, persist_path, embed_model, retrieve_top_k, name):
        self.persist_path = persist_path
        self.embed_model = embed_model
        self.retrieve_top_k = retrieve_top_k
        self.name = name

async def entity_vdb_build_tool(
    params: EntityVDBBuildInputs,
    graphrag_context: GraphRAGContext
) -> EntityVDBBuildOutputs:
    """
    Build a vector database (VDB) for graph entities.
    
    This tool creates a searchable index of entities from a graph,
    allowing for similarity-based retrieval of nodes based on their
    properties and descriptions.
    """
    logger.info(
        f"Building entity VDB: graph_id='{params.graph_reference_id}', "
        f"collection='{params.vdb_collection_name}'"
    )
    
    try:
        # Get the graph instance
        graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
        if not graph_instance:
            error_msg = f"Graph '{params.graph_reference_id}' not found in context"
            logger.error(error_msg)
            return EntityVDBBuildOutputs(
                vdb_reference_id="",
                num_entities_indexed=0,
                status=f"Error: {error_msg}"
            )
        
        # Get embedding provider
        embedding_provider = graphrag_context.embedding_provider
        if not embedding_provider:
            error_msg = "No embedding provider available in context"
            logger.error(error_msg)
            return EntityVDBBuildOutputs(
                vdb_reference_id="",
                num_entities_indexed=0,
                status=f"Error: {error_msg}"
            )
        
        # Check if VDB already exists
        vdb_id = params.vdb_collection_name
        existing_vdb = graphrag_context.get_vdb_instance(vdb_id)
        
        if existing_vdb and not params.force_rebuild:
            logger.info(f"VDB '{vdb_id}' already exists and force_rebuild=False, skipping build")
            # Get node count from graph
            nodes_data = await graph_instance.nodes_data()
            return EntityVDBBuildOutputs(
                vdb_reference_id=vdb_id,
                num_entities_indexed=len(nodes_data),
                status="VDB already exists"
            )
        
        # Get nodes data from graph
        nodes_data = await graph_instance.nodes_data()
        logger.info(f"Retrieved {len(nodes_data)} nodes from graph")
        
        # Prepare entity data for VDB
        entities_data = []
        for node in nodes_data:
            # Filter by entity types if specified
            if params.entity_types:
                node_type = node.get("type", node.get("entity_type", "entity"))
                if node_type not in params.entity_types:
                    continue
            
            # Create entity document
            entity_id = str(node.get("entity_name", node.get("id", uuid.uuid4().hex)))
            content = node.get("description") or node.get("content") or str(node.get("entity_name", ""))
            
            if content:  # Only add if there's content to embed
                entity_doc = {
                    "id": entity_id,
                    "content": content,
                    "name": node.get("entity_name", entity_id)
                }
                
                # Add metadata if requested
                if params.include_metadata:
                    for key, value in node.items():
                        if key not in ["id", "content", "name", "description"]:
                            entity_doc[key] = value
                
                entities_data.append(entity_doc)
        
        if not entities_data:
            logger.warning(f"No suitable entities found in graph '{params.graph_reference_id}'")
            return EntityVDBBuildOutputs(
                vdb_reference_id="",
                num_entities_indexed=0,
                status="No entities with content found in graph"
            )
        
        logger.info(f"Prepared {len(entities_data)} entities for indexing")
        
        # Create VDB storage path
        vdb_storage_path = f"storage/vdb/{vdb_id}"
        
        # Create index configuration
        config = MockIndexConfig(
            persist_path=vdb_storage_path,
            embed_model=embedding_provider,
            retrieve_top_k=10,
            name=vdb_id
        )
        
        # Create and build the index
        entity_vdb = FaissIndex(config)
        
        # Build the index using the correct method signature
        await entity_vdb.build_index(
            elements=entities_data,
            meta_data=["id", "content", "name"],
            force=params.force_rebuild
        )
        
        # Register the VDB in context
        graphrag_context.add_vdb_instance(vdb_id, entity_vdb)
        
        # Verify registration with detailed logging
        available_vdbs = list(graphrag_context._vdbs.keys()) if hasattr(graphrag_context, '_vdbs') else []
        logger.info(
            f"Entity.VDB.Build: Successfully built AND REGISTERED VDB with ID: '{vdb_id}'. "
            f"Available VDBs in context now: {available_vdbs}"
        )
        
        logger.info(
            f"Successfully built entity VDB '{vdb_id}' with "
            f"{len(entities_data)} entities indexed"
        )
        
        return EntityVDBBuildOutputs(
            vdb_reference_id=vdb_id,
            num_entities_indexed=len(entities_data),
            status=f"Successfully built VDB with {len(entities_data)} entities"
        )
        
    except Exception as e:
        error_msg = f"Error building entity VDB: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return EntityVDBBuildOutputs(
            vdb_reference_id="",
            num_entities_indexed=0,
            status=f"Error: {str(e)}"
        )
