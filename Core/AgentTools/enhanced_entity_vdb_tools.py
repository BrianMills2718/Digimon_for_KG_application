# Core/AgentTools/enhanced_entity_vdb_tools.py
"""
Enhanced Entity VDB Build Tool with batch embedding support
"""

import uuid
from typing import List, Optional
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    EntityVDBBuildInputs,
    EntityVDBBuildOutputs
)
from Core.Index.EnhancedFaissIndex import EnhancedFaissIndex
from Core.Common.Logger import logger
from Core.Common.StructuredErrors import (
    StructuredError, ErrorCategory, ErrorSeverity,
    EmbeddingError
)
from Core.Index.Schema import FAISSIndexConfig
from Core.Storage.NameSpace import Workspace, NameSpace
from Core.Storage.PickleBlobStorage import PickleBlobStorage
from Core.Common.BatchEmbeddingProcessor import BatchEmbeddingProcessor
from Core.Common.PerformanceMonitor import PerformanceMonitor

# Import proper index configuration helper
from Core.AgentTools.index_config_helper import create_faiss_index_config

async def entity_vdb_build_tool(
    params: EntityVDBBuildInputs,
    graphrag_context: GraphRAGContext
) -> EntityVDBBuildOutputs:
    """
    Enhanced entity VDB build tool with batch embedding and performance monitoring.
    
    This tool creates a searchable index of entities from a graph,
    allowing for similarity-based retrieval of nodes based on their
    properties and descriptions.
    """
    logger.info(
        f"Enhanced Entity VDB Build: graph_id='{params.graph_reference_id}', "
        f"collection='{params.vdb_collection_name}'"
    )
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    try:
        with monitor.measure_operation("entity_vdb_build_total"):
            # Get the graph instance
            with monitor.measure_operation("get_graph_instance"):
                graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
                if not graph_instance:
                    raise StructuredError(
                        message=f"Graph '{params.graph_reference_id}' not found in context",
                        category=ErrorCategory.VALIDATION_ERROR,
                        severity=ErrorSeverity.ERROR,
                        context={"graph_id": params.graph_reference_id}
                    )
            
            # Get embedding provider
            embedding_provider = graphrag_context.embedding_provider
            if not embedding_provider:
                raise StructuredError(
                    message="No embedding provider available in context",
                    category=ErrorCategory.CONFIGURATION_ERROR,
                    severity=ErrorSeverity.CRITICAL
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
            with monitor.measure_operation("retrieve_nodes_data"):
                nodes_data = await graph_instance.nodes_data()
                logger.info(f"Retrieved {len(nodes_data)} nodes from graph")
            
            # Prepare entity data for VDB
            with monitor.measure_operation("prepare_entity_data"):
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
            
            # Create index configuration using proper schema
            config = create_faiss_index_config(
                persist_path=vdb_storage_path,
                embed_model=embedding_provider,
                name=vdb_id
            )
            
            # Create enhanced index with batch processing
            with monitor.measure_operation("create_enhanced_index"):
                # Enable batch processing in config
                if hasattr(graphrag_context, 'main_config') and hasattr(graphrag_context.main_config, 'index'):
                    config_dict = config.model_dump()
                    config_dict['enable_batch_embeddings'] = True
                    config = FAISSIndexConfig(**config_dict)
                
                entity_vdb = EnhancedFaissIndex(config)
            
            # Build the index with batch processing
            try:
                with monitor.measure_operation("build_index_with_embeddings"):
                    await entity_vdb.build_index(
                        elements=entities_data,
                        meta_data=["id", "content", "name"],
                        force=params.force_rebuild
                    )
            except Exception as e:
                raise EmbeddingError(
                    message=f"Failed to build entity VDB embeddings: {str(e)}",
                    context={
                        "vdb_id": vdb_id,
                        "num_entities": len(entities_data),
                        "error": str(e)
                    },
                    recovery_strategies=[
                        {"strategy": "retry", "params": {"max_attempts": 3}},
                        {"strategy": "fallback", "params": {"method": "sequential"}}
                    ],
                    cause=e
                )
            
            # Register the VDB in context
            graphrag_context.add_vdb_instance(vdb_id, entity_vdb)
            
            # Verify registration with detailed logging
            available_vdbs = list(graphrag_context._vdbs.keys()) if hasattr(graphrag_context, '_vdbs') else []
            logger.info(
                f"Enhanced Entity.VDB.Build: Successfully built AND REGISTERED VDB with ID: '{vdb_id}'. "
                f"Available VDBs in context now: {available_vdbs}"
            )
            
            # Log performance metrics
            metrics = monitor.get_summary()
            logger.info(
                f"Successfully built entity VDB '{vdb_id}' with "
                f"{len(entities_data)} entities indexed. "
                f"Performance: {metrics}"
            )
            
            return EntityVDBBuildOutputs(
                vdb_reference_id=vdb_id,
                num_entities_indexed=len(entities_data),
                status=f"Successfully built VDB with {len(entities_data)} entities"
            )
            
    except StructuredError:
        raise  # Re-raise structured errors as-is
    except Exception as e:
        # Wrap unexpected errors
        raise StructuredError(
            message=f"Unexpected error building entity VDB: {str(e)}",
            category=ErrorCategory.SYSTEM_ERROR,
            severity=ErrorSeverity.ERROR,
            context={
                "graph_id": params.graph_reference_id,
                "vdb_id": params.vdb_collection_name,
                "error": str(e)
            },
            cause=e
        )