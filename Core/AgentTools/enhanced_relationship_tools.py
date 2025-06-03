# Core/AgentTools/enhanced_relationship_tools.py
"""
Enhanced Relationship VDB Build Tool with batch embedding support
"""

from typing import List, Optional, Dict, Any
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    RelationshipVDBBuildInputs,
    RelationshipVDBBuildOutputs
)
from Core.Index.EnhancedFaissIndex import EnhancedFaissIndex
from Core.Common.Logger import logger
from Core.Common.StructuredErrors import (
    StructuredError, ErrorCategory, ErrorSeverity,
    EmbeddingError
)
from Core.Index.Schema import FAISSIndexConfig
from Core.Common.BatchEmbeddingProcessor import BatchEmbeddingProcessor
from Core.Common.PerformanceMonitor import PerformanceMonitor

# Import proper index configuration helper
from Core.AgentTools.index_config_helper import create_faiss_index_config

async def relationship_vdb_build_tool(
    params: RelationshipVDBBuildInputs,
    graphrag_context: GraphRAGContext
) -> RelationshipVDBBuildOutputs:
    """
    Enhanced relationship VDB build tool with batch embedding and performance monitoring.
    
    This tool creates a searchable index of relationships from a graph,
    allowing for similarity-based retrieval based on relationship properties.
    """
    logger.info(
        f"Enhanced Relationship VDB Build: graph_id='{params.graph_reference_id}', "
        f"collection='{params.vdb_collection_name}'"
    )
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    try:
        with monitor.measure_operation("relationship_vdb_build_total"):
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
            
            # Extract the actual NetworkX graph
            if hasattr(graph_instance, '_graph') and hasattr(graph_instance._graph, 'graph'):
                nx_graph = graph_instance._graph.graph
            elif hasattr(graph_instance, 'graph'):
                nx_graph = graph_instance.graph
            else:
                nx_graph = graph_instance
                
            logger.info(f"Retrieved graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
            
            # Check if VDB already exists
            vdb_id = f"{params.vdb_collection_name}_relationships"
            existing_vdb = graphrag_context.get_vdb_instance(vdb_id)
            
            if existing_vdb and not params.force_rebuild:
                logger.info(f"VDB '{vdb_id}' already exists and force_rebuild=False, skipping build")
                return RelationshipVDBBuildOutputs(
                    vdb_reference_id=vdb_id,
                    num_relationships_indexed=nx_graph.number_of_edges(),
                    status="VDB already exists"
                )
            
            # Get embedding provider
            embedding_provider = graphrag_context.embedding_provider
            if not embedding_provider:
                raise StructuredError(
                    message="No embedding provider available in context",
                    category=ErrorCategory.CONFIGURATION_ERROR,
                    severity=ErrorSeverity.CRITICAL
                )
            
            # Prepare relationship data with performance monitoring
            with monitor.measure_operation("prepare_relationship_data"):
                relationships_data = []
                edge_metadata = ["source", "target", "id"]
                
                if params.include_metadata:
                    # Collect all possible metadata keys from edges
                    metadata_keys = set()
                    for u, v, data in nx_graph.edges(data=True):
                        metadata_keys.update(data.keys())
                    edge_metadata.extend(list(metadata_keys - set(params.embedding_fields)))
                
                # Extract edges and their properties
                for u, v, edge_data in nx_graph.edges(data=True):
                    # Create text content from embedding fields
                    content_parts = []
                    for field in params.embedding_fields:
                        if field in edge_data:
                            content_parts.append(f"{field}: {edge_data[field]}")
                    
                    if not content_parts:
                        # If no embedding fields found, use a default description
                        content_parts.append(f"Relationship from {u} to {v}")
                    
                    content = " | ".join(content_parts)
                    
                    # Create relationship document
                    rel_doc = {
                        "id": edge_data.get("id", f"{u}->{v}"),
                        "content": content,
                        "source": u,
                        "target": v
                    }
                    
                    # Add metadata if requested
                    if params.include_metadata:
                        for key, value in edge_data.items():
                            if key not in ["id"] and key not in params.embedding_fields:
                                rel_doc[key] = value
                    
                    relationships_data.append(rel_doc)
            
            if not relationships_data:
                logger.warning(f"No relationships found in graph '{params.graph_reference_id}'")
                return RelationshipVDBBuildOutputs(
                    vdb_reference_id="",
                    num_relationships_indexed=0,
                    status="No relationships found in graph"
                )
            
            logger.info(f"Prepared {len(relationships_data)} relationships for indexing")
            
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
                
                relationship_vdb = EnhancedFaissIndex(config)
            
            # Build the index with batch processing
            try:
                with monitor.measure_operation("build_index_with_embeddings"):
                    await relationship_vdb.build_index(
                        elements=relationships_data,
                        meta_data=edge_metadata,
                        force=params.force_rebuild
                    )
            except Exception as e:
                raise EmbeddingError(
                    message=f"Failed to build relationship VDB embeddings: {str(e)}",
                    context={
                        "vdb_id": vdb_id,
                        "num_relationships": len(relationships_data),
                        "error": str(e)
                    },
                    recovery_strategies=[
                        {"strategy": "retry", "params": {"max_attempts": 3}},
                        {"strategy": "fallback", "params": {"method": "sequential"}}
                    ],
                    cause=e
                )
            
            # Register the VDB in context
            graphrag_context.add_vdb_instance(vdb_id, relationship_vdb)
            
            # Log performance metrics
            metrics = monitor.get_summary()
            logger.info(
                f"Successfully built relationship VDB '{vdb_id}' with "
                f"{len(relationships_data)} relationships indexed. "
                f"Performance: {metrics}"
            )
            
            return RelationshipVDBBuildOutputs(
                vdb_reference_id=vdb_id,
                num_relationships_indexed=len(relationships_data),
                status=f"Successfully built VDB with {len(relationships_data)} relationships"
            )
            
    except StructuredError:
        raise  # Re-raise structured errors as-is
    except Exception as e:
        # Wrap unexpected errors
        raise StructuredError(
            message=f"Unexpected error building relationship VDB: {str(e)}",
            category=ErrorCategory.SYSTEM_ERROR,
            severity=ErrorSeverity.ERROR,
            context={
                "graph_id": params.graph_reference_id,
                "vdb_id": params.vdb_collection_name,
                "error": str(e)
            },
            cause=e
        )