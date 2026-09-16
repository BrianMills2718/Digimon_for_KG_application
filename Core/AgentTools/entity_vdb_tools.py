# Core/AgentTools/entity_vdb_tools.py
"""Entity VDB build tool."""

import uuid

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import EntityVDBBuildInputs, EntityVDBBuildOutputs
from Core.AgentTools.index_config_helper import create_faiss_index_config
from Core.Common.Logger import logger
from Core.Index.FaissIndex import FaissIndex


async def entity_vdb_build_tool(
    params: EntityVDBBuildInputs,
    graphrag_context: GraphRAGContext,
) -> EntityVDBBuildOutputs:
    """Build or load a searchable entity VDB and register it in context."""
    logger.info(
        f"Building entity VDB: graph_id='{params.graph_reference_id}', "
        f"collection='{params.vdb_collection_name}'"
    )

    try:
        graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
        if not graph_instance:
            error_msg = f"Graph '{params.graph_reference_id}' not found in context"
            logger.error(error_msg)
            return EntityVDBBuildOutputs(
                vdb_reference_id="",
                num_entities_indexed=0,
                status=f"Error: {error_msg}",
            )

        embedding_provider = graphrag_context.embedding_provider
        if not embedding_provider:
            error_msg = "No embedding provider available in context"
            logger.error(error_msg)
            return EntityVDBBuildOutputs(
                vdb_reference_id="",
                num_entities_indexed=0,
                status=f"Error: {error_msg}",
            )

        vdb_id = params.vdb_collection_name
        existing_vdb = graphrag_context.get_vdb_instance(vdb_id)
        if existing_vdb and not params.force_rebuild:
            nodes_data = await graph_instance.nodes_data()
            logger.info(f"VDB '{vdb_id}' already registered; reusing it")
            return EntityVDBBuildOutputs(
                vdb_reference_id=vdb_id,
                num_entities_indexed=len(nodes_data),
                status="VDB already exists",
            )

        nodes_data = await graph_instance.nodes_data()
        logger.info(f"Retrieved {len(nodes_data)} nodes from graph")

        entities_data = []
        for node in nodes_data:
            if params.entity_types:
                node_type = node.get("type", node.get("entity_type", "entity"))
                if node_type not in params.entity_types:
                    continue

            entity_id = str(node.get("entity_name", node.get("id", uuid.uuid4().hex)))
            content = (
                node.get("description")
                or node.get("content")
                or str(node.get("entity_name", ""))
            )
            if not content:
                continue

            entity_doc = {
                "id": entity_id,
                "content": content,
                "name": node.get("entity_name", entity_id),
            }
            if params.include_metadata:
                for key, value in node.items():
                    if key not in {"id", "content", "name", "description"}:
                        entity_doc[key] = value
            entities_data.append(entity_doc)

        if not entities_data:
            logger.warning(f"No suitable entities found in graph '{params.graph_reference_id}'")
            return EntityVDBBuildOutputs(
                vdb_reference_id="",
                num_entities_indexed=0,
                status="No entities with content found in graph",
            )

        logger.info(f"Prepared {len(entities_data)} entities for indexing")
        config = create_faiss_index_config(
            persist_path=f"storage/vdb/{vdb_id}",
            embed_model=embedding_provider,
            name=vdb_id,
        )
        entity_vdb = FaissIndex(config)

        build_ok = await entity_vdb.build_index(
            elements=entities_data,
            meta_data=["id", "content", "name"],
            force=params.force_rebuild,
        )
        if not build_ok:
            error_msg = f"Entity VDB '{vdb_id}' failed to build or load a usable index"
            logger.error(error_msg)
            return EntityVDBBuildOutputs(
                vdb_reference_id="",
                num_entities_indexed=0,
                status=f"Error: {error_msg}",
            )

        graphrag_context.add_vdb_instance(vdb_id, entity_vdb)
        available_vdbs = graphrag_context.list_vdbs()
        if vdb_id not in available_vdbs:
            error_msg = f"Entity VDB '{vdb_id}' built but failed context registration"
            logger.error(error_msg)
            return EntityVDBBuildOutputs(
                vdb_reference_id="",
                num_entities_indexed=0,
                status=f"Error: {error_msg}",
            )

        logger.info(
            f"Entity.VDB.Build: built and registered '{vdb_id}'. "
            f"Available VDBs: {available_vdbs}"
        )
        return EntityVDBBuildOutputs(
            vdb_reference_id=vdb_id,
            num_entities_indexed=len(entities_data),
            status=f"Successfully built VDB with {len(entities_data)} entities",
        )

    except Exception as e:
        error_msg = f"Error building entity VDB: {e}"
        logger.exception(error_msg)
        return EntityVDBBuildOutputs(
            vdb_reference_id="",
            num_entities_indexed=0,
            status=f"Error: {e}",
        )
