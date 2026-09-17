# Core/AgentTools/entity_vdb_tools.py
"""Entity VDB build tool."""

import uuid

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import EntityVDBBuildInputs, EntityVDBBuildOutputs
from Core.AgentTools.index_config_helper import create_faiss_index_config
from Core.Common.Logger import logger
from Core.Index.FaissIndex import FaissIndex


async def _registered_vdb_is_usable(vdb) -> bool:
    """Return True only when a registered VDB has or can load a live index."""
    if getattr(vdb, "_index", None) is not None:
        return True
    load = getattr(vdb, "load", None)
    if load is None:
        return False
    try:
        return bool(await load())
    except Exception as exc:
        logger.warning(f"Registered VDB reload failed: {exc}")
        return False


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
            if await _registered_vdb_is_usable(existing_vdb):
                nodes_data = await graph_instance.nodes_data()
                logger.info(f"VDB '{vdb_id}' already registered and usable; reusing it")
                return EntityVDBBuildOutputs(
                    vdb_reference_id=vdb_id,
                    num_entities_indexed=len(nodes_data),
                    status="VDB already exists",
                )
            logger.warning(
                f"Registered entity VDB '{vdb_id}' is unusable; rebuilding it"
            )

        nodes_data = await graph_instance.nodes_data()
        logger.info(f"Retrieved {len(nodes_data)} nodes from graph")

        entities_data = []
        for node in nodes_data:
            if params.entity_types:
                node_type = node.get("type", node.get("entity_type", "entity"))
                if node_type not in params.entity_types:
                    continue

            # Tree graphs expose a stable numeric ``index`` rather than an
            # entity_name. Prefer graph-native stable identifiers before UUIDs.
            raw_entity_id = (
                node.get("entity_name")
                if node.get("entity_name") is not None
                else node.get("id")
                if node.get("id") is not None
                else node.get("index")
            )
            entity_id = str(raw_entity_id if raw_entity_id is not None else uuid.uuid4().hex)

            # ``nodes_data()`` prepares a graph-native searchable content field.
            # For ER/RK graphs it includes entity name + type + description;
            # preferring description alone made exact/named-entity VDB queries
            # ignore the entity's own identity whenever a description existed.
            content = (
                node.get("content")
                or node.get("description")
                or str(node.get("entity_name", node.get("index", "")))
            )
            if not content:
                continue

            entity_doc = {
                "id": entity_id,
                "content": content,
                "name": str(node.get("entity_name", raw_entity_id if raw_entity_id is not None else entity_id)),
            }
            if params.include_metadata:
                for key, value in node.items():
                    # ``content`` is the embedded TextNode body. Preserve all
                    # other graph-native fields (source_id, entity_type,
                    # tree index/layer, etc.) as retrieval metadata.
                    if key != "content":
                        entity_doc.setdefault(key, value)
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

        # Metadata must follow the graph representation rather than a fixed ER
        # subset. In particular TreeGraph retrieval requires ``index``/``layer``
        # while ER/RK paths rely on source_id/entity_type provenance.
        metadata_keys = {"id", "name"}
        if params.include_metadata:
            for entity_doc in entities_data:
                metadata_keys.update(
                    key for key in entity_doc.keys() if key != "content"
                )

        build_ok = await entity_vdb.build_index(
            elements=entities_data,
            meta_data=sorted(metadata_keys),
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
