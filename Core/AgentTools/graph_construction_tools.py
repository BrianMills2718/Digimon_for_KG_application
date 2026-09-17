"""
Agent tool functions for building DIGIMON graph variants.

Each builder applies graph-specific overrides, loads dataset chunks, delegates to
the graph implementation, and returns a truthful build result containing the
populated graph instance on success.
"""
import asyncio
from pathlib import Path
from typing import Any, Optional

from Core.AgentSchema.graph_construction_tool_contracts import (
    BuildERGraphInputs,
    BuildERGraphOutputs,
    BuildPassageGraphInputs,
    BuildPassageGraphOutputs,
    BuildRKGraphInputs,
    BuildRKGraphOutputs,
    BuildTreeGraphBalancedInputs,
    BuildTreeGraphBalancedOutputs,
    BuildTreeGraphInputs,
    BuildTreeGraphOutputs,
)
from Core.AgentTools.derived_resource_cleanup import (
    invalidate_after_forced_graph_rebuild,
)
from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Common.Utils import split_string_by_multi_markers
from Core.Graph.GraphFactory import get_graph
from Option.Config2 import Config


def apply_overrides(config_copy, overrides: Optional[Any]):
    if not overrides:
        return

    try:
        override_dict = (
            overrides.model_dump(exclude_unset=True)
            if hasattr(overrides, "model_dump")
            else overrides.dict(exclude_unset=True)
        )
        for field_name, value in override_dict.items():
            if hasattr(config_copy, field_name):
                setattr(config_copy, field_name, value)
            else:
                logger.warning(
                    f"apply_overrides: Field '{field_name}' not found in target config; skipping."
                )
    except Exception as exc:
        logger.error(f"apply_overrides: Error applying overrides: {exc}")


def get_artifact_path(graph_instance):
    storage = graph_instance._graph
    if hasattr(storage, "namespace") and getattr(storage.namespace, "path", None):
        return str(storage.namespace.path)
    if hasattr(storage, "file_path"):
        return str(Path(storage.file_path).parent)
    if hasattr(storage, "tree_pkl_file"):
        return str(Path(storage.tree_pkl_file).parent)
    return None


async def get_graph_counts(graph_instance) -> dict:
    node_count = None
    edge_count = None
    layer_count = None

    if hasattr(graph_instance, "node_num"):
        node_count = graph_instance.node_num
        if callable(node_count):
            node_count = node_count()
            if asyncio.iscoroutine(node_count):
                node_count = await node_count
    elif hasattr(graph_instance._graph, "get_node_num"):
        node_count = graph_instance._graph.get_node_num()

    if hasattr(graph_instance, "edge_num"):
        edge_count = graph_instance.edge_num
        if callable(edge_count):
            edge_count = edge_count()
            if asyncio.iscoroutine(edge_count):
                edge_count = await edge_count
    elif hasattr(graph_instance._graph, "get_edge_num"):
        edge_count = graph_instance._graph.get_edge_num()

    if hasattr(graph_instance, "num_layers"):
        layer_count = graph_instance.num_layers
        if callable(layer_count):
            layer_count = layer_count()
            if asyncio.iscoroutine(layer_count):
                layer_count = await layer_count
    elif hasattr(graph_instance._graph, "get_layer_num"):
        layer_count = graph_instance._graph.get_layer_num()

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "layer_count": layer_count,
    }


def _graph_counts_are_usable(counts: dict) -> bool:
    """A graph claiming success must contain nodes when node count is known."""
    node_count = counts.get("node_count")
    if node_count is None:
        return True
    try:
        return int(node_count) > 0
    except (TypeError, ValueError):
        return False


def _invalidate_if_forced(
    main_config: Config,
    dataset_name: str,
    *,
    force_rebuild: bool,
    er_graph: bool,
) -> None:
    """Invalidate known derived artifacts only after a successful usable build."""
    if not force_rebuild:
        return
    invalidate_after_forced_graph_rebuild(
        main_config,
        dataset_name,
        invalidate_sparse_matrices=er_graph,
    )


def _chunk_ids(chunks) -> set[str]:
    return {
        str(chunk_id)
        for chunk_id, _chunk in chunks or []
        if chunk_id is not None and str(chunk_id)
    }


async def _graph_node_source_ids(graph) -> set[str]:
    """Collect exact chunk IDs referenced by ER/RK graph nodes."""
    try:
        nodes = await graph.nodes_data()
    except Exception as exc:
        logger.warning(f"Could not inspect graph node provenance: {exc}")
        return set()

    source_ids = set()
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        for chunk_id in split_string_by_multi_markers(
            str(node.get("source_id", "")),
            [GRAPH_FIELD_SEP],
        ):
            if chunk_id:
                source_ids.add(chunk_id)
    return source_ids


async def _ensure_graph_chunk_provenance(
    graph,
    chunks,
    *,
    force_requested: bool,
) -> tuple[bool, bool]:
    """Ensure a loaded ER/RK graph refers to the current corpus chunk IDs.

    Returns ``(success, rebuilt_for_migration)``. Fresh/forced builds already use
    the supplied chunks and need no migration check. A non-forced persisted graph
    is rebuilt once when its node provenance refers to old chunk identities (for
    example after changing the canonical chunking strategy or corpus contents).
    """
    if force_requested:
        return True, False

    current_chunk_ids = _chunk_ids(chunks)
    referenced_ids = await _graph_node_source_ids(graph)
    if referenced_ids and referenced_ids.issubset(current_chunk_ids):
        return True, False

    missing = sorted(referenced_ids - current_chunk_ids)
    logger.warning(
        "Loaded graph provenance does not match current corpus chunks; "
        f"rebuilding graph. missing_source_ids={missing[:10]}, "
        f"referenced={len(referenced_ids)}, current_chunks={len(current_chunk_ids)}"
    )
    rebuilt = await graph.build_graph(chunks=chunks, force=True)
    return bool(rebuilt), True


async def build_er_graph(
    tool_input: BuildERGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any,
) -> BuildERGraphOutputs:
    try:
        graph_config = main_config.graph.model_copy(deep=True)
        apply_overrides(graph_config, tool_input.config_overrides)
        config = main_config.model_copy(deep=True)
        config.graph = graph_config
        config.graph.type = "er_graph"

        graph = get_graph(config=config, llm=llm_instance, encoder=encoder_instance)
        if hasattr(graph._graph, "namespace"):
            graph._graph.namespace = chunk_factory.get_namespace(
                tool_input.target_dataset_name, graph_type="er_graph"
            )

        chunks = await chunk_factory.get_chunks_for_dataset(tool_input.target_dataset_name)
        if not chunks:
            return BuildERGraphOutputs(
                graph_id="",
                status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
            )

        success = await graph.build_graph(chunks=chunks, force=tool_input.force_rebuild)
        if not success:
            return BuildERGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_ERGraph",
                status="failure",
                message=f"ERGraph building failed internally for {tool_input.target_dataset_name}.",
            )

        provenance_ok, migrated = await _ensure_graph_chunk_provenance(
            graph,
            chunks,
            force_requested=tool_input.force_rebuild,
        )
        if not provenance_ok:
            return BuildERGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_ERGraph",
                status="failure",
                message=(
                    f"ERGraph for {tool_input.target_dataset_name} could not be rebuilt "
                    "after stale chunk provenance was detected."
                ),
            )

        counts = await get_graph_counts(graph)
        if not _graph_counts_are_usable(counts):
            return BuildERGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_ERGraph",
                status="failure",
                message=f"ERGraph for {tool_input.target_dataset_name} contains no usable nodes.",
                **counts,
            )

        _invalidate_if_forced(
            main_config,
            tool_input.target_dataset_name,
            force_rebuild=bool(tool_input.force_rebuild or migrated),
            er_graph=True,
        )
        return BuildERGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_ERGraph",
            status="success",
            message=(
                f"ERGraph built successfully for {tool_input.target_dataset_name}."
                + (" Rebuilt stale chunk provenance." if migrated else "")
            ),
            artifact_path=get_artifact_path(graph),
            graph_instance=graph,
            **counts,
        )
    except Exception as exc:
        logger.exception(f"ERGraph build failed for {tool_input.target_dataset_name}: {exc}")
        return BuildERGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_ERGraph",
            status="failure",
            message=str(exc),
        )


async def build_rk_graph(
    tool_input: BuildRKGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any,
) -> BuildRKGraphOutputs:
    try:
        graph_config = main_config.graph.model_copy(deep=True)
        apply_overrides(graph_config, tool_input.config_overrides)
        config = main_config.model_copy(deep=True)
        config.graph = graph_config
        config.graph.type = "rkg_graph"

        graph = get_graph(config=config, llm=llm_instance, encoder=encoder_instance)
        if hasattr(graph._graph, "namespace"):
            graph._graph.namespace = chunk_factory.get_namespace(
                tool_input.target_dataset_name, graph_type="rkg_graph"
            )

        chunks = await chunk_factory.get_chunks_for_dataset(tool_input.target_dataset_name)
        if not chunks:
            return BuildRKGraphOutputs(
                graph_id="",
                status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
            )

        success = await graph.build_graph(chunks=chunks, force=tool_input.force_rebuild)
        if not success:
            return BuildRKGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_RKGraph",
                status="failure",
                message=f"RKGraph building failed internally for {tool_input.target_dataset_name}.",
            )

        provenance_ok, migrated = await _ensure_graph_chunk_provenance(
            graph,
            chunks,
            force_requested=tool_input.force_rebuild,
        )
        if not provenance_ok:
            return BuildRKGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_RKGraph",
                status="failure",
                message=(
                    f"RKGraph for {tool_input.target_dataset_name} could not be rebuilt "
                    "after stale chunk provenance was detected."
                ),
            )

        counts = await get_graph_counts(graph)
        if not _graph_counts_are_usable(counts):
            return BuildRKGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_RKGraph",
                status="failure",
                message=f"RKGraph for {tool_input.target_dataset_name} contains no usable nodes.",
                **counts,
            )

        _invalidate_if_forced(
            main_config,
            tool_input.target_dataset_name,
            force_rebuild=bool(tool_input.force_rebuild or migrated),
            er_graph=False,
        )
        return BuildRKGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_RKGraph",
            status="success",
            message=(
                f"RKGraph built successfully for {tool_input.target_dataset_name}."
                + (" Rebuilt stale chunk provenance." if migrated else "")
            ),
            artifact_path=get_artifact_path(graph),
            graph_instance=graph,
            **counts,
        )
    except Exception as exc:
        logger.exception(f"RKGraph build failed for {tool_input.target_dataset_name}: {exc}")
        return BuildRKGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_RKGraph",
            status="failure",
            message=str(exc),
        )


async def build_tree_graph(
    tool_input: BuildTreeGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any,
) -> BuildTreeGraphOutputs:
    try:
        graph_config = main_config.graph.model_copy(deep=True)
        apply_overrides(graph_config, tool_input.config_overrides)
        config = main_config.model_copy(deep=True)
        config.graph = graph_config
        config.graph.type = "tree_graph"

        graph = get_graph(config=config, llm=llm_instance, encoder=encoder_instance)
        if hasattr(graph._graph, "namespace"):
            graph._graph.namespace = chunk_factory.get_namespace(
                tool_input.target_dataset_name, graph_type="tree_graph"
            )

        chunks = await chunk_factory.get_chunks_for_dataset(tool_input.target_dataset_name)
        if not chunks:
            return BuildTreeGraphOutputs(
                graph_id="",
                status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
            )

        success = await graph.build_graph(chunks=chunks, force=tool_input.force_rebuild)
        if not success:
            return BuildTreeGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_TreeGraph",
                status="failure",
                message=f"TreeGraph building failed internally for {tool_input.target_dataset_name}.",
            )

        counts = await get_graph_counts(graph)
        if not _graph_counts_are_usable(counts):
            return BuildTreeGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_TreeGraph",
                status="failure",
                message=f"TreeGraph for {tool_input.target_dataset_name} contains no usable nodes.",
                **counts,
            )

        _invalidate_if_forced(
            main_config,
            tool_input.target_dataset_name,
            force_rebuild=tool_input.force_rebuild,
            er_graph=False,
        )
        return BuildTreeGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraph",
            status="success",
            message=f"TreeGraph built successfully for {tool_input.target_dataset_name}.",
            artifact_path=get_artifact_path(graph),
            graph_instance=graph,
            **counts,
        )
    except Exception as exc:
        logger.exception(f"TreeGraph build failed for {tool_input.target_dataset_name}: {exc}")
        return BuildTreeGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraph",
            status="failure",
            message=str(exc),
        )


async def build_tree_graph_balanced(
    tool_input: BuildTreeGraphBalancedInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any,
) -> BuildTreeGraphBalancedOutputs:
    try:
        graph_config = main_config.graph.model_copy(deep=True)
        apply_overrides(graph_config, tool_input.config_overrides)
        config = main_config.model_copy(deep=True)
        config.graph = graph_config
        config.graph.type = "tree_graph_balanced"

        graph = get_graph(config=config, llm=llm_instance, encoder=encoder_instance)
        if hasattr(graph._graph, "namespace"):
            graph._graph.namespace = chunk_factory.get_namespace(
                tool_input.target_dataset_name, graph_type="tree_graph_balanced"
            )

        chunks = await chunk_factory.get_chunks_for_dataset(tool_input.target_dataset_name)
        if not chunks:
            return BuildTreeGraphBalancedOutputs(
                graph_id="",
                status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
            )

        success = await graph.build_graph(chunks=chunks, force=tool_input.force_rebuild)
        if not success:
            return BuildTreeGraphBalancedOutputs(
                graph_id=f"{tool_input.target_dataset_name}_TreeGraphBalanced",
                status="failure",
                message=f"TreeGraphBalanced building failed internally for {tool_input.target_dataset_name}.",
            )

        counts = await get_graph_counts(graph)
        if not _graph_counts_are_usable(counts):
            return BuildTreeGraphBalancedOutputs(
                graph_id=f"{tool_input.target_dataset_name}_TreeGraphBalanced",
                status="failure",
                message=f"TreeGraphBalanced for {tool_input.target_dataset_name} contains no usable nodes.",
                **counts,
            )

        _invalidate_if_forced(
            main_config,
            tool_input.target_dataset_name,
            force_rebuild=tool_input.force_rebuild,
            er_graph=False,
        )
        return BuildTreeGraphBalancedOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraphBalanced",
            status="success",
            message=f"TreeGraphBalanced built successfully for {tool_input.target_dataset_name}.",
            artifact_path=get_artifact_path(graph),
            graph_instance=graph,
            **counts,
        )
    except Exception as exc:
        logger.exception(
            f"TreeGraphBalanced build failed for {tool_input.target_dataset_name}: {exc}"
        )
        return BuildTreeGraphBalancedOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraphBalanced",
            status="failure",
            message=str(exc),
        )


async def build_passage_graph(
    tool_input: BuildPassageGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any,
) -> BuildPassageGraphOutputs:
    try:
        graph_config = main_config.graph.model_copy(deep=True)
        apply_overrides(graph_config, tool_input.config_overrides)
        config = main_config.model_copy(deep=True)
        config.graph = graph_config
        config.graph.type = "passage_graph"

        graph = get_graph(config=config, llm=llm_instance, encoder=encoder_instance)
        if hasattr(graph._graph, "namespace"):
            graph._graph.namespace = chunk_factory.get_namespace(
                tool_input.target_dataset_name, graph_type="passage_graph"
            )

        chunks = await chunk_factory.get_chunks_for_dataset(tool_input.target_dataset_name)
        if not chunks:
            return BuildPassageGraphOutputs(
                graph_id="",
                status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
            )

        success = await graph.build_graph(chunks=chunks, force=tool_input.force_rebuild)
        if not success:
            return BuildPassageGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_PassageGraph",
                status="failure",
                message=f"PassageGraph building failed internally for {tool_input.target_dataset_name}.",
            )

        counts = await get_graph_counts(graph)
        if not _graph_counts_are_usable(counts):
            return BuildPassageGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_PassageGraph",
                status="failure",
                message=f"PassageGraph for {tool_input.target_dataset_name} contains no usable nodes.",
                **counts,
            )

        _invalidate_if_forced(
            main_config,
            tool_input.target_dataset_name,
            force_rebuild=tool_input.force_rebuild,
            er_graph=False,
        )
        return BuildPassageGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_PassageGraph",
            status="success",
            message=f"PassageGraph built successfully for {tool_input.target_dataset_name}.",
            artifact_path=get_artifact_path(graph),
            graph_instance=graph,
            **counts,
        )
    except Exception as exc:
        logger.exception(f"PassageGraph build failed for {tool_input.target_dataset_name}: {exc}")
        return BuildPassageGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_PassageGraph",
            status="failure",
            message=str(exc),
        )
