"""Agent tools for building DIGIMON graph variants truthfully."""

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
from Core.AgentTools.derived_resource_cleanup import invalidate_after_forced_graph_rebuild
from Core.AgentTools.graph_chunk_manifest import manifest_matches, write_manifest
from Core.Common.Logger import logger
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
    node_count = counts.get("node_count")
    if node_count is None:
        return True
    try:
        return int(node_count) > 0
    except (TypeError, ValueError):
        return False


def _invalidate_if_rebuilt(
    main_config: Config,
    dataset_name: str,
    *,
    rebuilt: bool,
    er_graph: bool,
) -> None:
    if not rebuilt:
        return
    invalidate_after_forced_graph_rebuild(
        main_config,
        dataset_name,
        invalidate_sparse_matrices=er_graph,
    )


def _effective_force_for_manifest(graph, chunks, requested_force: bool) -> tuple[bool, str | None]:
    """Decide whether ER/RK artifacts can be safely reused.

    A source-chunk manifest gives us an exact corpus/chunk identity check. Missing
    manifests trigger a one-time rebuild for old artifacts; changed manifests
    catch additions, removals, edits, and chunking-strategy changes.
    """
    if requested_force:
        return True, "requested"
    match = manifest_matches(graph, chunks)
    if match is True:
        return False, None
    return True, "manifest_missing" if match is None else "manifest_changed"


def _manifest_message(reason: str | None) -> str:
    if reason == "manifest_missing":
        return " Rebuilt graph because its source-chunk manifest was missing."
    if reason == "manifest_changed":
        return " Rebuilt graph because the source chunks changed."
    return ""


async def _finalize_chunk_scoped_graph(
    graph,
    chunks,
    main_config,
    dataset_name,
    *,
    effective_force: bool,
    er_graph: bool,
) -> tuple[dict, bool]:
    counts = await get_graph_counts(graph)
    if not _graph_counts_are_usable(counts):
        return counts, False

    if not write_manifest(graph, chunks):
        logger.warning(
            f"Graph for '{dataset_name}' is usable but its source-chunk manifest "
            "could not be persisted; the next load will conservatively rebuild it."
        )
    _invalidate_if_rebuilt(
        main_config,
        dataset_name,
        rebuilt=effective_force,
        er_graph=er_graph,
    )
    return counts, True


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

        effective_force, rebuild_reason = _effective_force_for_manifest(
            graph, chunks, tool_input.force_rebuild
        )
        success = await graph.build_graph(chunks=chunks, force=effective_force)
        if not success:
            return BuildERGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_ERGraph",
                status="failure",
                message=f"ERGraph building failed internally for {tool_input.target_dataset_name}.",
            )

        counts, usable = await _finalize_chunk_scoped_graph(
            graph,
            chunks,
            main_config,
            tool_input.target_dataset_name,
            effective_force=effective_force,
            er_graph=True,
        )
        if not usable:
            return BuildERGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_ERGraph",
                status="failure",
                message=f"ERGraph for {tool_input.target_dataset_name} contains no usable nodes.",
                **counts,
            )

        return BuildERGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_ERGraph",
            status="success",
            message=(
                f"ERGraph built successfully for {tool_input.target_dataset_name}."
                + _manifest_message(rebuild_reason)
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

        effective_force, rebuild_reason = _effective_force_for_manifest(
            graph, chunks, tool_input.force_rebuild
        )
        success = await graph.build_graph(chunks=chunks, force=effective_force)
        if not success:
            return BuildRKGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_RKGraph",
                status="failure",
                message=f"RKGraph building failed internally for {tool_input.target_dataset_name}.",
            )

        counts, usable = await _finalize_chunk_scoped_graph(
            graph,
            chunks,
            main_config,
            tool_input.target_dataset_name,
            effective_force=effective_force,
            er_graph=False,
        )
        if not usable:
            return BuildRKGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_RKGraph",
                status="failure",
                message=f"RKGraph for {tool_input.target_dataset_name} contains no usable nodes.",
                **counts,
            )

        return BuildRKGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_RKGraph",
            status="success",
            message=(
                f"RKGraph built successfully for {tool_input.target_dataset_name}."
                + _manifest_message(rebuild_reason)
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
                graph_id="", status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
            )
        success = await graph.build_graph(chunks=chunks, force=tool_input.force_rebuild)
        if not success:
            return BuildTreeGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_TreeGraph", status="failure",
                message=f"TreeGraph building failed internally for {tool_input.target_dataset_name}.",
            )
        counts = await get_graph_counts(graph)
        if not _graph_counts_are_usable(counts):
            return BuildTreeGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_TreeGraph", status="failure",
                message=f"TreeGraph for {tool_input.target_dataset_name} contains no usable nodes.", **counts,
            )
        _invalidate_if_rebuilt(
            main_config, tool_input.target_dataset_name,
            rebuilt=tool_input.force_rebuild, er_graph=False,
        )
        return BuildTreeGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraph", status="success",
            message=f"TreeGraph built successfully for {tool_input.target_dataset_name}.",
            artifact_path=get_artifact_path(graph), graph_instance=graph, **counts,
        )
    except Exception as exc:
        logger.exception(f"TreeGraph build failed for {tool_input.target_dataset_name}: {exc}")
        return BuildTreeGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraph", status="failure", message=str(exc)
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
                graph_id="", status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
            )
        success = await graph.build_graph(chunks=chunks, force=tool_input.force_rebuild)
        if not success:
            return BuildTreeGraphBalancedOutputs(
                graph_id=f"{tool_input.target_dataset_name}_TreeGraphBalanced", status="failure",
                message=f"TreeGraphBalanced building failed internally for {tool_input.target_dataset_name}.",
            )
        counts = await get_graph_counts(graph)
        if not _graph_counts_are_usable(counts):
            return BuildTreeGraphBalancedOutputs(
                graph_id=f"{tool_input.target_dataset_name}_TreeGraphBalanced", status="failure",
                message=f"TreeGraphBalanced for {tool_input.target_dataset_name} contains no usable nodes.", **counts,
            )
        _invalidate_if_rebuilt(
            main_config, tool_input.target_dataset_name,
            rebuilt=tool_input.force_rebuild, er_graph=False,
        )
        return BuildTreeGraphBalancedOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraphBalanced", status="success",
            message=f"TreeGraphBalanced built successfully for {tool_input.target_dataset_name}.",
            artifact_path=get_artifact_path(graph), graph_instance=graph, **counts,
        )
    except Exception as exc:
        logger.exception(
            f"TreeGraphBalanced build failed for {tool_input.target_dataset_name}: {exc}"
        )
        return BuildTreeGraphBalancedOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraphBalanced",
            status="failure", message=str(exc),
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
                graph_id="", status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
            )
        success = await graph.build_graph(chunks=chunks, force=tool_input.force_rebuild)
        if not success:
            return BuildPassageGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_PassageGraph", status="failure",
                message=f"PassageGraph building failed internally for {tool_input.target_dataset_name}.",
            )
        counts = await get_graph_counts(graph)
        if not _graph_counts_are_usable(counts):
            return BuildPassageGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_PassageGraph", status="failure",
                message=f"PassageGraph for {tool_input.target_dataset_name} contains no usable nodes.", **counts,
            )
        _invalidate_if_rebuilt(
            main_config, tool_input.target_dataset_name,
            rebuilt=tool_input.force_rebuild, er_graph=False,
        )
        return BuildPassageGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_PassageGraph", status="success",
            message=f"PassageGraph built successfully for {tool_input.target_dataset_name}.",
            artifact_path=get_artifact_path(graph), graph_instance=graph, **counts,
        )
    except Exception as exc:
        logger.exception(f"PassageGraph build failed for {tool_input.target_dataset_name}: {exc}")
        return BuildPassageGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_PassageGraph", status="failure", message=str(exc)
        )
