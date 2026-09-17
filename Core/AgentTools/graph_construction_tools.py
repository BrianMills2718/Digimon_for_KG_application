"""Agent tools for building DIGIMON graph variants truthfully.

All maintained graph builders share the same concrete lifecycle:
chunks -> source-manifest decision -> build/load -> usability check -> manifest
write -> invalidate known derived artifacts when the graph was rebuilt.
"""

import asyncio
from pathlib import Path
from typing import Any, Optional, Type

from Core.AgentSchema.graph_construction_tool_contracts import (
    BaseGraphBuildOutputs,
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


def _effective_force_for_manifest(graph, chunks, requested_force: bool) -> tuple[bool, str | None]:
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


async def _build_graph_variant(
    tool_input,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any,
    *,
    graph_type: str,
    graph_id_suffix: str,
    display_name: str,
    output_class: Type[BaseGraphBuildOutputs],
    invalidate_sparse_matrices: bool = False,
):
    dataset = tool_input.target_dataset_name
    graph_id = f"{dataset}_{graph_id_suffix}"

    try:
        graph_config = main_config.graph.model_copy(deep=True)
        apply_overrides(graph_config, tool_input.config_overrides)
        config = main_config.model_copy(deep=True)
        config.graph = graph_config
        config.graph.type = graph_type

        graph = get_graph(config=config, llm=llm_instance, encoder=encoder_instance)
        if hasattr(graph._graph, "namespace"):
            graph._graph.namespace = chunk_factory.get_namespace(
                dataset,
                graph_type=graph_type,
            )

        chunks = await chunk_factory.get_chunks_for_dataset(dataset)
        if not chunks:
            return output_class(
                graph_id="",
                status="failure",
                message=f"No input chunks found for dataset: {dataset}",
            )

        effective_force, rebuild_reason = _effective_force_for_manifest(
            graph,
            chunks,
            bool(tool_input.force_rebuild),
        )
        success = await graph.build_graph(chunks=chunks, force=effective_force)
        if not success:
            return output_class(
                graph_id=graph_id,
                status="failure",
                message=f"{display_name} building failed internally for {dataset}.",
            )

        counts = await get_graph_counts(graph)
        if not _graph_counts_are_usable(counts):
            return output_class(
                graph_id=graph_id,
                status="failure",
                message=f"{display_name} for {dataset} contains no usable nodes.",
                **counts,
            )

        if not write_manifest(graph, chunks):
            logger.warning(
                f"{display_name} for '{dataset}' is usable but its source-chunk "
                "manifest could not be persisted; the next load will conservatively rebuild it."
            )

        if effective_force:
            invalidate_after_forced_graph_rebuild(
                main_config,
                dataset,
                invalidate_sparse_matrices=invalidate_sparse_matrices,
            )

        return output_class(
            graph_id=graph_id,
            status="success",
            message=(
                f"{display_name} built successfully for {dataset}."
                + _manifest_message(rebuild_reason)
            ),
            artifact_path=get_artifact_path(graph),
            graph_instance=graph,
            **counts,
        )
    except Exception as exc:
        logger.exception(f"{display_name} build failed for {dataset}: {exc}")
        return output_class(
            graph_id=graph_id,
            status="failure",
            message=str(exc),
        )


async def build_er_graph(
    tool_input: BuildERGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any,
) -> BuildERGraphOutputs:
    return await _build_graph_variant(
        tool_input,
        main_config,
        llm_instance,
        encoder_instance,
        chunk_factory,
        graph_type="er_graph",
        graph_id_suffix="ERGraph",
        display_name="ERGraph",
        output_class=BuildERGraphOutputs,
        invalidate_sparse_matrices=True,
    )


async def build_rk_graph(
    tool_input: BuildRKGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any,
) -> BuildRKGraphOutputs:
    return await _build_graph_variant(
        tool_input,
        main_config,
        llm_instance,
        encoder_instance,
        chunk_factory,
        graph_type="rkg_graph",
        graph_id_suffix="RKGraph",
        display_name="RKGraph",
        output_class=BuildRKGraphOutputs,
    )


async def build_tree_graph(
    tool_input: BuildTreeGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any,
) -> BuildTreeGraphOutputs:
    return await _build_graph_variant(
        tool_input,
        main_config,
        llm_instance,
        encoder_instance,
        chunk_factory,
        graph_type="tree_graph",
        graph_id_suffix="TreeGraph",
        display_name="TreeGraph",
        output_class=BuildTreeGraphOutputs,
    )


async def build_tree_graph_balanced(
    tool_input: BuildTreeGraphBalancedInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any,
) -> BuildTreeGraphBalancedOutputs:
    return await _build_graph_variant(
        tool_input,
        main_config,
        llm_instance,
        encoder_instance,
        chunk_factory,
        graph_type="tree_graph_balanced",
        graph_id_suffix="TreeGraphBalanced",
        display_name="TreeGraphBalanced",
        output_class=BuildTreeGraphBalancedOutputs,
    )


async def build_passage_graph(
    tool_input: BuildPassageGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any,
) -> BuildPassageGraphOutputs:
    return await _build_graph_variant(
        tool_input,
        main_config,
        llm_instance,
        encoder_instance,
        chunk_factory,
        graph_type="passage_graph",
        graph_id_suffix="PassageGraph",
        display_name="PassageGraph",
        output_class=BuildPassageGraphOutputs,
    )
