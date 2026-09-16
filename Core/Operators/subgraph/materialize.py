"""Materialize a selected subgraph back into entities and source chunks.

This is the bridge between structural graph reasoning (PCST, path filtering,
Steiner trees) and evidence-grounded answer generation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Utils import split_string_by_multi_markers
from Core.Schema.SlotTypes import (
    ChunkRecord,
    EntityRecord,
    SlotKind,
    SlotValue,
)


async def subgraph_materialize(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"subgraph": SUBGRAPH}
    Outputs: {"entities": ENTITY_SET, "chunks": CHUNK_SET}
    Params:  {"top_k_chunks": int | None}
    """
    subgraph = inputs["subgraph"].data
    if subgraph is None:
        subgraph_nodes = set()
        subgraph_edges = []
    else:
        subgraph_nodes = set(subgraph.nodes or set())
        subgraph_edges = list(subgraph.edges or [])

    entity_records = []
    chunk_ids = []
    seen_chunk_ids = set()

    def add_source_ids(source_id):
        if not source_id:
            return
        for chunk_id in split_string_by_multi_markers(
            str(source_id), [GRAPH_FIELD_SEP]
        ):
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                chunk_ids.append(chunk_id)

    for node_id in sorted(subgraph_nodes):
        node_data = await ctx.graph.get_node(node_id)
        node_data = node_data or {}
        add_source_ids(node_data.get("source_id", ""))
        entity_records.append(
            EntityRecord(
                entity_name=str(node_id),
                source_id=node_data.get("source_id", ""),
                entity_type=node_data.get("entity_type", ""),
                description=node_data.get("description", ""),
                extra={"selected_by_subgraph": True},
            )
        )

    for edge in subgraph_edges:
        if not isinstance(edge, (tuple, list)) or len(edge) < 2:
            continue
        src, tgt = str(edge[0]), str(edge[1])
        edge_data = await ctx.graph.get_edge(src, tgt)
        if edge_data is None:
            edge_data = await ctx.graph.get_edge(tgt, src)
        if edge_data:
            add_source_ids(edge_data.get("source_id", ""))

    top_k_chunks = (params or {}).get("top_k_chunks")
    if top_k_chunks is not None:
        chunk_ids = chunk_ids[: max(0, int(top_k_chunks))]

    chunk_records = []
    for chunk_id in chunk_ids:
        data = await ctx.doc_chunks.get_data_by_key(chunk_id)
        if data is None:
            continue
        if isinstance(data, str):
            text = data
        elif hasattr(data, "content"):
            text = str(data.content)
        elif hasattr(data, "text"):
            text = str(data.text)
        elif isinstance(data, dict):
            text = str(data.get("content", data.get("text", "")))
        else:
            text = str(data)

        chunk_records.append(
            ChunkRecord(
                chunk_id=str(chunk_id),
                text=text,
                extra={"selected_by_subgraph": True},
            )
        )

    return {
        "entities": SlotValue(
            kind=SlotKind.ENTITY_SET,
            data=entity_records,
            producer="subgraph.materialize",
        ),
        "chunks": SlotValue(
            kind=SlotKind.CHUNK_SET,
            data=chunk_records,
            producer="subgraph.materialize",
        ),
    }


def ensure_subgraph_materialize_registered() -> None:
    """Register the bridge lazily without creating registry import cycles."""
    from Core.Operators.registry import REGISTRY
    from Core.Schema.OperatorDescriptor import CostTier, OperatorDescriptor, SlotSpec

    if REGISTRY.get("subgraph.materialize") is not None:
        return

    REGISTRY.register(
        OperatorDescriptor(
            operator_id="subgraph.materialize",
            display_name="Materialize Subgraph Evidence",
            category="subgraph",
            input_slots=[SlotSpec("subgraph", SlotKind.SUBGRAPH)],
            output_slots=[
                SlotSpec("entities", SlotKind.ENTITY_SET),
                SlotSpec("chunks", SlotKind.CHUNK_SET),
            ],
            cost_tier=CostTier.FREE,
            when_to_use=(
                "Convert a selected structural subgraph into graph entities and "
                "the original source chunks supporting its nodes/edges."
            ),
            implementation=subgraph_materialize,
        )
    )
