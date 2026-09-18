"""Materialize a selected subgraph back into entities and source chunks.

This is the bridge between structural graph reasoning (PCST, path filtering,
Steiner trees) and evidence-grounded answer generation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from Core.Common.Constants import GRAPH_FIELD_SEP
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

    Only graph-resolved entities and source chunks with real textual content are
    emitted. Unknown runtime objects are skipped rather than stringified into
    fake evidence.
    """
    subgraph = inputs["subgraph"].data
    subgraph_nodes = set(subgraph.nodes or set()) if subgraph is not None else set()
    subgraph_edges = list(subgraph.edges or []) if subgraph is not None else []

    entity_records = []
    chunk_ids = []
    seen_chunk_ids = set()

    def add_source_ids(metadata):
        # Explicit lists preserve opaque IDs even when they contain <SEP>.
        if "passage_ids_json" in metadata:
            source_ids = json.loads(metadata["passage_ids_json"])
            if not isinstance(source_ids, list) or not all(
                isinstance(item, str) and item for item in source_ids
            ):
                raise ValueError("invalid graph passage identity list")
        else:
            source_ids = str(metadata.get("source_id", "")).split(GRAPH_FIELD_SEP)
        for chunk_id in source_ids:
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                chunk_ids.append(chunk_id)

    for node_id in sorted(subgraph_nodes):
        node_data = await ctx.graph.get_node(node_id)
        if not node_data:
            continue
        add_source_ids(node_data)
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
            add_source_ids(edge_data)

    top_k_chunks = (params or {}).get("top_k_chunks")
    if top_k_chunks is not None:
        chunk_ids = chunk_ids[: max(0, int(top_k_chunks))]

    chunk_records = []
    for chunk_id in chunk_ids:
        data = await ctx.doc_chunks.get_data_by_key(chunk_id)
        if data is None:
            continue

        text = None
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            text = data.get("content", data.get("text"))
        elif hasattr(data, "content"):
            text = getattr(data, "content")
        elif hasattr(data, "text"):
            text = getattr(data, "text")

        # Test emptiness without rewriting the producer's source text. Unknown
        # payloads are not strings and must never become fabricated evidence.
        if not isinstance(text, str) or not text.strip():
            continue

        provenance = {}
        if isinstance(data, dict):
            provenance = {
                key: data[key]
                for key in ("source_ref", "namespace_id", "source_registry_id",
                            "source_urls", "supporting_provenance_refs", "assertion_ids")
                if key in data
            }
        chunk_records.append(
            ChunkRecord(
                chunk_id=str(chunk_id),
                text=text,
                extra={**provenance, "selected_by_subgraph": True},
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
