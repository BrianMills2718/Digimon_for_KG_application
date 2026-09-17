"""Chunk entity occurrence operator.

Find exact source chunks where retrieved entities co-occur, ranked by relation
density. Chunk stores may expose raw strings, dictionaries, or TextChunk-like
objects; normalize those values before token truncation and answer generation.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Utils import split_string_by_multi_markers, truncate_list_by_token_size
from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


def _chunk_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content", value.get("text", "")) or "")
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    if hasattr(value, "text"):
        return str(getattr(value, "text") or "")
    return ""


async def chunk_occurrence(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": ENTITY_SET}
    Outputs: {"chunks": CHUNK_SET}

    Only exact chunk IDs referenced by graph provenance are returned. Unknown
    store objects are skipped instead of being stringified into synthetic
    evidence.
    """
    entities = inputs["entities"].data
    if not entities:
        return {
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=[],
                producer="chunk.occurrence",
            )
        }

    text_units = [
        split_string_by_multi_markers(entity.source_id, [GRAPH_FIELD_SEP])
        for entity in entities
    ]
    edges = await asyncio.gather(
        *[ctx.graph.get_node_edges(entity.entity_name) for entity in entities]
    )

    all_one_hop_nodes = set()
    for this_edges in edges:
        if this_edges:
            all_one_hop_nodes.update(edge[1] for edge in this_edges)
    all_one_hop_nodes = list(all_one_hop_nodes)

    all_one_hop_data = await asyncio.gather(
        *[ctx.graph.get_node(node_id) for node_id in all_one_hop_nodes]
    )
    one_hop_text_lookup = {
        node_id: set(
            split_string_by_multi_markers(
                node_data.get("source_id", ""),
                [GRAPH_FIELD_SEP],
            )
        )
        for node_id, node_data in zip(all_one_hop_nodes, all_one_hop_data)
        if node_data is not None
    }

    all_text_units_lookup = {}
    for order, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for chunk_id in this_text_units:
            if not chunk_id or chunk_id in all_text_units_lookup:
                continue

            relation_counts = 0
            for edge in this_edges or []:
                if (
                    edge[1] in one_hop_text_lookup
                    and chunk_id in one_hop_text_lookup[edge[1]]
                ):
                    relation_counts += 1

            raw_data = await ctx.doc_chunks.get_data_by_key(chunk_id)
            text = _chunk_text(raw_data).strip()
            if not text:
                continue

            all_text_units_lookup[chunk_id] = {
                "text": text,
                "order": order,
                "relation_counts": relation_counts,
            }

    items = [
        {"id": chunk_id, **value}
        for chunk_id, value in all_text_units_lookup.items()
    ]
    items.sort(key=lambda item: (item["order"], -item["relation_counts"]))

    if ctx.config and hasattr(ctx.config, "local_max_token_for_text_unit"):
        items = truncate_list_by_token_size(
            items,
            key=lambda item: item["text"],
            max_token_size=ctx.config.local_max_token_for_text_unit,
        )

    records = [
        ChunkRecord(
            chunk_id=item["id"],
            text=item["text"],
            extra={
                "order": item["order"],
                "relation_counts": item["relation_counts"],
            },
        )
        for item in items
    ]

    return {
        "chunks": SlotValue(
            kind=SlotKind.CHUNK_SET,
            data=records,
            producer="chunk.occurrence",
        )
    }
