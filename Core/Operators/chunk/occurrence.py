"""Chunk entity occurrence operator.

Find text chunks where entities co-occur, ranked by relation density.
Ported from ChunkRetriever._find_relevant_chunks_from_entity_occurrence.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Common.Utils import split_string_by_multi_markers, truncate_list_by_token_size
from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


async def chunk_occurrence(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": SlotValue(ENTITY_SET)}
    Outputs: {"chunks": SlotValue(CHUNK_SET)}
    """
    entities = inputs["entities"].data  # List[EntityRecord]
    if not entities:
        return {"chunks": SlotValue(kind=SlotKind.CHUNK_SET, data=[], producer="chunk.occurrence")}

    # Get source_ids and edges for each entity
    text_units = [
        split_string_by_multi_markers(e.source_id, [GRAPH_FIELD_SEP])
        for e in entities
    ]
    edges = await asyncio.gather(
        *[ctx.graph.get_node_edges(e.entity_name) for e in entities]
    )

    # Collect one-hop neighbors and their source_ids
    all_one_hop_nodes = set()
    for this_edges in edges:
        if this_edges:
            all_one_hop_nodes.update(e[1] for e in this_edges)
    all_one_hop_nodes = list(all_one_hop_nodes)

    all_one_hop_data = await asyncio.gather(
        *[ctx.graph.get_node(n) for n in all_one_hop_nodes]
    )
    one_hop_text_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_data)
        if v is not None
    }

    # Score chunks by relation co-occurrence
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if not c_id or c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in (this_edges or []):
                if e[1] in one_hop_text_lookup and c_id in one_hop_text_lookup[e[1]]:
                    relation_counts += 1
            data = await ctx.doc_chunks.get_data_by_key(c_id)
            all_text_units_lookup[c_id] = {
                "data": data,
                "order": index,
                "relation_counts": relation_counts,
            }

    items = [
        {"id": k, **v} for k, v in all_text_units_lookup.items()
        if v.get("data") is not None
    ]
    items.sort(key=lambda x: (x["order"], -x["relation_counts"]))

    if ctx.config and hasattr(ctx.config, "local_max_token_for_text_unit"):
        items = truncate_list_by_token_size(
            items,
            key=lambda x: x["data"],
            max_token_size=ctx.config.local_max_token_for_text_unit,
        )

    records = [
        ChunkRecord(
            chunk_id=it["id"],
            text=it["data"],
            extra={"order": it["order"], "relation_counts": it["relation_counts"]},
        )
        for it in items
    ]

    return {"chunks": SlotValue(kind=SlotKind.CHUNK_SET, data=records, producer="chunk.occurrence")}
