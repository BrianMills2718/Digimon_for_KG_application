"""Chunk from relationships operator.

Extract text chunks referenced by relationship source_ids.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Common.Utils import split_string_by_multi_markers, truncate_list_by_token_size
from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


async def chunk_from_relation(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"relationships": SlotValue(RELATIONSHIP_SET)}
    Outputs: {"chunks": SlotValue(CHUNK_SET)}
    """
    rels = inputs["relationships"].data  # List[RelationshipRecord]
    if not rels:
        return {"chunks": SlotValue(kind=SlotKind.CHUNK_SET, data=[], producer="chunk.from_relation")}

    all_text_units_lookup = {}
    for index, rel in enumerate(rels):
        source_id = rel.source_id
        if not source_id:
            # Try extra dict for raw edge data
            source_id = rel.extra.get("source_id", "")
        chunk_ids = split_string_by_multi_markers(source_id, [GRAPH_FIELD_SEP])
        for c_id in chunk_ids:
            if c_id and c_id not in all_text_units_lookup:
                data = await ctx.doc_chunks.get_data_by_key(c_id)
                all_text_units_lookup[c_id] = {"data": data, "order": index}

    # Filter None, sort by order, truncate
    items = [
        {"id": k, **v} for k, v in all_text_units_lookup.items()
        if v.get("data") is not None
    ]
    items.sort(key=lambda x: x["order"])

    if ctx.config and hasattr(ctx.config, "local_max_token_for_text_unit"):
        items = truncate_list_by_token_size(
            items,
            key=lambda x: x["data"],
            max_token_size=ctx.config.local_max_token_for_text_unit,
        )

    records = [
        ChunkRecord(chunk_id=it["id"], text=it["data"])
        for it in items
    ]

    return {"chunks": SlotValue(kind=SlotKind.CHUNK_SET, data=records, producer="chunk.from_relation")}
