"""Known adapters for slot type conversions.

Adapters are lightweight operators that transform one SlotValue into another.
They are also registered in the operator registry.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import (
    ChunkRecord,
    SlotKind,
    SlotValue,
)


async def adapter_attach_clusters(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """Attach community cluster info to entities that lack it."""
    entities = inputs["entities"].data
    if not entities or not ctx.community:
        return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=entities or [], producer="adapter.attach_clusters")}

    for ent in entities:
        if ent.clusters:
            continue
        if hasattr(ctx.community, "community_node_map") and ctx.community.community_node_map:
            try:
                cluster_info = await ctx.community.community_node_map.get_by_id(ent.entity_name)
                if cluster_info:
                    ent.clusters = cluster_info
            except Exception:
                pass

    return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=entities, producer="adapter.attach_clusters")}


async def adapter_entities_to_names(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """Extract just entity names as a list of strings (stored in extra)."""
    entities = inputs["entities"].data
    names = [e.entity_name for e in entities]
    return {"names": SlotValue(
        kind=SlotKind.ENTITY_SET,
        data=names,
        producer="adapter.entities_to_names",
        metadata={"type": "name_list"},
    )}


async def adapter_subgraph_to_chunks(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """Get text chunks for nodes in a subgraph."""
    sg = inputs["subgraph"].data
    if not sg or not sg.nodes:
        return {"chunks": SlotValue(kind=SlotKind.CHUNK_SET, data=[], producer="adapter.subgraph_to_chunks")}

    records = []
    for node_name in sg.nodes:
        nd = await ctx.graph.get_node(node_name)
        if nd and "source_id" in nd:
            from Core.Common.Constants import GRAPH_FIELD_SEP
            from Core.Common.Utils import split_string_by_multi_markers
            chunk_ids = split_string_by_multi_markers(nd["source_id"], [GRAPH_FIELD_SEP])
            for cid in chunk_ids:
                if cid:
                    data = await ctx.doc_chunks.get_data_by_key(cid)
                    if data:
                        records.append(ChunkRecord(chunk_id=cid, text=data))

    return {"chunks": SlotValue(kind=SlotKind.CHUNK_SET, data=records, producer="adapter.subgraph_to_chunks")}


# Adapter registry for quick lookup
ADAPTER_REGISTRY = {
    "adapter.attach_clusters": adapter_attach_clusters,
    "adapter.entities_to_names": adapter_entities_to_names,
    "adapter.subgraph_to_chunks": adapter_subgraph_to_chunks,
}
