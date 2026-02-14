"""Entity TF-IDF ranking operator.

Rank entities by TF-IDF similarity of their descriptions to the query.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Index.TFIDFStore import TFIDFIndex
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def entity_tfidf(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": SlotValue(QUERY_TEXT), "entities": SlotValue(ENTITY_SET)}
    Outputs: {"entities": SlotValue(ENTITY_SET)}
    Params:  {"top_k": int}
    """
    query = inputs["query"].data
    seed = inputs.get("entities")
    p = params or {}
    top_k = p.get("top_k", ctx.config.top_k)

    try:
        # Build corpus from graph nodes (or from seed entity set)
        if seed and seed.data:
            candidates = seed.data
            names = [r.entity_name for r in candidates]
            descriptions = [r.description for r in candidates]
        else:
            graph_nodes = list(await ctx.graph.get_nodes())
            names = graph_nodes
            node_data = [await ctx.graph.get_node(n) for n in names]
            descriptions = [nd.get("description", "") if nd else "" for nd in node_data]

        index = TFIDFIndex()
        index._build_index_from_list(descriptions)
        idxs = index.query(query_str=query, top_k=top_k)

        records = []
        for idx in idxs:
            name = names[idx]
            desc = descriptions[idx]
            records.append(EntityRecord(
                entity_name=str(name),
                description=desc,
                score=float(idx),  # rank position
                extra={"tfidf_rank": idx},
            ))

        return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=records, producer="entity.tfidf")}

    except Exception as e:
        logger.exception(f"entity_tfidf failed: {e}")
        return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=[], producer="entity.tfidf")}
