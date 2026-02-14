"""Meta: LLM reranking operator.

Use LLM to re-score entities or chunks by relevance to the query.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue


async def meta_rerank(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": SlotValue(QUERY_TEXT), "items": SlotValue(ENTITY_SET or CHUNK_SET)}
    Outputs: {"items": SlotValue(same kind as input, re-scored)}
    Params:  {"top_k": int}
    """
    query = inputs["query"].data
    items_slot = inputs["items"]
    items = items_slot.data
    p = params or {}
    top_k = p.get("top_k", len(items))

    if not items:
        return {"items": SlotValue(kind=items_slot.kind, data=[], producer="meta.rerank")}

    try:
        # Format items for LLM
        if items_slot.kind == SlotKind.CHUNK_SET:
            item_strs = [f"[{i+1}] {c.text[:200]}" for i, c in enumerate(items)]
        else:  # ENTITY_SET
            item_strs = [f"[{i+1}] {e.entity_name}: {e.description[:100]}" for i, e in enumerate(items)]

        prompt = (
            f"Query: {query}\n\n"
            f"Items:\n" + "\n".join(item_strs) + "\n\n"
            f"Rank the top {top_k} most relevant items by their numbers, "
            "from most to least relevant. Return comma-separated numbers."
        )

        result = await ctx.llm.aask(msg=[{"role": "user", "content": prompt}])
        ranked_indices = [int(n) - 1 for n in re.findall(r"\d+", result)]
        ranked_indices = [i for i in ranked_indices if 0 <= i < len(items)]

        # Assign new scores based on rank position
        reranked = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            item = items[idx]
            item.score = 1.0 - (rank / top_k)
            reranked.append(item)

        return {"items": SlotValue(kind=items_slot.kind, data=reranked, producer="meta.rerank")}

    except Exception as e:
        logger.exception(f"meta_rerank failed: {e}")
        return {"items": SlotValue(kind=items_slot.kind, data=items[:top_k], producer="meta.rerank")}
