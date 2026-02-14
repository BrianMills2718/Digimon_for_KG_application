"""Meta: LLM reasoning step operator.

Use LLM to reason over retrieved chunks and refine the query or produce an intermediate answer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue


async def meta_reason_step(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": SlotValue(QUERY_TEXT), "chunks": SlotValue(CHUNK_SET)}
    Outputs: {"query": SlotValue(QUERY_TEXT)}  -- refined query or intermediate reasoning

    Params:  {"prompt_template": str, "mode": "refine"|"decompose"}
    """
    query = inputs["query"].data
    chunks = inputs.get("chunks")
    chunk_data = chunks.data if chunks else []
    p = params or {}
    mode = p.get("mode", "refine")

    chunk_text = "\n\n".join(c.text for c in chunk_data if c.text)

    try:
        if mode == "decompose":
            prompt = (
                f"Given the question: {query}\n\n"
                f"And the following context:\n{chunk_text}\n\n"
                "What sub-questions should be explored next to fully answer this? "
                "Return one focused follow-up question."
            )
        else:
            prompt = (
                f"Original question: {query}\n\n"
                f"Retrieved context:\n{chunk_text}\n\n"
                "Based on this context, refine the question to focus on "
                "what additional information is still needed. "
                "Return only the refined question."
            )

        template = p.get("prompt_template")
        if template:
            prompt = template.format(query=query, context=chunk_text)

        result = await ctx.llm.aask(msg=[{"role": "user", "content": prompt}])
        return {"query": SlotValue(kind=SlotKind.QUERY_TEXT, data=result.strip(), producer="meta.reason_step")}

    except Exception as e:
        logger.exception(f"meta_reason_step failed: {e}")
        return {"query": SlotValue(kind=SlotKind.QUERY_TEXT, data=query, producer="meta.reason_step")}
