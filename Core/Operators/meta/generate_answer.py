"""Meta: LLM answer generation operator.

Generate a final answer from query + retrieved chunks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue


async def meta_generate_answer(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": SlotValue(QUERY_TEXT), "chunks": SlotValue(CHUNK_SET)}
    Outputs: {"answer": SlotValue(QUERY_TEXT)}  -- using QUERY_TEXT kind for raw text
    Params:  {"system_prompt": str, "response_type": str}
    """
    query = inputs["query"].data
    chunks = inputs.get("chunks")
    chunk_data = chunks.data if chunks else []
    p = params or {}

    context = "\n\n---\n\n".join(c.text for c in chunk_data if c.text)

    try:
        system_prompt = p.get("system_prompt")
        if system_prompt:
            response = await ctx.llm.aask(
                msg=query,
                system_msgs=[system_prompt.format(context_data=context, response_type=p.get("response_type", ""))],
            )
        else:
            prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer the question based on the provided context. "
                "Be concise and accurate."
            )
            response = await ctx.llm.aask(msg=[{"role": "user", "content": prompt}])

        return {"answer": SlotValue(kind=SlotKind.QUERY_TEXT, data=response, producer="meta.generate_answer")}

    except Exception as e:
        logger.exception(f"meta_generate_answer failed: {e}")
        return {"answer": SlotValue(kind=SlotKind.QUERY_TEXT, data="Failed to generate answer.", producer="meta.generate_answer")}
