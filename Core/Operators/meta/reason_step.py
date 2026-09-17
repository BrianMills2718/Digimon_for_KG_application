"""Meta: evidence-gated LLM reasoning-step operator."""

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
    Inputs:  {"query": QUERY_TEXT, "chunks": CHUNK_SET}
    Outputs: {"query": QUERY_TEXT}

    Params:  {"prompt_template": str, "mode": "refine"|"decompose"}

    This is an advisory reasoning heuristic, not a license to invent state. If
    retrieval produced no usable evidence, preserve the current query unchanged
    so the harness can choose another retrieval route explicitly.
    """
    query = inputs["query"].data
    chunks = inputs.get("chunks")
    chunk_data = chunks.data if chunks else []
    p = params or {}
    mode = p.get("mode", "refine")

    chunk_text = "\n\n".join(
        str(getattr(chunk, "text", "") or "").strip()
        for chunk in chunk_data
        if str(getattr(chunk, "text", "") or "").strip()
    )

    if not chunk_text:
        return {
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data=query,
                producer="meta.reason_step",
                metadata={
                    "status": "unchanged_no_evidence",
                    "mode": mode,
                },
            )
        }

    try:
        if mode == "decompose":
            prompt = (
                f"Given the question: {query}\n\n"
                f"And the following retrieved evidence:\n{chunk_text}\n\n"
                "Suggest one focused follow-up question that would resolve the most important "
                "remaining information gap. Do not invent facts not present in the evidence. "
                "Return only the follow-up question."
            )
        else:
            prompt = (
                f"Current information need: {query}\n\n"
                f"Retrieved evidence:\n{chunk_text}\n\n"
                "Refine the information need to focus only on additional information still required. "
                "Do not introduce unsupported entities, claims, or relationships. "
                "Return only the refined question."
            )

        template = p.get("prompt_template")
        if template:
            prompt = template.format(query=query, context=chunk_text)

        result = await ctx.llm.aask(
            msg=[{"role": "user", "content": prompt}]
        )
        refined = str(result or "").strip()
        if not refined:
            refined = query
            status = "unchanged_empty_response"
        else:
            status = "refined_from_evidence"

        return {
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data=refined,
                producer="meta.reason_step",
                metadata={
                    "status": status,
                    "mode": mode,
                    "evidence_chunks": len(chunk_data),
                },
            )
        }

    except Exception as exc:
        logger.exception(f"meta_reason_step failed: {exc}")
        return {
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data=query,
                producer="meta.reason_step",
                metadata={
                    "status": "unchanged_error",
                    "mode": mode,
                    "error": str(exc),
                },
            )
        }
