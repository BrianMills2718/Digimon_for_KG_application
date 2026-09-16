"""Meta: grounded answer generation operator."""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue


INSUFFICIENT_EVIDENCE_ANSWER = (
    "Insufficient retrieved evidence to answer the question."
)


async def meta_generate_answer(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": QUERY_TEXT, "chunks": CHUNK_SET}
    Outputs: {"answer": QUERY_TEXT}
    Params:  {"system_prompt": str, "response_type": str}
    """
    query = inputs["query"].data
    chunks = inputs.get("chunks")
    chunk_data = chunks.data if chunks else []
    p = params or {}

    evidence_texts = [
        str(chunk.text).strip()
        for chunk in chunk_data
        if getattr(chunk, "text", None) and str(chunk.text).strip()
    ]
    if not evidence_texts:
        return {
            "answer": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data=INSUFFICIENT_EVIDENCE_ANSWER,
                producer="meta.generate_answer",
                metadata={"status": "insufficient_evidence"},
            )
        }

    context = "\n\n---\n\n".join(evidence_texts)

    try:
        system_prompt = p.get("system_prompt")
        if system_prompt:
            grounded_system = system_prompt.format(
                context_data=context,
                response_type=p.get("response_type", ""),
            )
            grounded_system += (
                "\n\nUse only the supplied evidence. Do not invent unsupported facts. "
                f"If the evidence is insufficient, reply exactly: {INSUFFICIENT_EVIDENCE_ANSWER}"
            )
            response = await ctx.llm.aask(
                msg=query,
                system_msgs=[grounded_system],
            )
        else:
            prompt = (
                f"Evidence:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer using only the evidence above. Do not add facts that are not "
                "supported by the evidence. If the evidence does not support an answer, "
                f"reply exactly: {INSUFFICIENT_EVIDENCE_ANSWER}"
            )
            response = await ctx.llm.aask(
                msg=[{"role": "user", "content": prompt}]
            )

        response = str(response).strip()
        if not response:
            response = INSUFFICIENT_EVIDENCE_ANSWER

        return {
            "answer": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data=response,
                producer="meta.generate_answer",
                metadata={
                    "status": (
                        "insufficient_evidence"
                        if response == INSUFFICIENT_EVIDENCE_ANSWER
                        else "grounded_answer"
                    ),
                    "evidence_chunks": len(evidence_texts),
                },
            )
        }
    except Exception as exc:
        logger.exception(f"meta_generate_answer failed: {exc}")
        return {
            "answer": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="Failed to generate answer.",
                producer="meta.generate_answer",
                metadata={"status": "error", "error": str(exc)},
            )
        }
