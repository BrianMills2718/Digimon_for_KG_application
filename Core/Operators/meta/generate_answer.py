"""Meta: evidence-grounded answer generation operator."""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue


INSUFFICIENT_EVIDENCE_ANSWER = "Insufficient retrieved evidence to answer the question."


def _evidence_entries(chunk_data) -> list[tuple[str, str]]:
    """Return stable ``(evidence_id, text)`` pairs from non-empty chunks."""
    entries = []
    seen_ids = set()
    for index, chunk in enumerate(chunk_data or []):
        text = str(getattr(chunk, "text", "") or "").strip()
        if not text:
            continue
        evidence_id = str(getattr(chunk, "chunk_id", "") or f"evidence-{index + 1}")
        if evidence_id in seen_ids:
            continue
        seen_ids.add(evidence_id)
        entries.append((evidence_id, text))
    return entries


async def meta_generate_answer(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": QUERY_TEXT, "chunks": CHUNK_SET}
    Outputs: {"answer": QUERY_TEXT}
    Params:  {"system_prompt": str, "response_type": str}

    Evidence IDs are preserved in the prompt and output metadata. Factual claims
    should cite the supporting ID in square brackets, e.g. ``[chunk-abc]``.
    """
    query = inputs["query"].data
    chunks = inputs.get("chunks")
    chunk_data = chunks.data if chunks else []
    p = params or {}

    evidence = _evidence_entries(chunk_data)
    if not evidence:
        return {
            "answer": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data=INSUFFICIENT_EVIDENCE_ANSWER,
                producer="meta.generate_answer",
                metadata={
                    "status": "insufficient_evidence",
                    "evidence_chunk_ids": [],
                },
            )
        }

    evidence_ids = [evidence_id for evidence_id, _text in evidence]
    context = "\n\n---\n\n".join(
        f"[{evidence_id}]\n{text}" for evidence_id, text in evidence
    )
    grounding_instruction = (
        "Use only the supplied evidence. Do not invent unsupported facts. "
        "For factual claims, cite the supporting evidence ID in square brackets "
        "using only IDs that appear in the evidence. "
        f"If the evidence is insufficient, reply exactly: {INSUFFICIENT_EVIDENCE_ANSWER}"
    )

    try:
        system_prompt = p.get("system_prompt")
        if system_prompt:
            grounded_system = system_prompt.format(
                context_data=context,
                response_type=p.get("response_type", ""),
            )
            grounded_system += f"\n\n{grounding_instruction}"
            response = await ctx.llm.aask(
                msg=query,
                system_msgs=[grounded_system],
            )
        else:
            prompt = (
                f"Evidence:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"{grounding_instruction}"
            )
            response = await ctx.llm.aask(
                msg=[{"role": "user", "content": prompt}]
            )

        response = str(response).strip() or INSUFFICIENT_EVIDENCE_ANSWER
        status = (
            "insufficient_evidence"
            if response == INSUFFICIENT_EVIDENCE_ANSWER
            else "grounded_answer"
        )

        return {
            "answer": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data=response,
                producer="meta.generate_answer",
                metadata={
                    "status": status,
                    "evidence_chunks": len(evidence),
                    "evidence_chunk_ids": evidence_ids,
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
                metadata={
                    "status": "error",
                    "error": str(exc),
                    "evidence_chunk_ids": evidence_ids,
                },
            )
        }
