"""Meta: evidence-aware LLM answer-synthesis operator.

Merges sub-answers or retrieved chunks into a coherent final answer while
preserving available evidence markers and unresolved uncertainty.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue


INSUFFICIENT_EVIDENCE_ANSWER = "Insufficient retrieved evidence to answer the question."


def _format_evidence_item(item: Any) -> str:
    """Render one chunk/sub-answer with whatever provenance markers are available."""
    text = getattr(item, "text", "") or ""
    markers = []

    chunk_id = getattr(item, "chunk_id", "")
    if chunk_id:
        markers.append(f"chunk_id={chunk_id}")

    extra = getattr(item, "extra", {}) or {}
    for key in ("source_id", "document_id", "doc_id", "citation", "provenance"):
        value = extra.get(key)
        if value:
            markers.append(f"{key}={value}")

    prefix = f"[{', '.join(markers)}] " if markers else ""
    return f"- {prefix}{text}"


async def meta_synthesize_answers(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """Synthesize supplied evidence/sub-answers into a final answer.

    Inputs:  {"query": SlotValue(QUERY_TEXT), "chunks": SlotValue(CHUNK_SET)}
    Outputs: {"answer": SlotValue(QUERY_TEXT)}
    Params:  {"synthesis_style": str (default "concise")}

    This operator does not create provenance that upstream retrieval did not
    provide. It preserves available chunk/source markers in the model context
    and instructs synthesis not to bridge unsupported facts.
    """
    query = inputs["query"].data
    chunks = inputs.get("chunks")
    chunk_data = chunks.data if chunks else []
    p = params or {}
    style = p.get("synthesis_style", "concise")

    evidence_lines = [
        _format_evidence_item(c)
        for c in chunk_data
        if str(getattr(c, "text", "") or "").strip()
    ]
    evidence_ids = [
        str(getattr(c, "chunk_id", ""))
        for c in chunk_data
        if str(getattr(c, "text", "") or "").strip()
        and getattr(c, "chunk_id", "")
    ]

    if not evidence_lines:
        return {
            "answer": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data=INSUFFICIENT_EVIDENCE_ANSWER,
                producer="meta.synthesize_answers",
                metadata={
                    "status": "insufficient_evidence",
                    "evidence_item_count": 0,
                    "evidence_chunk_ids": [],
                },
            )
        }

    try:
        evidence_block = "\n".join(evidence_lines)
        prompt = (
            "You synthesize retrieved evidence and sub-answers into one coherent answer.\n\n"
            "The evidence may come from different reasoning paths. Do not assume every item is complete, "
            "mutually consistent, or equally well supported.\n\n"
            "Rules:\n"
            "- Answer the original question directly and without unnecessary repetition.\n"
            "- Only assert claims supported by the supplied evidence.\n"
            "- Preserve source, citation, chunk, or provenance markers when they are present and useful.\n"
            "- Distinguish retrieved evidence from inferred conclusions when that distinction matters.\n"
            "- If evidence conflicts, report the conflict instead of silently choosing one version.\n"
            "- If an information dependency remains unresolved, state what is missing rather than inventing a bridge.\n"
            "- Absence of retrieved evidence is not itself evidence that a claim is false.\n"
            "- Do not manufacture confidence scores or certainty that the evidence does not justify.\n\n"
            f"Original question: {query}\n\n"
            f"Evidence / sub-answers:\n{evidence_block}\n\n"
            f"Produce a {style} final answer."
        )
        response = await ctx.llm.aask(msg=[{"role": "user", "content": prompt}])

        logger.info(
            f"meta_synthesize_answers: synthesized {len(evidence_lines)} evidence items"
        )
        return {
            "answer": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data=str(response).strip() or INSUFFICIENT_EVIDENCE_ANSWER,
                producer="meta.synthesize_answers",
                metadata={
                    "status": "synthesized",
                    "evidence_item_count": len(evidence_lines),
                    "evidence_chunk_ids": list(dict.fromkeys(evidence_ids)),
                },
            )
        }

    except Exception as e:
        logger.exception(f"meta_synthesize_answers failed: {e}")
        return {
            "answer": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="Failed to synthesize answer.",
                producer="meta.synthesize_answers",
                metadata={
                    "status": "error",
                    "error": str(e),
                    "evidence_item_count": len(evidence_lines),
                    "evidence_chunk_ids": list(dict.fromkeys(evidence_ids)),
                },
            )
        }
