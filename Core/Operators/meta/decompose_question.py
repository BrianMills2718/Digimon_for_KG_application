"""Meta: advisory LLM question-decomposition operator.

Provides a lightweight, dependency-aware Atom-of-Thought / Graph-of-Thought
planning heuristic for a capable harness. It does not define or execute a
mandatory reasoning graph.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


def _fallback_subquestions(query: str) -> list[str]:
    """Fail conservatively by preserving the original question unchanged."""
    value = str(query or "").strip()
    return [value] if value else []


def _parse_subquestions(response: Any, query: str) -> list[str]:
    """Parse only a JSON array of strings; otherwise preserve the query.

    Decomposition is advisory. Turning arbitrary explanatory prose into planner
    state is worse than declining to decompose, so malformed model output falls
    back to the original question instead of line-splitting prose.
    """
    if isinstance(response, list):
        values = response
    else:
        text = str(response or "").strip()
        match = re.search(r"\[[\s\S]*?\]", text)
        if match is None:
            return _fallback_subquestions(query)
        try:
            values = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return _fallback_subquestions(query)

    if not isinstance(values, list):
        return _fallback_subquestions(query)

    cleaned = [str(value).strip() for value in values if isinstance(value, str) and value.strip()]
    return cleaned or _fallback_subquestions(query)


async def meta_decompose_question(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """Suggest a small dependency-aware decomposition when it helps.

    Inputs:  {"query": SlotValue(QUERY_TEXT)}
    Outputs: {"sub_questions": SlotValue(ENTITY_SET)}
    Params:  {"max_questions": int (default 5)}

    ``ENTITY_SET`` is currently a transitional carrier for sub-question text:
    each ``EntityRecord.entity_name`` contains one suggested sub-question.
    The output is advisory; the calling harness may merge, reorder, branch,
    revise, parallelize, or ignore the suggestions.
    """
    query = inputs["query"].data
    p = params or {}
    try:
        max_questions = max(1, int(p.get("max_questions", 5)))
    except (TypeError, ValueError):
        max_questions = 5

    try:
        prompt = (
            "You are a planning heuristic for an intelligent agent harness operating DIGIMON.\n\n"
            "Do not impose a fixed reasoning graph and do not assume decomposition is always needed. "
            "The harness remains responsible for choosing tools, revising its approach after observations, "
            "and deciding when to stop.\n\n"
            "When decomposition helps, suggest the smallest set of focused sub-questions that makes "
            "information dependencies explicit. Some sub-questions may be independent; others may depend "
            "on an entity, relation, attribute, or evidence discovered by an earlier item. Do not force "
            "dependency-linked work into falsely independent questions.\n\n"
            "Prefer evidence-seeking sub-questions. Do not invent intermediate answers, entities, or "
            "relationships. Preserve ambiguity until evidence resolves it. Do not prescribe tool calls.\n\n"
            "Use q1:, q2:, etc. prefixes. If a later item depends on an earlier discovery, use a semantic "
            "placeholder such as <q1.entity> or <q2.result>. The harness may choose a different route.\n\n"
            "If decomposition would not help, return a one-item array containing the original question.\n\n"
            f"Question: {query}\n\n"
            f"Suggest up to {max_questions} focused sub-questions. Return only a JSON array of strings."
        )
        result = await ctx.llm.aask(msg=[{"role": "user", "content": prompt}])
        sub_qs = _parse_subquestions(result, str(query))

        # Transitional representation: EntityRecord.entity_name carries sub-question text.
        records = [
            EntityRecord(
                entity_name=q,
                entity_type="sub_question",
                score=1.0,
                extra={"advisory": True, "decomposition_index": i + 1},
            )
            for i, q in enumerate(sub_qs[:max_questions])
        ]

        logger.info(f"meta_decompose_question: suggested {len(records)} sub-questions")
        return {
            "sub_questions": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=records,
                producer="meta.decompose_question",
                metadata={"advisory": True, "reasoning_policy": "aot_got_heuristic"},
            )
        }

    except Exception as e:
        logger.exception(f"meta_decompose_question failed: {e}")
        fallback = [
            EntityRecord(
                entity_name=q,
                entity_type="sub_question",
                score=1.0,
                extra={"advisory": True, "decomposition_index": i + 1},
            )
            for i, q in enumerate(_fallback_subquestions(str(query)))
        ]
        return {
            "sub_questions": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=fallback,
                producer="meta.decompose_question",
                metadata={"advisory": True, "error": str(e), "fallback": "original_query"},
            )
        }
