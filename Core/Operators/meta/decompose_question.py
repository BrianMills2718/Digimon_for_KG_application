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
    max_questions = p.get("max_questions", 5)

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

        # Parse JSON array from response.
        match = re.search(r"\[.*?\]", result, re.DOTALL)
        if match:
            sub_qs = json.loads(match.group())
        else:
            sub_qs = [q.strip().strip('"').strip("'") for q in result.split("\n") if q.strip()]

        # Transitional representation: EntityRecord.entity_name carries sub-question text.
        records = [
            EntityRecord(
                entity_name=q,
                entity_type="sub_question",
                score=1.0,
                extra={"advisory": True, "decomposition_index": i + 1},
            )
            for i, q in enumerate(sub_qs[:max_questions])
            if isinstance(q, str) and q.strip()
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
        return {
            "sub_questions": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[],
                producer="meta.decompose_question",
                metadata={"advisory": True, "error": str(e)},
            )
        }
