"""Entity agent operator used by Think-on-Graph exploration.

Consumes candidate EntityRecords produced from scored relationships by
``entity.rel_node`` and selects the next graph entities for the following hop.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def entity_agent(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": QUERY_TEXT, "entity_relation_list": ENTITY_SET}
    Outputs: {"entities": ENTITY_SET}
    Params:  {"width": int}

    Each candidate record carries these ToG fields in ``extra``:
    ``relation``, ``head``, and ``relations_dict``.
    """
    from Core.Prompt.TogPrompt import score_entity_candidates_prompt

    query = inputs["query"].data
    candidates = inputs["entity_relation_list"].data
    width = (params or {}).get("width", 3)

    if not candidates:
        return {
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[],
                producer="entity.agent",
            )
        }

    ranked_candidates = []

    for candidate_record in candidates:
        topic_entity = candidate_record.entity_name
        relation = candidate_record.extra.get("relation", "")
        relation_score = float(candidate_record.score or 0.0)
        head = bool(candidate_record.extra.get("head", True))
        relations_dict = candidate_record.extra.get("relations_dict", {}) or {}
        entity_candidates = list(relations_dict.get((topic_entity, relation), []))

        if not entity_candidates:
            continue

        if len(entity_candidates) == 1:
            scores = [relation_score]
        else:
            prompt = (
                score_entity_candidates_prompt.format(query, relation)
                + "; ".join(entity_candidates)
                + ";\nScore: "
            )
            try:
                result = await ctx.llm.aask(
                    msg=[{"role": "user", "content": prompt}]
                )
                scores = [
                    float(value)
                    for value in re.findall(r"\d+(?:\.\d+)?", result)
                ]
            except Exception as exc:
                logger.warning(
                    f"entity.agent scoring failed for relation '{relation}': {exc}"
                )
                scores = []

            if len(scores) != len(entity_candidates):
                scores = [relation_score] * len(entity_candidates)

        for entity_name, score in zip(entity_candidates, scores):
            score = float(score)
            if score <= 0.0:
                continue
            ranked_candidates.append(
                (
                    score,
                    EntityRecord(
                        entity_name=str(entity_name),
                        score=score,
                        extra={
                            "relation": relation,
                            "topic_entity": topic_entity,
                            "head": head,
                        },
                    ),
                )
            )

    ranked_candidates.sort(key=lambda item: item[0], reverse=True)

    records = []
    seen = set()
    for _score, record in ranked_candidates:
        if record.entity_name in seen:
            continue
        seen.add(record.entity_name)
        records.append(record)
        if len(records) >= width:
            break

    return {
        "entities": SlotValue(
            kind=SlotKind.ENTITY_SET,
            data=records,
            producer="entity.agent",
        )
    }
