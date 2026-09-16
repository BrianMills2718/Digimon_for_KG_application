"""Entity agent operator used by Think-on-Graph exploration.

Consumes the scored relationship choices produced by ``relationship.agent`` and
selects the next entity candidates for the following graph hop.
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
    Inputs:  {"query": QUERY_TEXT, "relationships": RELATIONSHIP_SET}
    Outputs: {"entities": ENTITY_SET}
    Params:  {"width": int}
    """
    from Core.Prompt.TogPrompt import score_entity_candidates_prompt

    query = inputs["query"].data
    relationships = inputs["relationships"].data
    width = (params or {}).get("width", 3)

    if not relationships:
        return {
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[],
                producer="entity.agent",
            )
        }

    ranked_candidates = []

    for relationship in relationships:
        relation = relationship.relation_name
        relation_score = float(relationship.score or 0.0)
        head = bool(relationship.extra.get("head", bool(relationship.src_id)))
        topic_entity = relationship.src_id if head else relationship.tgt_id
        relations_dict = relationship.extra.get("relations_dict", {}) or {}
        candidate_list = list(relations_dict.get((topic_entity, relation), []))

        if not candidate_list:
            continue

        if len(candidate_list) == 1:
            candidate_scores = [relation_score]
        else:
            prompt = (
                score_entity_candidates_prompt.format(query, relation)
                + "; ".join(candidate_list)
                + ";\nScore: "
            )
            try:
                result = await ctx.llm.aask(
                    msg=[{"role": "user", "content": prompt}]
                )
                candidate_scores = [
                    float(value)
                    for value in re.findall(r"\d+(?:\.\d+)?", result)
                ]
            except Exception as exc:
                logger.warning(
                    f"entity.agent scoring failed for relation '{relation}': {exc}"
                )
                candidate_scores = []

            if len(candidate_scores) != len(candidate_list):
                # Keep relation relevance as the fallback signal rather than
                # inventing a second arbitrary ranking.
                candidate_scores = [relation_score] * len(candidate_list)

        for candidate, candidate_score in zip(candidate_list, candidate_scores):
            score = float(candidate_score)
            if score <= 0.0:
                continue
            ranked_candidates.append(
                (
                    score,
                    EntityRecord(
                        entity_name=str(candidate),
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

    # Deduplicate candidates while preserving highest score.
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
