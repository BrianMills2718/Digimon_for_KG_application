"""Entity agent operator.

Use LLM to score and select entity candidates from relationships.
Ported from EntityRetriever._find_relevant_entities_by_relationships_agent.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def entity_agent(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": SlotValue(QUERY_TEXT), "entity_relation_list": SlotValue(ENTITY_SET)}
    Outputs: {"entities": SlotValue(ENTITY_SET)}
    Params:  {"width": int, "relations_dict": dict}

    The entity_relation_list input should have EntityRecords whose 'extra' dict contains
    'relation', 'score', 'head' keys — matching the ToG pattern.
    relations_dict maps (entity, relation) -> [target_entities].
    """
    from Core.Prompt.TogPrompt import score_entity_candidates_prompt

    query = inputs["query"].data
    entity_rel_list = inputs["entity_relation_list"].data  # List[EntityRecord]
    p = params or {}
    width = p.get("width", 3)
    relations_dict = p.get("relations_dict", defaultdict(list))

    total_candidates = []
    total_scores = []
    total_relations = []
    total_topic_entities = []
    total_head = []

    for entity_rec in entity_rel_list:
        entity_name = entity_rec.entity_name
        relation = entity_rec.extra.get("relation", "")
        entity_score = entity_rec.score or 0.0
        head = entity_rec.extra.get("head", True)

        candidate_list = relations_dict.get((entity_name, relation), [])

        if len(candidate_list) <= 1:
            scores = [entity_score] if candidate_list else [0.0]
        else:
            prompt = score_entity_candidates_prompt.format(query, relation) + "; ".join(
                candidate_list
            ) + ";\nScore: "
            result = await ctx.llm.aask(msg=[{"role": "user", "content": prompt}])
            scores = [float(s) for s in re.findall(r"\d+\.\d+", result)]
            if len(scores) != len(candidate_list):
                scores = [1 / len(candidate_list)] * len(candidate_list)

        if not candidate_list:
            candidate_list = ["[FINISH]"]

        total_candidates.extend(candidate_list)
        total_scores.extend(scores)
        total_relations.extend([relation] * len(candidate_list))
        total_topic_entities.extend([entity_name] * len(candidate_list))
        total_head.extend([head] * len(candidate_list))

    # Prune by width
    zipped = sorted(
        zip(total_relations, total_candidates, total_topic_entities, total_head, total_scores),
        key=lambda x: x[4], reverse=True,
    )[:width]

    records = []
    for rel, cand, topic, head, score in zipped:
        if score == 0:
            continue
        records.append(EntityRecord(
            entity_name=cand,
            score=score,
            extra={"relation": rel, "topic_entity": topic, "head": head},
        ))

    return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=records, producer="entity.agent")}
