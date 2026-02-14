"""Relationship agent operator.

Use LLM to select top-K relations from an entity's edges.
Ported from RelationshipRetriever._find_relevant_relations_by_entity_agent.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Optional

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Schema.SlotTypes import RelationshipRecord, SlotKind, SlotValue


async def relationship_agent(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": SlotValue(QUERY_TEXT), "entities": SlotValue(ENTITY_SET)}
    Outputs: {"relationships": SlotValue(RELATIONSHIP_SET)}
    Params:  {"width": int, "pre_relations_name": list, "pre_head": bool}

    Processes the first entity in the set. For multi-entity ToG, call in a loop.
    """
    from Core.Prompt.TogPrompt import extract_relation_prompt

    query = inputs["query"].data
    entities = inputs["entities"].data
    p = params or {}
    width = p.get("width", 3)
    pre_relations_name = p.get("pre_relations_name")
    pre_head = p.get("pre_head")

    if not entities:
        return {"relationships": SlotValue(kind=SlotKind.RELATIONSHIP_SET, data=[], producer="relationship.agent")}

    entity = entities[0].entity_name

    try:
        edges = await ctx.graph.get_node_edges(source_node_id=entity)
        rel_names_raw = await ctx.graph.get_edge_relation_name_batch(edges=edges)
        rel_names = [r.split(GRAPH_FIELD_SEP) for r in rel_names_raw]

        relations_dict = defaultdict(list)
        for i, edge in enumerate(edges):
            src, tgt = edge[0], edge[1]
            for rel in rel_names[i]:
                relations_dict[(src, rel)].append(tgt)

        head_relations = []
        tail_relations = []
        for i, rels in enumerate(rel_names):
            if edges[i][0] == entity:
                head_relations.extend(rels)
            else:
                tail_relations.extend(rels)

        if pre_relations_name:
            if pre_head:
                tail_relations = list(set(tail_relations) - set(pre_relations_name))
            else:
                head_relations = list(set(head_relations) - set(pre_relations_name))

        head_set = set(head_relations)
        total_relations = sorted(set(head_relations + tail_relations))

        prompt = (
            extract_relation_prompt % (str(width), str(width), str(width))
            + query + "\nTopic Entity: " + entity
            + f"\nRelations: There are {len(total_relations)} relations provided in total, separated by ;."
            + "; ".join(total_relations) + ";\nA: "
        )

        result = await ctx.llm.aask(msg=[
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": prompt},
        ])

        pattern = r"\{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)\}"
        records = []
        for match in re.finditer(pattern, result):
            relation = match.group("relation").strip()
            if ";" in relation:
                continue
            try:
                score = float(match.group("score"))
            except ValueError:
                continue
            is_head = relation in head_set
            records.append(RelationshipRecord(
                src_id=entity if is_head else "",
                tgt_id="" if is_head else entity,
                relation_name=relation,
                score=score,
                extra={"head": is_head, "relations_dict": dict(relations_dict)},
            ))

        return {"relationships": SlotValue(
            kind=SlotKind.RELATIONSHIP_SET, data=records, producer="relationship.agent",
        )}

    except Exception as e:
        logger.exception(f"relationship_agent failed: {e}")
        return {"relationships": SlotValue(kind=SlotKind.RELATIONSHIP_SET, data=[], producer="relationship.agent")}
