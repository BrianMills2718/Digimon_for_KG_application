"""Relationship-agent operator for Think-on-Graph exploration.

Scores candidate graph relations for every current beam entity. When a graph
has only DIGIMON's generic ``relationship`` label, the edge description is used
as the semantic relation label so ToG still has meaningful choices.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Optional

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Schema.SlotTypes import RelationshipRecord, SlotKind, SlotValue


_GENERIC_RELATION_LABELS = {"", "relationship", "related_to", "unknown_relationship"}


def _relation_labels(edge_data: dict) -> list[str]:
    """Return semantic labels suitable for ToG relation selection."""
    relation_name = str(edge_data.get("relation_name", "") or "").strip()
    labels = [
        value.strip()
        for value in relation_name.split(GRAPH_FIELD_SEP)
        if value.strip()
    ]
    meaningful = [
        label
        for label in labels
        if label.lower() not in _GENERIC_RELATION_LABELS
    ]
    if meaningful:
        return meaningful

    # Default ER extraction has no typed relation name. Use the evidence-bearing
    # description (or keyword summary) instead of presenting every edge to the
    # LLM as the indistinguishable label "relationship".
    description = str(edge_data.get("description", "") or "").strip()
    if description:
        return [description[:300]]

    keywords = str(edge_data.get("keywords", "") or "").strip()
    if keywords:
        return [keywords[:300]]

    return ["related_to"]


async def relationship_agent(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": QUERY_TEXT, "entities": ENTITY_SET}
    Outputs: {"relationships": RELATIONSHIP_SET}
    Params:  {"width": int}

    Every input entity is explored. Each output record stores the candidate
    neighbor mapping in ``extra['relations_dict']`` for ``entity.rel_node`` and
    ``entity.agent`` to select the next-hop entities.
    """
    from Core.Prompt.TogPrompt import extract_relation_prompt

    query = inputs["query"].data
    entities = inputs["entities"].data
    width = max(1, int((params or {}).get("width", 3)))

    if not entities:
        return {
            "relationships": SlotValue(
                kind=SlotKind.RELATIONSHIP_SET,
                data=[],
                producer="relationship.agent",
            )
        }

    records = []

    for entity_record in entities:
        entity = entity_record.entity_name
        try:
            edges = await ctx.graph.get_node_edges(source_node_id=entity)
            if not edges:
                continue

            relations_dict = defaultdict(list)
            relation_labels = []

            for edge in edges:
                if not edge or len(edge) < 2:
                    continue
                src, tgt = str(edge[0]), str(edge[1])
                edge_data = await ctx.graph.get_edge(src, tgt)
                if edge_data is None:
                    edge_data = await ctx.graph.get_edge(tgt, src)
                edge_data = edge_data or {}

                neighbor = tgt if src == entity else src
                for label in _relation_labels(edge_data):
                    if neighbor not in relations_dict[(entity, label)]:
                        relations_dict[(entity, label)].append(neighbor)
                    relation_labels.append(label)

            candidates = list(dict.fromkeys(relation_labels))
            if not candidates:
                continue

            prompt = (
                extract_relation_prompt % (str(width), str(width), str(width))
                + query
                + "\nTopic Entity: "
                + entity
                + f"\nRelations: There are {len(candidates)} relations provided in total, separated by ;."
                + "; ".join(candidates)
                + ";\nA: "
            )

            response = str(
                await ctx.llm.aask(
                    msg=[
                        {
                            "role": "system",
                            "content": "You are an AI assistant that helps people find information.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
            )

            candidate_lookup = {label.casefold(): label for label in candidates}
            pattern = r"\{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)\}"
            selected_for_entity = []

            for match in re.finditer(pattern, response):
                proposed = match.group("relation").strip()
                if ";" in proposed:
                    continue
                label = candidate_lookup.get(proposed.casefold())
                if label is None:
                    continue
                try:
                    score = float(match.group("score"))
                except ValueError:
                    continue
                selected_for_entity.append((score, label))

            selected_for_entity.sort(key=lambda item: item[0], reverse=True)
            for score, label in selected_for_entity[:width]:
                records.append(
                    RelationshipRecord(
                        src_id=entity,
                        tgt_id="",
                        relation_name=label,
                        description=label,
                        score=score,
                        extra={
                            "head": True,
                            "relations_dict": dict(relations_dict),
                        },
                    )
                )

        except Exception as exc:
            logger.exception(
                f"relationship.agent failed while exploring '{entity}': {exc}"
            )

    records.sort(
        key=lambda record: float(record.score or 0.0),
        reverse=True,
    )
    return {
        "relationships": SlotValue(
            kind=SlotKind.RELATIONSHIP_SET,
            data=records,
            producer="relationship.agent",
        )
    }
