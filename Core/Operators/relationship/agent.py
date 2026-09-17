"""Relationship-agent operator for Think-on-Graph exploration.

Scores candidate graph relations for every current beam entity. When a graph
has only DIGIMON's generic ``relationship`` label, the edge description is used
as the semantic relation label so ToG still has meaningful choices. Selected
relations preserve their original graph ``source_id`` values so downstream
answer generation can remain grounded in the documents used for each hop.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Optional

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Common.Utils import split_string_by_multi_markers
from Core.Schema.SlotTypes import RelationshipRecord, SlotKind, SlotValue


_GENERIC_RELATION_LABELS = {"", "relationship", "related_to", "unknown_relationship"}


def _safe_relation_label(value: Any, max_length: int = 220) -> str:
    """Normalize graph text into a ToG prompt/parser-safe relation label."""
    text = " ".join(str(value or "").split())
    for char, replacement in ((";", ","), ("(", "["), (")", "]"), ("{", ""), ("}", "")):
        text = text.replace(char, replacement)
    return text[:max_length].strip()


def _relation_labels(edge_data: dict) -> list[str]:
    relation_name = str(edge_data.get("relation_name", "") or "").strip()
    labels = [
        _safe_relation_label(value)
        for value in relation_name.split(GRAPH_FIELD_SEP)
        if value.strip()
    ]
    meaningful = [
        label
        for label in labels
        if label and label.lower() not in _GENERIC_RELATION_LABELS
    ]
    if meaningful:
        return meaningful

    description = _safe_relation_label(edge_data.get("description", ""))
    if description:
        return [description]

    keywords = _safe_relation_label(edge_data.get("keywords", ""))
    if keywords:
        return [keywords]

    return ["related_to"]


def _append_source_ids(target: list[str], source_id: Any) -> None:
    if not source_id:
        return
    for chunk_id in split_string_by_multi_markers(
        str(source_id), [GRAPH_FIELD_SEP]
    ):
        if chunk_id and chunk_id not in target:
            target.append(chunk_id)


def _fallback_relation_selections(entity, candidates, relation_weights, width):
    """Keep graph exploration alive if the LLM response format is unusable."""
    ranked = []
    for label in candidates:
        weights = relation_weights.get((entity, label), [])
        score = max(weights) if weights else 1.0
        ranked.append((float(score), label, True))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:width]


async def relationship_agent(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": QUERY_TEXT, "entities": ENTITY_SET}
    Outputs: {"relationships": RELATIONSHIP_SET}
    Params:  {"width": int}

    Every input entity is explored. Each selected relation carries both the
    candidate-neighbor mapping used by the next ToG hop and the merged source
    chunk IDs of the concrete graph edges represented by that relation.
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
            relation_sources = defaultdict(list)
            relation_weights = defaultdict(list)
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
                    key = (entity, label)
                    if neighbor not in relations_dict[key]:
                        relations_dict[key].append(neighbor)
                    _append_source_ids(
                        relation_sources[key],
                        edge_data.get("source_id", ""),
                    )
                    try:
                        relation_weights[key].append(
                            float(edge_data.get("weight", 0.0) or 0.0)
                        )
                    except (TypeError, ValueError):
                        pass
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
                proposed = _safe_relation_label(match.group("relation"))
                label = candidate_lookup.get(proposed.casefold())
                if label is None:
                    continue
                try:
                    score = float(match.group("score"))
                except ValueError:
                    continue
                selected_for_entity.append((score, label, False))

            if not selected_for_entity:
                logger.warning(
                    f"relationship.agent could not parse scored relations for '{entity}'; "
                    "falling back to graph-weight ordering"
                )
                selected_for_entity = _fallback_relation_selections(
                    entity,
                    candidates,
                    relation_weights,
                    width,
                )
            else:
                selected_for_entity.sort(key=lambda item: item[0], reverse=True)
                selected_for_entity = selected_for_entity[:width]

            for score, label, used_fallback in selected_for_entity:
                key = (entity, label)
                weights = relation_weights.get(key, [])
                records.append(
                    RelationshipRecord(
                        src_id=entity,
                        tgt_id="",
                        relation_name=label,
                        description=label,
                        source_id=GRAPH_FIELD_SEP.join(relation_sources.get(key, [])),
                        weight=max(weights) if weights else 0.0,
                        score=score,
                        extra={
                            "head": True,
                            "relations_dict": dict(relations_dict),
                            "selection_fallback": used_fallback,
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
