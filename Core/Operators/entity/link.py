"""Entity linking operator.

Link entity mentions to graph entities via VDB top-1 matching.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def entity_link(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": ENTITY_SET} -- entity mentions to canonicalize
    Outputs: {"entities": ENTITY_SET} -- linked graph entities
    Params:  {"similarity_threshold": float, "top_k": int}

    The VDB adapter exposes higher-is-better normalized similarity. Preserve
    that score on the linked EntityRecord so downstream reasoning can use it.
    Default threshold is 0.0 to preserve the previous top-1 linking behavior;
    callers can opt into stricter grounding explicitly.
    """
    seed = inputs["entities"].data
    if not seed:
        return {
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[],
                producer="entity.link",
            )
        }

    p = params or {}
    threshold = float(p.get("similarity_threshold", 0.0))
    top_k = max(1, int(p.get("top_k", 1)))

    queries = [record.entity_name for record in seed]
    results = await asyncio.gather(
        *[
            ctx.entities_vdb.retrieval_nodes(
                query,
                top_k=top_k,
                graph=ctx.graph,
                need_score=True,
            )
            for query in queries
        ]
    )

    records = []
    for query, result in zip(queries, results):
        if not result:
            continue

        if isinstance(result, tuple) and len(result) == 2:
            nodes, scores = result
        else:
            nodes, scores = result, None

        if not nodes:
            continue

        linked_node = nodes[0]
        if linked_node is None:
            continue

        score = None
        if scores and scores[0] is not None:
            score = float(scores[0])
            if score < threshold:
                continue
        elif threshold > 0:
            # A requested threshold cannot be enforced without a score.
            continue

        name = linked_node.get(
            ctx.graph.entity_metakey,
            linked_node.get("entity_name", query),
        )
        records.append(
            EntityRecord(
                entity_name=str(name),
                source_id=linked_node.get("source_id", ""),
                entity_type=linked_node.get("entity_type", ""),
                description=linked_node.get("description", ""),
                score=score,
                extra={"linked_from": query},
            )
        )

    return {
        "entities": SlotValue(
            kind=SlotKind.ENTITY_SET,
            data=records,
            producer="entity.link",
        )
    }
