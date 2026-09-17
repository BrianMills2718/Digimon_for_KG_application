"""Entity linking operator.

Link entity mentions to exact graph identities first, then VDB approximation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Utils import clean_str
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def _exact_graph_match(ctx: Any, query: str):
    """Resolve an entity mention directly when the graph already contains it."""
    get_node = getattr(ctx.graph, "get_node", None)
    if get_node is None:
        return None, None

    candidates = []
    for candidate in (query, clean_str(query)):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        node = await get_node(candidate)
        if node is not None:
            return candidate, node
    return None, None


async def entity_link(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": ENTITY_SET} -- entity mentions to canonicalize
    Outputs: {"entities": ENTITY_SET} -- linked graph entities
    Params:  {"similarity_threshold": float, "top_k": int}

    Exact graph identity is authoritative and avoids approximating a known entity
    through vector search. Unmatched mentions fall back to VDB retrieval, whose
    normalized higher-is-better score is preserved for downstream reasoning.
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

    records = []
    for source_record in seed:
        query = source_record.entity_name
        exact_name, exact_node = await _exact_graph_match(ctx, query)
        if exact_node is not None:
            name = exact_node.get(
                getattr(ctx.graph, "entity_metakey", "entity_name"),
                exact_node.get("entity_name", exact_name or query),
            )
            records.append(
                EntityRecord(
                    entity_name=str(name),
                    source_id=exact_node.get("source_id", ""),
                    entity_type=exact_node.get("entity_type", ""),
                    description=exact_node.get("description", ""),
                    score=1.0,
                    extra={"linked_from": query, "link_method": "exact_graph"},
                )
            )
            continue

        result = await ctx.entities_vdb.retrieval_nodes(
            query,
            top_k=top_k,
            graph=ctx.graph,
            need_score=True,
        )
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
                extra={"linked_from": query, "link_method": "vdb"},
            )
        )

    return {
        "entities": SlotValue(
            kind=SlotKind.ENTITY_SET,
            data=records,
            producer="entity.link",
        )
    }
