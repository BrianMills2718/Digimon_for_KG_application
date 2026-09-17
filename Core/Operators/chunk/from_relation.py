"""Materialize exact source chunks referenced by relationship provenance."""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Utils import split_string_by_multi_markers, truncate_list_by_token_size
from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


def _chunk_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content", value.get("text", "")) or "")
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    if hasattr(value, "text"):
        return str(getattr(value, "text") or "")
    return ""


async def chunk_from_relation(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"relationships": RELATIONSHIP_SET}
    Outputs: {"chunks": CHUNK_SET}

    Relationship source IDs are the only admissible source references. Missing
    or unrecognized stored values are skipped rather than converted to object
    representations or synthetic evidence.
    """
    relationships = inputs["relationships"].data
    if not relationships:
        return {
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=[],
                producer="chunk.from_relation",
            )
        }

    evidence: dict[str, dict[str, Any]] = {}
    for order, relationship in enumerate(relationships):
        source_id = relationship.source_id or relationship.extra.get("source_id", "")
        chunk_ids = split_string_by_multi_markers(source_id, [GRAPH_FIELD_SEP])
        for chunk_id in chunk_ids:
            if not chunk_id:
                continue

            if chunk_id not in evidence:
                raw = await ctx.doc_chunks.get_data_by_key(chunk_id)
                text = _chunk_text(raw).strip()
                if not text:
                    continue
                evidence[chunk_id] = {
                    "text": text,
                    "order": order,
                    "score": relationship.score,
                    "relationships": [
                        (relationship.src_id, relationship.tgt_id)
                    ],
                }
                continue

            current = evidence[chunk_id]
            if relationship.score is not None and (
                current["score"] is None
                or float(relationship.score) > float(current["score"])
            ):
                current["score"] = float(relationship.score)
            pair = (relationship.src_id, relationship.tgt_id)
            if pair not in current["relationships"]:
                current["relationships"].append(pair)

    items = [
        {"id": chunk_id, **value}
        for chunk_id, value in evidence.items()
    ]
    items.sort(
        key=lambda item: (
            item["order"],
            -(float(item["score"]) if item["score"] is not None else 0.0),
        )
    )

    if ctx.config and hasattr(ctx.config, "local_max_token_for_text_unit"):
        items = truncate_list_by_token_size(
            items,
            key=lambda item: item["text"],
            max_token_size=ctx.config.local_max_token_for_text_unit,
        )

    records = [
        ChunkRecord(
            chunk_id=item["id"],
            text=item["text"],
            score=(
                float(item["score"])
                if item["score"] is not None
                else None
            ),
            extra={
                "relationship_order": item["order"],
                "relationships": item["relationships"],
            },
        )
        for item in items
    ]

    return {
        "chunks": SlotValue(
            kind=SlotKind.CHUNK_SET,
            data=records,
            producer="chunk.from_relation",
        )
    }
