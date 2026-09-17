"""Community-from-level operator.

Retrieve persisted community reports while preserving authoritative community
identity and graph-derived source provenance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Operators.community.resource_state import stale_community_reason
from Core.Schema.SlotTypes import CommunityRecord, SlotKind, SlotValue


def _community_level(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _source_chunk_ids(schema) -> list[str]:
    return sorted(
        str(chunk_id)
        for chunk_id in (getattr(schema, "chunk_ids", []) or [])
        if chunk_id
    )


async def community_from_level(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    p = params or {}
    level = int(p.get("level", getattr(ctx.config, "level", 2)))
    max_consider = int(
        p.get(
            "max_consider",
            getattr(ctx.config, "global_max_consider_community", 50),
        )
    )
    min_rating = float(
        p.get(
            "min_rating",
            getattr(ctx.config, "global_min_community_rating", 0),
        )
    )

    if ctx.community is None:
        return {
            "communities": SlotValue(
                kind=SlotKind.COMMUNITY_SET,
                data=[],
                producer="community.from_level",
                metadata={"error": "community resource unavailable"},
            )
        }

    stale_reason = stale_community_reason(ctx.community)
    if stale_reason:
        return {
            "communities": SlotValue(
                kind=SlotKind.COMMUNITY_SET,
                data=[],
                producer="community.from_level",
                metadata={"error": stale_reason, "status": "stale_resource"},
            )
        }

    community_schema = ctx.community.community_schema or {}
    selected = [
        (community_id, schema)
        for community_id, schema in community_schema.items()
        if _community_level(schema.level) <= level
    ]
    selected.sort(key=lambda item: float(item[1].occurrence or 0.0), reverse=True)
    selected = selected[: max(0, max_consider)]

    if not selected:
        return {
            "communities": SlotValue(
                kind=SlotKind.COMMUNITY_SET,
                data=[],
                producer="community.from_level",
            )
        }

    report_values = await ctx.community.community_reports.get_by_ids(
        [community_id for community_id, _schema in selected]
    )

    records = []
    for (community_id, schema), report_data in zip(selected, report_values):
        if report_data is None:
            continue

        report_json = report_data.get("report_json", {}) or {}
        rating = float(report_json.get("rating", 0.0) or 0.0)
        if rating < min_rating:
            continue

        records.append(
            CommunityRecord(
                community_id=str(community_id),
                level=_community_level(schema.level),
                title=str(
                    report_json.get("title")
                    or getattr(schema, "title", "")
                    or community_id
                ),
                report=str(report_data.get("report_string", "") or ""),
                occurrence=float(getattr(schema, "occurrence", 0.0) or 0.0),
                rating=rating,
                nodes=set(getattr(schema, "nodes", set()) or set()),
                extra={
                    "report_json": report_json,
                    "source_chunk_ids": _source_chunk_ids(schema),
                },
            )
        )

    records.sort(
        key=lambda record: (record.occurrence, record.rating),
        reverse=True,
    )
    return {
        "communities": SlotValue(
            kind=SlotKind.COMMUNITY_SET,
            data=records,
            producer="community.from_level",
        )
    }
