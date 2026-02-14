"""Community from level operator.

Retrieve community reports by hierarchy level, sorted by occurrence and rating.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Schema.SlotTypes import CommunityRecord, SlotKind, SlotValue


async def community_from_level(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  (none required — uses config)
    Outputs: {"communities": SlotValue(COMMUNITY_SET)}
    Params:  {"level": int, "max_consider": int, "min_rating": float}
    """
    p = params or {}
    level = p.get("level", getattr(ctx.config, "level", 2))
    max_consider = p.get("max_consider", getattr(ctx.config, "global_max_consider_community", 50))
    min_rating = p.get("min_rating", getattr(ctx.config, "global_min_community_rating", 0))

    community_schema = ctx.community.community_schema
    filtered = {
        k: v for k, v in community_schema.items()
        if v.level <= level
    }
    if not filtered:
        return {"communities": SlotValue(kind=SlotKind.COMMUNITY_SET, data=[], producer="community.from_level")}

    sorted_schemas = sorted(
        filtered.items(),
        key=lambda x: x[1].occurrence,
        reverse=True,
    )[:max_consider]

    community_datas = await ctx.community.community_reports.get_by_ids(
        [k[0] for k in sorted_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c for c in community_datas
        if c.get("report_json", {}).get("rating", 0) >= min_rating
    ]
    community_datas.sort(
        key=lambda x: (
            x.get("community_info", {}).get("occurrence", 0),
            x.get("report_json", {}).get("rating", 0),
        ),
        reverse=True,
    )

    records = []
    for cd in community_datas:
        rj = cd.get("report_json", {})
        records.append(CommunityRecord(
            community_id=str(rj.get("id", "")),
            level=rj.get("level", 0),
            title=rj.get("title", ""),
            report=cd.get("report_string", ""),
            occurrence=cd.get("community_info", {}).get("occurrence", 0.0),
            rating=rj.get("rating", 0.0),
            extra={"report_json": rj},
        ))

    return {"communities": SlotValue(kind=SlotKind.COMMUNITY_SET, data=records, producer="community.from_level")}
