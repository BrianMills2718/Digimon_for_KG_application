"""Community from entity operator.

Find community reports associated with a set of entities via their cluster memberships.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Optional

import asyncio

from Core.Common.Logger import logger
from Core.Common.Utils import truncate_list_by_token_size
from Core.Schema.SlotTypes import CommunityRecord, SlotKind, SlotValue


async def community_from_entity(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"entities": SlotValue(ENTITY_SET)}  -- entities must have clusters populated
    Outputs: {"communities": SlotValue(COMMUNITY_SET)}
    Params:  {"level": int, "max_token": int, "single_one": bool}
    """
    entities = inputs["entities"].data
    p = params or {}
    level = p.get("level", getattr(ctx.config, "level", 2))
    single_one = p.get("single_one", False)
    max_token = p.get("max_token", getattr(ctx.config, "local_max_token_for_community_report", 4096))

    if not entities:
        return {"communities": SlotValue(kind=SlotKind.COMMUNITY_SET, data=[], producer="community.from_entity")}

    related_communities = []
    for ent in entities:
        if not ent.clusters:
            continue
        cluster_data = ent.clusters
        if isinstance(cluster_data, str):
            try:
                cluster_data = json.loads(cluster_data)
            except Exception:
                continue
        if isinstance(cluster_data, list):
            related_communities.extend(cluster_data)

    # Filter by level
    dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp.get("level", 0) <= level
    ]

    if not dup_keys:
        return {"communities": SlotValue(kind=SlotKind.COMMUNITY_SET, data=[], producer="community.from_entity")}

    key_counts = dict(Counter(dup_keys))
    community_reports = ctx.community.community_reports

    raw_data = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in key_counts.keys()]
    )
    community_data = {
        k: v for k, v in zip(key_counts.keys(), raw_data)
        if v is not None
    }

    sorted_keys = sorted(
        key_counts.keys(),
        key=lambda k: (
            key_counts[k],
            community_data[k]["report_json"].get("rating", -1) if k in community_data else -1,
        ),
        reverse=True,
    )

    sorted_data = [community_data[k] for k in sorted_keys if k in community_data]
    sorted_data = truncate_list_by_token_size(
        sorted_data,
        key=lambda x: x["report_string"],
        max_token_size=max_token,
    )
    if single_one:
        sorted_data = sorted_data[:1]

    records = []
    for cd in sorted_data:
        rj = cd.get("report_json", {})
        records.append(CommunityRecord(
            community_id=str(rj.get("id", "")),
            level=rj.get("level", 0),
            title=rj.get("title", ""),
            report=cd.get("report_string", ""),
            rating=rj.get("rating", 0.0),
            extra={"report_json": rj},
        ))

    return {"communities": SlotValue(kind=SlotKind.COMMUNITY_SET, data=records, producer="community.from_entity")}
