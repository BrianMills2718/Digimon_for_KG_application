"""Community-from-entity operator.

Resolve entity cluster memberships to persisted community reports while keeping
identity/level/occurrence/source provenance anchored to the Leiden schema.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any, Dict, Optional

from Core.Common.Utils import truncate_list_by_token_size
from Core.Schema.SlotTypes import CommunityRecord, SlotKind, SlotValue


def _level_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _schema_source_chunk_ids(schema) -> list[str]:
    return sorted(
        str(chunk_id)
        for chunk_id in (getattr(schema, "chunk_ids", []) or [])
        if chunk_id
    )


async def community_from_entity(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    entities = inputs["entities"].data
    p = params or {}
    level = int(p.get("level", getattr(ctx.config, "level", 2)))
    single_one = bool(p.get("single_one", False))
    max_token = int(
        p.get(
            "max_token",
            getattr(ctx.config, "local_max_token_for_community_report", 4096),
        )
    )

    if not entities or ctx.community is None:
        return {
            "communities": SlotValue(
                kind=SlotKind.COMMUNITY_SET,
                data=[],
                producer="community.from_entity",
            )
        }

    related_memberships = []
    for entity in entities:
        cluster_data = entity.clusters
        if not cluster_data:
            continue
        if isinstance(cluster_data, str):
            try:
                cluster_data = json.loads(cluster_data)
            except Exception:
                continue
        if isinstance(cluster_data, list):
            related_memberships.extend(
                membership
                for membership in cluster_data
                if isinstance(membership, dict)
            )

    cluster_ids = [
        str(membership["cluster"])
        for membership in related_memberships
        if membership.get("cluster") is not None
        and _level_int(membership.get("level", 0)) <= level
    ]
    if not cluster_ids:
        return {
            "communities": SlotValue(
                kind=SlotKind.COMMUNITY_SET,
                data=[],
                producer="community.from_entity",
            )
        }

    cluster_counts = Counter(cluster_ids)
    ordered_ids = list(cluster_counts.keys())
    raw_reports = await asyncio.gather(
        *[ctx.community.community_reports.get_by_id(key) for key in ordered_ids]
    )
    reports = {
        key: report
        for key, report in zip(ordered_ids, raw_reports)
        if report is not None
    }
    schema_map = ctx.community.community_schema or {}

    ranked_ids = sorted(
        reports,
        key=lambda key: (
            cluster_counts[key],
            float(reports[key].get("report_json", {}).get("rating", 0.0) or 0.0),
            float(getattr(schema_map.get(key), "occurrence", 0.0) or 0.0),
        ),
        reverse=True,
    )

    ranked_pairs = truncate_list_by_token_size(
        [(key, reports[key]) for key in ranked_ids],
        key=lambda pair: pair[1].get("report_string", ""),
        max_token_size=max_token,
    )
    if single_one:
        ranked_pairs = ranked_pairs[:1]

    records = []
    for community_id, report_data in ranked_pairs:
        report_json = report_data.get("report_json", {}) or {}
        schema = schema_map.get(community_id)
        records.append(
            CommunityRecord(
                community_id=str(community_id),
                level=_level_int(getattr(schema, "level", 0)),
                title=str(
                    report_json.get("title")
                    or getattr(schema, "title", "")
                    or community_id
                ),
                report=str(report_data.get("report_string", "") or ""),
                occurrence=float(getattr(schema, "occurrence", 0.0) or 0.0),
                rating=float(report_json.get("rating", 0.0) or 0.0),
                nodes=set(getattr(schema, "nodes", set()) or set()),
                extra={
                    "report_json": report_json,
                    "entity_membership_count": int(cluster_counts[community_id]),
                    "source_chunk_ids": _schema_source_chunk_ids(schema),
                },
            )
        )

    return {
        "communities": SlotValue(
            kind=SlotKind.COMMUNITY_SET,
            data=records,
            producer="community.from_entity",
        )
    }
