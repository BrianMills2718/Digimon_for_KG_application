"""Small validity checks for persisted community resources.

Community objects can remain in MCP process memory after a forced graph rebuild.
The rebuild cleanup removes their persisted report/map files. If those files are
missing, the old in-memory object must not be used as evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def stale_community_reason(community: Any) -> Optional[str]:
    """Return a reason when a community object is known to be stale.

    Test doubles/alternate community implementations that do not expose a
    persisted file path are treated as unverifiable rather than stale.
    """
    if community is None:
        return "community resource unavailable"

    reports = getattr(community, "community_reports", None)
    if reports is None:
        return None

    try:
        file_name = getattr(reports, "_file_name", None)
    except Exception:
        file_name = None

    if not file_name:
        return None

    if not Path(file_name).exists():
        return (
            "community reports were invalidated by a graph rebuild; "
            "rebuild communities before using community evidence"
        )
    return None
