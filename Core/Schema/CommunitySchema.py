from dataclasses import dataclass, asdict, field, Field
from typing import Set, List
from pydantic import Field


class CommunityReportsResult:
    """Community reports result class definition."""
    report_string: str
    report_json: dict

@dataclass
class LeidenInfo:
    level: str = field(default="")
    title: str = field(default="")
    edges: Set[str] = field(default_factory=set)
    nodes: Set[str] = field(default_factory=set)
    chunk_ids: Set[str] = field(default_factory=set)
    occurrence: float = field(default=0.0)
    sub_communities: List[str] = field(default_factory=list)

