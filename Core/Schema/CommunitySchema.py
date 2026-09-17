from dataclasses import asdict, dataclass, field
from typing import List, Set


class CommunityReportsResult:
    """Community reports result class definition."""

    report_string: str
    report_json: dict


@dataclass
class LeidenInfo:
    community_id: str = field(default="")
    level: str = field(default="")
    title: str = field(default="")
    edges: Set[str] = field(default_factory=set)
    nodes: Set[str] = field(default_factory=set)
    chunk_ids: Set[str] = field(default_factory=set)
    occurrence: float = field(default=0.0)
    sub_communities: List[str] = field(default_factory=list)

    @property
    def as_dict(self):
        """Return a deterministic JSON-safe representation.

        Community reports are persisted through ``JsonKVStorage``/``json.dump``.
        The in-memory schema intentionally uses sets for deduplication, but raw
        ``dataclasses.asdict`` preserves those sets and makes persistence fail.
        Convert set-valued fields at this serialization boundary instead of
        weakening the in-memory representation.
        """
        data = asdict(self)
        data["nodes"] = sorted(self.nodes)
        data["chunk_ids"] = sorted(self.chunk_ids)
        data["edges"] = sorted(
            [list(edge) if isinstance(edge, tuple) else edge for edge in self.edges],
            key=lambda value: repr(value),
        )
        data["sub_communities"] = list(self.sub_communities)
        return data
