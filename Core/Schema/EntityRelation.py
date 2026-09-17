from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Entity:
    entity_name: str
    source_id: str
    entity_type: str = field(default="")
    description: str = field(default="")
    attributes: Dict[str, Any] = field(default_factory=dict)
    # Optional extraction metadata used by direct agent tools. Keeping it on the
    # canonical dataclass makes it survive dataclass/Pydantic serialization.
    extraction_confidence: Optional[float] = None

    @property
    def as_dict(self):
        return asdict(self)


@dataclass
class Relationship:
    """Canonical graph relationship record."""

    src_id: str
    tgt_id: str
    source_id: str
    relation_name: str = field(default="")
    weight: float = field(default=0.0)
    description: str = field(default="")
    keywords: str = field(default="")
    rank: int = field(default=0)
    attributes: Dict[str, Any] = field(default_factory=dict)
    # Optional retrieval score used by direct relationship tool outputs.
    relevance_score: Optional[float] = None

    @property
    def as_dict(self):
        return asdict(self)
