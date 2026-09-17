from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TextChunk:
    """Canonical text-chunk record used across corpus, retrieval, and MCP tools."""

    tokens: int
    chunk_id: str
    content: str
    doc_id: str
    index: int
    title: Optional[str] = None
    # Retrieval/tool metadata is optional for corpus chunks but making it part of
    # the canonical dataclass means direct MCP ChunkData subclasses serialize it
    # instead of attaching invisible post-hoc attributes.
    relevance_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def as_dict(self):
        return asdict(self)
