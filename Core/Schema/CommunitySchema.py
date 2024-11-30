
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
class CommunityReportsResult:
    """Community reports result class definition."""
    report_string: str
    report_json: dict


class LeidonInfo(BaseModel):
    level: int = None
    title: str = None
    edges: set = set()
    nodes: set = set()
    chunk_ids: set = set()
    occurrence: float = 0.0
    sub_communities: list = []
