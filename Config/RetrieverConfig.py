from typing import Optional
from pydantic import BaseModel, Field

class RetrieverConfig(BaseModel):
  # Retrieval Config
    query_type: str = "ppr"
    enable_local: bool = False
    use_entity_similarity_for_ppr: bool = True
    top_k_entity_for_ppr: int = 8
    node_specificity: bool = True
    damping: float = 0.1
    top_k: int = 5
    node_specificity: bool = True
    damping: float = 0.1