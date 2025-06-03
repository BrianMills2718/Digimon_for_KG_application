"""RAG factories"""

from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Index.IndexConfigFactory import get_index_config

# Import get_index lazily to avoid ColBERT import issues
def __getattr__(name):
    if name == "get_index":
        from Core.Index.IndexFactory import get_index
        return get_index
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["get_rag_embedding", "get_index", "get_index_config"]