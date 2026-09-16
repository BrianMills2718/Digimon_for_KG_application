from types import SimpleNamespace

import pytest

import Core.Index.FaissIndex as faiss_module
from Core.Index.FaissIndex import FaissIndex


class FakeEmbedding:
    dimensions = None
    embed_dim = None

    async def aget_text_embedding_batch(self, texts, show_progress=False):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def __init__(self, faiss_index):
        self.faiss_index = faiss_index


class FakeVectorStoreIndex:
    def __init__(self, nodes, storage_context, embed_model):
        self.nodes = list(nodes)
        self.storage_context = storage_context
        self.embed_model = embed_model

    def insert_nodes(self, nodes):
        self.nodes.extend(nodes)


@pytest.mark.asyncio
async def test_faiss_uses_returned_vector_dimension_when_provider_has_no_metadata(monkeypatch, tmp_path):
    captured = {}

    def fake_hnsw(dimension, neighbors):
        captured["dimension"] = dimension
        captured["neighbors"] = neighbors
        return SimpleNamespace()

    monkeypatch.setattr(faiss_module.faiss, "IndexHNSWFlat", fake_hnsw)
    monkeypatch.setattr(faiss_module, "FaissVectorStore", FakeVectorStore)
    monkeypatch.setattr(
        faiss_module,
        "StorageContext",
        SimpleNamespace(from_defaults=lambda vector_store: SimpleNamespace(vector_store=vector_store)),
    )
    monkeypatch.setattr(faiss_module, "VectorStoreIndex", FakeVectorStoreIndex)
    monkeypatch.setattr(faiss_module, "Settings", SimpleNamespace(embed_model=None))

    config = SimpleNamespace(
        embed_model=FakeEmbedding(),
        persist_path=tmp_path / "faiss",
        retrieve_top_k=5,
    )
    index = FaissIndex(config)

    await index._update_index(
        [{"id": "alpha", "name": "alpha", "content": "alpha description"}],
        ["id", "name"],
    )

    assert captured == {"dimension": 3, "neighbors": 32}
    assert index._index is not None
    assert len(index._index.nodes) == 1
