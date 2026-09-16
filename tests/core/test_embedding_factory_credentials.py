from types import SimpleNamespace

import Core.Index.EmbeddingFactory as embedding_factory


def test_openai_embedding_omits_empty_api_key(monkeypatch):
    captured = {}

    class FakeOpenAIEmbedding:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(embedding_factory, "OpenAIEmbedding", FakeOpenAIEmbedding)

    config = SimpleNamespace(
        embedding=SimpleNamespace(
            api_key="",
            base_url=None,
            model="text-embedding-3-small",
            embed_batch_size=16,
            dimensions=1536,
        ),
        llm=SimpleNamespace(api_key="", base_url="https://api.openai.com/v1"),
    )

    embedding_factory.RAGEmbeddingFactory()._create_openai(config)

    assert "api_key" not in captured
    assert captured["api_base"] == "https://api.openai.com/v1"
    assert captured["model_name"] == "text-embedding-3-small"
    assert captured["dimensions"] == 1536
