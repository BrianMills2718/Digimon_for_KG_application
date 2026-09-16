from types import SimpleNamespace

import pytest

from Core.Graph.BaseGraph import BaseGraph


class TinyGraph(BaseGraph):
    async def _extract_entity_relationship(self, chunk_key_pair):
        return None

    async def _build_graph(self, chunks):
        return True


class FakeLLM:
    async def aask(self, *args, **kwargs):
        raise AssertionError("short descriptions should not require summarization")


@pytest.mark.asyncio
async def test_short_description_does_not_require_embedding_encode_decode_methods():
    graph = TinyGraph(
        config=SimpleNamespace(summary_max_tokens=500),
        llm=FakeLLM(),
        encoder=object(),  # deliberately has no encode/decode methods
    )

    description = "A short entity description."

    assert await graph._handle_entity_relation_summary("entity", description) == description
