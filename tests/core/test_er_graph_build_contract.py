from types import SimpleNamespace

import pytest

from Core.Graph.ERGraph import ERGraph
from Core.Schema.EntityRelation import Entity
from Core.Storage.NetworkXStorage import NetworkXStorage


class FakeTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8")


class EmptyExtractionERGraph(ERGraph):
    async def _extract_entity_relationship(self, chunk_key_pair):
        return {}, {}


class OneNodeERGraph(ERGraph):
    async def _extract_entity_relationship(self, chunk_key_pair):
        chunk_id, _ = chunk_key_pair
        entity = Entity(
            entity_name="alpha",
            source_id=chunk_id,
            entity_type="concept",
            description="",
        )
        return {"alpha": [entity]}, {}


def make_config():
    return SimpleNamespace(
        extract_two_step=False,
        auto_generate_ontology=False,
        enable_entity_description=True,
        enable_entity_type=True,
        enable_edge_description=True,
        enable_edge_keywords=False,
        enable_edge_name=True,
    )


@pytest.mark.asyncio
async def test_er_build_rejects_nonempty_corpus_that_extracts_zero_nodes():
    graph = EmptyExtractionERGraph(
        config=make_config(),
        llm=SimpleNamespace(model="test-model"),
        encoder=FakeTokenizer(),
        storage_instance=NetworkXStorage(),
    )

    ok = await graph._build_graph([("chunk-1", object())])

    assert ok is False
    assert graph.node_num == 0


@pytest.mark.asyncio
async def test_er_build_accepts_real_extracted_node():
    graph = OneNodeERGraph(
        config=make_config(),
        llm=SimpleNamespace(model="test-model"),
        encoder=FakeTokenizer(),
        storage_instance=NetworkXStorage(),
    )

    ok = await graph._build_graph([("chunk-1", object())])

    assert ok is True
    assert graph.node_num == 1
    assert await graph.get_node("alpha") is not None
