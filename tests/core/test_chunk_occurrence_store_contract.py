from types import SimpleNamespace

import pytest

from Core.Operators.chunk.occurrence import chunk_occurrence
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


class FakeGraph:
    async def get_node_edges(self, entity_name):
        if entity_name == "alpha":
            return [("alpha", "beta")]
        return []

    async def get_node(self, entity_name):
        if entity_name == "beta":
            return {"source_id": "chunk-1"}
        return None


class FakeChunkStore:
    async def get_data_by_key(self, chunk_id):
        if chunk_id != "chunk-1":
            return None
        return TextChunk(
            tokens=4,
            chunk_id="chunk-1",
            content="Alpha and beta are connected by the source evidence.",
            doc_id="doc-1",
            index=0,
            title="Evidence",
        )


@pytest.mark.asyncio
async def test_chunk_occurrence_normalizes_textchunk_store_values_to_exact_text():
    result = await chunk_occurrence(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[
                    EntityRecord(
                        entity_name="alpha",
                        source_id="chunk-1",
                    )
                ],
                producer="test",
            )
        },
        ctx=SimpleNamespace(
            graph=FakeGraph(),
            doc_chunks=FakeChunkStore(),
            config=SimpleNamespace(),
        ),
        params={},
    )

    chunks = result["chunks"].data
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk-1"
    assert chunks[0].text == "Alpha and beta are connected by the source evidence."
    assert chunks[0].extra["relation_counts"] == 1


class UnknownChunkStore:
    async def get_data_by_key(self, chunk_id):
        return object()


@pytest.mark.asyncio
async def test_chunk_occurrence_skips_unrecognized_store_objects_instead_of_stringifying_them():
    result = await chunk_occurrence(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[EntityRecord(entity_name="alpha", source_id="chunk-1")],
                producer="test",
            )
        },
        ctx=SimpleNamespace(
            graph=FakeGraph(),
            doc_chunks=UnknownChunkStore(),
            config=SimpleNamespace(),
        ),
        params={},
    )

    assert result["chunks"].data == []
