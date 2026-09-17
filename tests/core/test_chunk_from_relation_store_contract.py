from types import SimpleNamespace

import pytest

from Core.Operators.chunk.from_relation import chunk_from_relation
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.SlotTypes import RelationshipRecord, SlotKind, SlotValue


class FakeChunkStore:
    async def get_data_by_key(self, chunk_id):
        if chunk_id != "chunk-1":
            return None
        return TextChunk(
            tokens=5,
            chunk_id="chunk-1",
            content="Exact relationship evidence from the corpus.",
            doc_id="doc-1",
            index=0,
        )


@pytest.mark.asyncio
async def test_chunk_from_relation_materializes_textchunk_and_preserves_best_relationship_score():
    relationships = [
        RelationshipRecord(
            src_id="alpha",
            tgt_id="beta",
            source_id="chunk-1",
            relation_name="related_to",
            score=0.4,
        ),
        RelationshipRecord(
            src_id="gamma",
            tgt_id="beta",
            source_id="chunk-1",
            relation_name="supports",
            score=0.9,
        ),
    ]

    result = await chunk_from_relation(
        inputs={
            "relationships": SlotValue(
                kind=SlotKind.RELATIONSHIP_SET,
                data=relationships,
                producer="test",
            )
        },
        ctx=SimpleNamespace(
            doc_chunks=FakeChunkStore(),
            config=SimpleNamespace(),
        ),
        params={},
    )

    chunks = result["chunks"].data
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk-1"
    assert chunks[0].text == "Exact relationship evidence from the corpus."
    assert chunks[0].score == pytest.approx(0.9)
    assert chunks[0].extra["relationships"] == [
        ("alpha", "beta"),
        ("gamma", "beta"),
    ]


class UnknownStore:
    async def get_data_by_key(self, chunk_id):
        return object()


@pytest.mark.asyncio
async def test_chunk_from_relation_skips_unknown_store_objects():
    result = await chunk_from_relation(
        inputs={
            "relationships": SlotValue(
                kind=SlotKind.RELATIONSHIP_SET,
                data=[
                    RelationshipRecord(
                        src_id="alpha",
                        tgt_id="beta",
                        source_id="chunk-1",
                    )
                ],
                producer="test",
            )
        },
        ctx=SimpleNamespace(doc_chunks=UnknownStore(), config=SimpleNamespace()),
        params={},
    )

    assert result["chunks"].data == []
