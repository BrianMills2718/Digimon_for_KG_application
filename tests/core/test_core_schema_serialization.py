from dataclasses import asdict

from Core.AgentSchema.tool_contracts import (
    ChunkData,
    ExtractedEntityData,
    RelationshipData,
)
from Core.Schema.ChunkSchema import TextChunk


def test_text_chunk_as_dict_contains_real_fields():
    chunk = TextChunk(
        tokens=3,
        chunk_id="chunk-1",
        content="alpha beta gamma",
        doc_id="doc-1",
        index=0,
        title="Demo",
        relevance_score=0.7,
        metadata={"source": "test"},
    )

    assert chunk.as_dict == {
        "tokens": 3,
        "chunk_id": "chunk-1",
        "content": "alpha beta gamma",
        "doc_id": "doc-1",
        "index": 0,
        "title": "Demo",
        "relevance_score": 0.7,
        "metadata": {"source": "test"},
    }


def test_direct_tool_dataclass_subclasses_serialize_added_metadata():
    chunk = ChunkData(
        tokens=2,
        chunk_id="chunk-2",
        content="source text",
        doc_id="doc-2",
        index=1,
    )
    chunk.relevance_score = 0.9
    chunk.metadata = {"relationship_id": "alpha->beta"}

    entity = ExtractedEntityData(
        entity_name="alpha",
        source_id="agent",
        extraction_confidence=0.8,
    )
    relationship = RelationshipData(
        src_id="alpha",
        tgt_id="beta",
        source_id="chunk-2",
        relevance_score=0.75,
    )

    chunk_data = asdict(chunk)
    entity_data = asdict(entity)
    relationship_data = asdict(relationship)

    assert chunk_data["relevance_score"] == 0.9
    assert chunk_data["metadata"]["relationship_id"] == "alpha->beta"
    assert entity_data["extraction_confidence"] == 0.8
    assert relationship_data["relevance_score"] == 0.75
