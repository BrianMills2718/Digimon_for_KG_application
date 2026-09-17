import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Core.Chunk.ChunkFactory import ChunkFactory


def make_config(tmp_path: Path, chunk_size=8, overlap=2):
    return SimpleNamespace(
        working_dir=str(tmp_path / "results"),
        data_root=str(tmp_path / "Data"),
        chunk=SimpleNamespace(
            chunk_method="chunking_by_token_size",
            chunk_token_size=chunk_size,
            chunk_overlap_token_size=overlap,
        ),
    )


def write_corpus(config, dataset, records):
    path = Path(config.working_dir) / dataset / "corpus" / "Corpus.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle)
            handle.write("\n")
    return path


@pytest.mark.asyncio
async def test_document_records_are_split_using_configured_chunking(tmp_path):
    config = make_config(tmp_path, chunk_size=8, overlap=2)
    write_corpus(
        config,
        "Demo",
        [
            {
                "doc_id": "doc-a",
                "title": "A",
                "content": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen",
            }
        ],
    )

    chunks = await ChunkFactory(config).get_chunks_for_dataset("Demo")

    assert len(chunks) > 1
    ids = [chunk_id for chunk_id, _chunk in chunks]
    assert len(ids) == len(set(ids))
    assert all(chunk.doc_id == "doc-a" for _chunk_id, chunk in chunks)
    assert [chunk.index for _chunk_id, chunk in chunks] == list(range(len(chunks)))
    assert all(
        chunk.metadata["chunk_method"] == "chunking_by_token_size"
        for _chunk_id, chunk in chunks
    )


@pytest.mark.asyncio
async def test_identical_text_in_different_documents_keeps_distinct_source_ids(tmp_path):
    config = make_config(tmp_path, chunk_size=100, overlap=0)
    text = "the same source text appears in two different documents"
    write_corpus(
        config,
        "Demo",
        [
            {"doc_id": "doc-a", "title": "A", "content": text},
            {"doc_id": "doc-b", "title": "B", "content": text},
        ],
    )

    chunks = await ChunkFactory(config).get_chunks_for_dataset("Demo")

    assert len(chunks) == 2
    assert chunks[0][0] != chunks[1][0]
    assert {chunk.doc_id for _chunk_id, chunk in chunks} == {"doc-a", "doc-b"}


@pytest.mark.asyncio
async def test_explicit_chunk_id_is_treated_as_prechunked_for_compatibility(tmp_path):
    config = make_config(tmp_path, chunk_size=4, overlap=1)
    write_corpus(
        config,
        "Demo",
        [
            {
                "doc_id": "doc-a",
                "chunk_id": "existing-chunk-1",
                "content": "this record is intentionally much longer than the configured chunk size",
                "tokens": 42,
            }
        ],
    )

    chunks = await ChunkFactory(config).get_chunks_for_dataset("Demo")

    assert len(chunks) == 1
    chunk_id, chunk = chunks[0]
    assert chunk_id == "existing-chunk-1"
    assert chunk.chunk_id == "existing-chunk-1"
    assert chunk.tokens == 42
