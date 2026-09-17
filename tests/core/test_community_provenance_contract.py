from types import SimpleNamespace

import pytest

from Core.Operators.community.materialize import community_materialize
from Core.Operators.meta.generate_answer import meta_generate_answer
from Core.Schema.SlotTypes import CommunityRecord, SlotKind, SlotValue


class FakeLLM:
    async def aask(self, msg, **kwargs):
        return "The global pattern is explained by the community [community:42]."


@pytest.mark.asyncio
async def test_community_answer_metadata_traces_summary_to_raw_source_chunks():
    community = CommunityRecord(
        community_id="42",
        level=1,
        title="Crystal network",
        report="A summary of the crystal technology network.",
        occurrence=0.8,
        rating=9.0,
        nodes={"alpha", "beta"},
        extra={"source_chunk_ids": ["chunk-a", "chunk-b"]},
    )

    materialized = await community_materialize(
        inputs={
            "communities": SlotValue(
                kind=SlotKind.COMMUNITY_SET,
                data=[community],
                producer="test",
            )
        },
        ctx=SimpleNamespace(),
        params={},
    )
    chunk = materialized["chunks"].data[0]
    assert chunk.extra["source_chunk_ids"] == ["chunk-a", "chunk-b"]

    answered = await meta_generate_answer(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="What is the global pattern?",
                producer="test",
            ),
            "chunks": materialized["chunks"],
        },
        ctx=SimpleNamespace(llm=FakeLLM()),
        params={},
    )

    metadata = answered["answer"].metadata
    assert metadata["status"] == "grounded_answer"
    assert metadata["cited_evidence_ids"] == ["community:42"]
    assert metadata["evidence_provenance"]["community:42"]["source_chunk_ids"] == [
        "chunk-a",
        "chunk-b",
    ]
