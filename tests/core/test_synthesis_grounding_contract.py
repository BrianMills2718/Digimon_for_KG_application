from types import SimpleNamespace

import pytest

from Core.Operators.meta.synthesize_answers import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    meta_synthesize_answers,
)
from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


class NoCallLLM:
    async def aask(self, *args, **kwargs):
        raise AssertionError("LLM should not be called without evidence")


class EchoLLM:
    async def aask(self, *args, **kwargs):
        return "Supported synthesis [chunk-a]"


@pytest.mark.asyncio
async def test_synthesis_does_not_call_llm_without_evidence():
    result = await meta_synthesize_answers(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="What happened?",
                producer="test",
            ),
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=[],
                producer="test",
            ),
        },
        ctx=SimpleNamespace(llm=NoCallLLM()),
    )

    answer = result["answer"]
    assert answer.data == INSUFFICIENT_EVIDENCE_ANSWER
    assert answer.metadata["status"] == "insufficient_evidence"
    assert answer.metadata["evidence_chunk_ids"] == []


@pytest.mark.asyncio
async def test_synthesis_preserves_evidence_ids_in_output_metadata():
    result = await meta_synthesize_answers(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="What happened?",
                producer="test",
            ),
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=[ChunkRecord(chunk_id="chunk-a", text="Grounded evidence")],
                producer="test",
            ),
        },
        ctx=SimpleNamespace(llm=EchoLLM()),
    )

    answer = result["answer"]
    assert answer.metadata["status"] == "synthesized"
    assert answer.metadata["evidence_chunk_ids"] == ["chunk-a"]
