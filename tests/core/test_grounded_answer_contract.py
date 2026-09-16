from types import SimpleNamespace

import pytest

from Core.Operators.meta.generate_answer import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    meta_generate_answer,
)
from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


class FakeLLM:
    def __init__(self):
        self.calls = []

    async def aask(self, msg, **kwargs):
        self.calls.append((msg, kwargs))
        return "Crystal technology uses levitite crystals."


@pytest.mark.asyncio
async def test_answer_generation_does_not_call_llm_without_evidence():
    llm = FakeLLM()
    result = await meta_generate_answer(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="What is crystal technology?",
                producer="test",
            ),
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=[],
                producer="test",
            ),
        },
        ctx=SimpleNamespace(llm=llm),
        params={},
    )

    answer = result["answer"]
    assert answer.data == INSUFFICIENT_EVIDENCE_ANSWER
    assert answer.metadata["status"] == "insufficient_evidence"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_answer_prompt_is_grounded_in_retrieved_chunk_text():
    llm = FakeLLM()
    result = await meta_generate_answer(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="What is crystal technology?",
                producer="test",
            ),
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=[
                    ChunkRecord(
                        chunk_id="chunk-a",
                        text="Levitite crystals power Zorathian floating cities.",
                    )
                ],
                producer="test",
            ),
        },
        ctx=SimpleNamespace(llm=llm),
        params={},
    )

    assert result["answer"].metadata["status"] == "grounded_answer"
    assert len(llm.calls) == 1
    prompt = llm.calls[0][0][0]["content"]
    assert "Levitite crystals power Zorathian floating cities." in prompt
    assert "Do not add facts" in prompt
