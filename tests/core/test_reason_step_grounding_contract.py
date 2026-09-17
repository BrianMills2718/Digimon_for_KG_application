from types import SimpleNamespace

import pytest

from Core.Operators.meta.reason_step import meta_reason_step
from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


class FailIfCalledLLM:
    async def aask(self, *args, **kwargs):
        raise AssertionError("LLM must not be called without retrieved evidence")


class RefiningLLM:
    async def aask(self, *args, **kwargs):
        return "Which organization created the device?"


@pytest.mark.asyncio
async def test_reason_step_preserves_query_and_skips_llm_without_evidence():
    result = await meta_reason_step(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="Who made the device?",
                producer="test",
            ),
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=[],
                producer="test",
            ),
        },
        ctx=SimpleNamespace(llm=FailIfCalledLLM()),
        params={"mode": "refine"},
    )

    output = result["query"]
    assert output.data == "Who made the device?"
    assert output.metadata["status"] == "unchanged_no_evidence"


@pytest.mark.asyncio
async def test_reason_step_can_refine_from_retrieved_evidence():
    result = await meta_reason_step(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="Who made the device?",
                producer="test",
            ),
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=[
                    ChunkRecord(
                        chunk_id="chunk-1",
                        text="The device was developed by a laboratory, but the organization is not named here.",
                    )
                ],
                producer="test",
            ),
        },
        ctx=SimpleNamespace(llm=RefiningLLM()),
        params={"mode": "refine"},
    )

    output = result["query"]
    assert output.data == "Which organization created the device?"
    assert output.metadata["status"] == "refined_from_evidence"
    assert output.metadata["evidence_chunks"] == 1
