from types import SimpleNamespace

import pytest

from Core.Operators.meta.decompose_question import meta_decompose_question
from Core.Schema.SlotTypes import SlotKind, SlotValue


class FakeLLM:
    def __init__(self, response):
        self.response = response

    async def aask(self, **kwargs):
        return self.response


def query_input(text):
    return {
        "query": SlotValue(
            kind=SlotKind.QUERY_TEXT,
            data=text,
            producer="test",
        )
    }


@pytest.mark.asyncio
async def test_dependency_aware_json_is_preserved_as_advisory_subquestions():
    ctx = SimpleNamespace(
        llm=FakeLLM(
            '["q1: identify performer", "q2: find government positions of <q1.entity>"]'
        )
    )

    result = await meta_decompose_question(
        query_input("What government position was held by the performer?"),
        ctx,
        {"max_questions": 5},
    )

    records = result["sub_questions"].data
    assert [record.entity_name for record in records] == [
        "q1: identify performer",
        "q2: find government positions of <q1.entity>",
    ]
    assert result["sub_questions"].metadata["advisory"] is True


@pytest.mark.asyncio
async def test_malformed_planner_prose_falls_back_to_original_question():
    original = "Who held the office and why?"
    ctx = SimpleNamespace(
        llm=FakeLLM(
            "I would first investigate the person, then maybe search their career."
        )
    )

    result = await meta_decompose_question(
        query_input(original),
        ctx,
        {},
    )

    records = result["sub_questions"].data
    assert len(records) == 1
    assert records[0].entity_name == original
    assert records[0].extra["advisory"] is True
