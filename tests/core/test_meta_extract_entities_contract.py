from types import SimpleNamespace

import pytest

from Core.Operators.meta.extract_entities import (
    meta_extract_entities,
    parse_entity_names,
)
from Core.Schema.SlotTypes import SlotKind, SlotValue


def test_entity_parser_accepts_fenced_json_list():
    assert parse_entity_names('```json\n["Alpha", "Beta"]\n```') == [
        "Alpha",
        "Beta",
    ]


def test_entity_parser_accepts_common_object_wrapper():
    assert parse_entity_names(
        {"entities": [{"entity_name": "Alpha"}, {"name": "Beta"}]}
    ) == ["Alpha", "Beta"]


def test_entity_parser_does_not_turn_arbitrary_prose_into_comma_entities():
    assert parse_entity_names("I could not identify any entities, sorry.") == []


class FakeLLM:
    async def aask(self, msg, **kwargs):
        return 'Here you go: {"entities": ["Alpha", "Beta"]}'


@pytest.mark.asyncio
async def test_meta_extract_entities_handles_explanatory_text_around_json():
    result = await meta_extract_entities(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="How is Alpha related to Beta?",
                producer="test",
            )
        },
        ctx=SimpleNamespace(llm=FakeLLM()),
        params={},
    )

    assert [
        record.entity_name for record in result["entities"].data
    ] == ["Alpha", "Beta"]
