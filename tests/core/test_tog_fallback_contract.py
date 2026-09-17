from types import SimpleNamespace

import pytest

from Core.Operators.relationship.agent import relationship_agent
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


class PunctuatedGraph:
    async def get_node_edges(self, source_node_id):
        return [("alpha", "beta")]

    async def get_edge(self, src, tgt):
        if {src, tgt} == {"alpha", "beta"}:
            return {
                "relation_name": "relationship",
                "description": "alpha powers beta (primary); strongly",
                "source_id": "chunk-alpha-beta",
                "weight": 0.6,
            }
        return None


class UnparseableLLM:
    async def aask(self, msg, **kwargs):
        return "The first relation looks relevant, but I did not follow the requested format."


@pytest.mark.asyncio
async def test_tog_relation_selection_falls_back_without_dropping_evidence():
    result = await relationship_agent(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="How is alpha connected?",
                producer="test",
            ),
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[EntityRecord(entity_name="alpha")],
                producer="test",
            ),
        },
        ctx=SimpleNamespace(graph=PunctuatedGraph(), llm=UnparseableLLM()),
        params={"width": 1},
    )

    records = result["relationships"].data
    assert len(records) == 1
    assert records[0].relation_name == "alpha powers beta [primary], strongly"
    assert records[0].source_id == "chunk-alpha-beta"
    assert records[0].score == pytest.approx(0.6)
    assert records[0].extra["selection_fallback"] is True
