from types import SimpleNamespace

import pytest

from Core.Operators.entity.link import entity_link
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


class FakeVDB:
    def __init__(self, score):
        self.score = score

    async def retrieval_nodes(self, query, top_k, graph, need_score=False):
        assert need_score is True
        return (
            [
                {
                    "entity_name": "Canonical Alpha",
                    "source_id": "chunk-alpha",
                    "entity_type": "concept",
                    "description": "canonical node",
                }
            ],
            [self.score],
        )


@pytest.mark.asyncio
async def test_entity_link_preserves_normalized_similarity_score():
    ctx = SimpleNamespace(
        entities_vdb=FakeVDB(0.82),
        graph=SimpleNamespace(entity_metakey="entity_name"),
    )
    result = await entity_link(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[EntityRecord(entity_name="alpha mention")],
                producer="test",
            )
        },
        ctx=ctx,
        params={},
    )

    linked = result["entities"].data
    assert len(linked) == 1
    assert linked[0].entity_name == "Canonical Alpha"
    assert linked[0].score == pytest.approx(0.82)
    assert linked[0].extra["linked_from"] == "alpha mention"


@pytest.mark.asyncio
async def test_entity_link_can_reject_weak_top1_match():
    ctx = SimpleNamespace(
        entities_vdb=FakeVDB(0.25),
        graph=SimpleNamespace(entity_metakey="entity_name"),
    )
    result = await entity_link(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[EntityRecord(entity_name="weak mention")],
                producer="test",
            )
        },
        ctx=ctx,
        params={"similarity_threshold": 0.5},
    )

    assert result["entities"].data == []
