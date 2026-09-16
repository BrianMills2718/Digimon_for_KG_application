import sys
from types import SimpleNamespace

import pytest

from Core.Operators.meta.pcst_optimize import (
    _entity_prize,
    _relationship_relevance,
    meta_pcst_optimize,
)
from Core.Schema.SlotTypes import (
    EntityRecord,
    RelationshipRecord,
    SlotKind,
    SlotValue,
)


def test_pcst_helpers_preserve_zero_and_use_retrieval_relevance():
    assert _entity_prize(EntityRecord(entity_name="zero", score=0.0), 1.0) == 0.0
    assert _entity_prize(EntityRecord(entity_name="half", score=0.5), 2.0) == 1.0

    assert _relationship_relevance(
        RelationshipRecord(src_id="a", tgt_id="b", score=0.8, weight=99.0)
    ) == pytest.approx(0.8)
    assert _relationship_relevance(
        RelationshipRecord(src_id="a", tgt_id="b", score=None, weight=2.0)
    ) == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_pcst_fallback_prefers_total_retrieved_prize_not_component_size(monkeypatch):
    # Force the deterministic fallback even if pcst-fast is installed in a dev env.
    monkeypatch.setitem(sys.modules, "pcst_fast", None)

    entities = [
        EntityRecord(entity_name="high", score=0.9),
        EntityRecord(entity_name="low", score=0.1),
    ]
    relationships = [
        # "external" is only a relationship endpoint and must receive zero prize.
        RelationshipRecord(
            src_id="low",
            tgt_id="external",
            score=0.95,
            weight=1.0,
        )
    ]

    result = await meta_pcst_optimize(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=entities,
                producer="test",
            ),
            "relationships": SlotValue(
                kind=SlotKind.RELATIONSHIP_SET,
                data=relationships,
                producer="test",
            ),
        },
        ctx=SimpleNamespace(),
        params={},
    )

    selected = result["subgraph"].data
    assert selected.nodes == {"high"}
    assert selected.edges == []
