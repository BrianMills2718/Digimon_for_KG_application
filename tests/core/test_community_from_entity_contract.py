from types import SimpleNamespace

import pytest

from Core.Operators.community.from_entity import community_from_entity
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


class FakeReports:
    async def get_by_id(self, community_id):
        return {
            "report_string": f"Report for {community_id}",
            # No id/level here: those must come from the schema.
            "report_json": {"title": f"Title {community_id}", "rating": 6.0},
        }


@pytest.mark.asyncio
async def test_community_from_entity_preserves_schema_identity_without_llm_ids():
    entity = EntityRecord(
        entity_name="alpha",
        clusters=[{"level": 1, "cluster": "cluster-7"}],
    )
    schema = SimpleNamespace(
        level=1,
        title="Schema title",
        occurrence=0.75,
        nodes={"alpha", "beta"},
    )
    ctx = SimpleNamespace(
        community=SimpleNamespace(
            community_reports=FakeReports(),
            community_schema={"cluster-7": schema},
        ),
        config=SimpleNamespace(
            level=2,
            local_max_token_for_community_report=4096,
        ),
    )

    result = await community_from_entity(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[entity],
                producer="test",
            )
        },
        ctx=ctx,
        params={},
    )

    records = result["communities"].data
    assert len(records) == 1
    assert records[0].community_id == "cluster-7"
    assert records[0].level == 1
    assert records[0].occurrence == pytest.approx(0.75)
    assert records[0].rating == pytest.approx(6.0)
    assert records[0].nodes == {"alpha", "beta"}
