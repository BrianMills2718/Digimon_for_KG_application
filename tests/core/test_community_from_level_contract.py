from types import SimpleNamespace

import pytest

from Core.Operators.community.from_level import community_from_level


class FakeReports:
    async def get_by_ids(self, ids):
        assert ids == ["community-a", "community-b"]
        return [
            {
                "report_string": "Report A",
                "report_json": {"title": "Alpha report", "rating": 7.0},
                # Persisted Leiden fields are flat, not nested in community_info.
                "occurrence": 0.9,
            },
            {
                "report_string": "Report B",
                "report_json": {"title": "Beta report", "rating": 5.0},
                "occurrence": 0.4,
            },
        ]


@pytest.mark.asyncio
async def test_community_from_level_uses_schema_identity_and_occurrence():
    schemas = {
        "community-a": SimpleNamespace(
            level=1,
            title="Schema A",
            occurrence=0.9,
            nodes={"alpha", "beta"},
        ),
        "community-b": SimpleNamespace(
            level=2,
            title="Schema B",
            occurrence=0.4,
            nodes={"gamma"},
        ),
    }
    ctx = SimpleNamespace(
        community=SimpleNamespace(
            community_schema=schemas,
            community_reports=FakeReports(),
        ),
        config=SimpleNamespace(
            level=2,
            global_max_consider_community=50,
            global_min_community_rating=0.0,
        ),
    )

    result = await community_from_level({}, ctx, {})
    communities = result["communities"].data

    assert [c.community_id for c in communities] == ["community-a", "community-b"]
    assert communities[0].level == 1
    assert communities[0].occurrence == pytest.approx(0.9)
    assert communities[0].rating == pytest.approx(7.0)
    assert communities[0].nodes == {"alpha", "beta"}
    assert communities[0].report == "Report A"
