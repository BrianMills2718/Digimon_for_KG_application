from pathlib import Path
from types import SimpleNamespace

import pytest

from Core.Operators.community.from_level import community_from_level
from Core.Operators.community.resource_state import stale_community_reason
from Core.Schema.SlotTypes import SlotKind, SlotValue


class FakeReports:
    def __init__(self, file_name):
        self._path = str(file_name)

    @property
    def _file_name(self):
        return self._path


class FakeCommunity:
    def __init__(self, file_name):
        self.community_reports = FakeReports(file_name)
        self.community_schema = {
            "0": SimpleNamespace(level=0, occurrence=1.0),
        }


@pytest.mark.asyncio
async def test_missing_persisted_report_marks_live_community_stale(tmp_path):
    missing = tmp_path / "community_report.json"
    community = FakeCommunity(missing)

    reason = stale_community_reason(community)
    assert reason is not None
    assert "rebuild communities" in reason

    ctx = SimpleNamespace(
        community=community,
        config=SimpleNamespace(
            level=2,
            global_max_consider_community=50,
            global_min_community_rating=0.0,
        ),
    )
    result = await community_from_level(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="test",
                producer="test",
            )
        },
        ctx=ctx,
        params={},
    )

    slot = result["communities"]
    assert slot.data == []
    assert slot.metadata["status"] == "stale_resource"


def test_existing_persisted_report_is_not_marked_stale(tmp_path):
    report = tmp_path / "community_report.json"
    report.write_text("{}")

    assert stale_community_reason(FakeCommunity(report)) is None
