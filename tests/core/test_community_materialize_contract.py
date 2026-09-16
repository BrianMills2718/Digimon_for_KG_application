from types import SimpleNamespace

import pytest

from Core.Methods.basic_global import basic_global_plan
from Core.Operators.community.materialize import community_materialize
from Core.Schema.SlotTypes import CommunityRecord, SlotKind, SlotValue


@pytest.mark.asyncio
async def test_community_reports_materialize_as_answer_context():
    community = CommunityRecord(
        community_id="42",
        level=1,
        title="Crystal technology",
        report="The community report explains the crystal technology network.",
        occurrence=0.8,
        rating=9.0,
    )

    result = await community_materialize(
        inputs={
            "communities": SlotValue(
                kind=SlotKind.COMMUNITY_SET,
                data=[community],
                producer="test",
            )
        },
        ctx=SimpleNamespace(),
        params={},
    )

    chunks = result["chunks"].data
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "community:42"
    assert "crystal technology" in chunks[0].text
    assert chunks[0].score == pytest.approx(9.0)


def test_basic_global_plan_ends_in_answer_generation():
    plan = basic_global_plan("What are the major themes?", dataset="Demo")
    assert plan.steps[-1].action.tools[-1].tool_id == "meta.generate_answer"
