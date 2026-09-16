from types import SimpleNamespace

import pytest

from Core.Methods.kgp import kgp_plan
from Core.Operators.entity.tfidf import entity_tfidf
from Core.Schema.SlotTypes import SlotKind, SlotValue


class FakeGraph:
    async def get_nodes(self):
        return ["alpha", "beta"]

    async def get_node(self, name):
        return {
            "alpha": {
                "entity_type": "concept",
                "description": "alpha crystal technology levitation",
                "source_id": "chunk-a",
            },
            "beta": {
                "entity_type": "person",
                "description": "beta agricultural history",
                "source_id": "chunk-b",
            },
        }[name]


@pytest.mark.asyncio
async def test_entity_tfidf_reports_cosine_similarity_not_candidate_index():
    ctx = SimpleNamespace(
        graph=FakeGraph(),
        config=SimpleNamespace(top_k=2),
    )
    result = await entity_tfidf(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="crystal technology",
                producer="test",
            )
        },
        ctx=ctx,
        params={"top_k": 2},
    )

    records = result["entities"].data
    assert [record.entity_name for record in records] == ["alpha", "beta"]
    assert 0.0 <= records[1].score <= records[0].score <= 1.0
    assert records[0].source_id == "chunk-a"


def test_kgp_second_hop_depends_on_first_hop_selection_and_reasoning():
    plan = kgp_plan("What is crystal technology?", dataset="Demo", depth=2)
    steps = {step.step_id: step for step in plan.steps}

    expand2 = steps["hop_2_expand"].action.tools[0]
    assert expand2.inputs["entities"].from_step_id == "hop_1_rerank"

    reason2 = steps["hop_2_reason"].action.tools[0]
    assert reason2.inputs["query"].from_step_id == "hop_1_reason"
    assert reason2.inputs["chunks"].from_step_id == "hop_2_evidence"
