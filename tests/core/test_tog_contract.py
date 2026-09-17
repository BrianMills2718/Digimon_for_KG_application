from types import SimpleNamespace

import pytest

from Core.Methods.tog import tog_plan
from Core.Operators.entity.agent import entity_agent
from Core.Operators.entity.rel_node import entity_rel_node
from Core.Operators.relationship.agent import relationship_agent
from Core.Schema.SlotTypes import (
    EntityRecord,
    RelationshipRecord,
    SlotKind,
    SlotValue,
)


def test_tog_plan_wires_each_hop_from_previous_selected_entities():
    plan = tog_plan("How is alpha connected?", dataset="Demo", depth=3)
    steps = {step.step_id: step for step in plan.steps}

    for hop in range(2, 4):
        tool = steps[f"hop_{hop}_relationships"].action.tools[0]
        source = tool.inputs["entities"]
        assert source.from_step_id == f"hop_{hop - 1}_entities"
        assert source.named_output_key == "entities"

    relationship_steps = [
        step for step in plan.steps if step.step_id.endswith("_relationships")
    ]
    assert len(relationship_steps) == 3


@pytest.mark.asyncio
async def test_tog_relation_adapter_preserves_candidate_mapping_for_entity_agent():
    relationship = RelationshipRecord(
        src_id="alpha",
        tgt_id="",
        relation_name="related_to",
        score=0.9,
        extra={
            "head": True,
            "relations_dict": {("alpha", "related_to"): ["beta"]},
        },
    )
    context = SimpleNamespace(
        graph=SimpleNamespace(),
        config=SimpleNamespace(),
        llm=SimpleNamespace(),
    )

    adapted = await entity_rel_node(
        inputs={
            "relationships": SlotValue(
                kind=SlotKind.RELATIONSHIP_SET,
                data=[relationship],
                producer="test",
            )
        },
        ctx=context,
        params={},
    )
    candidates = adapted["entities"].data

    assert len(candidates) == 1
    assert candidates[0].entity_name == "alpha"
    assert candidates[0].extra["relation"] == "related_to"

    selected = await entity_agent(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="How is alpha connected?",
                producer="test",
            ),
            "entity_relation_list": adapted["entities"],
        },
        ctx=context,
        params={"width": 3},
    )

    records = selected["entities"].data
    assert len(records) == 1
    assert records[0].entity_name == "beta"
    assert records[0].score == pytest.approx(0.9)


class BeamGraph:
    def __init__(self):
        self.edges = {
            ("alpha", "beta"): {
                "relation_name": "relationship",
                "description": "alpha powers beta",
            },
            ("gamma", "delta"): {
                "relation_name": "relationship",
                "description": "gamma regulates delta",
            },
        }

    async def get_node_edges(self, source_node_id):
        return [
            edge
            for edge in self.edges
            if source_node_id in edge
        ]

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))


class BeamLLM:
    async def aask(self, msg, **kwargs):
        prompt = msg[-1]["content"]
        if "Topic Entity: alpha" in prompt:
            return "{alpha powers beta (Score: 0.9)}"
        if "Topic Entity: gamma" in prompt:
            return "{gamma regulates delta (Score: 0.8)}"
        return ""


@pytest.mark.asyncio
async def test_relationship_agent_explores_full_beam_and_uses_description_for_generic_edges():
    result = await relationship_agent(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="What are the important connections?",
                producer="test",
            ),
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[
                    EntityRecord(entity_name="alpha"),
                    EntityRecord(entity_name="gamma"),
                ],
                producer="test",
            ),
        },
        ctx=SimpleNamespace(graph=BeamGraph(), llm=BeamLLM()),
        params={"width": 2},
    )

    records = result["relationships"].data
    assert [record.src_id for record in records] == ["alpha", "gamma"]
    assert [record.relation_name for record in records] == [
        "alpha powers beta",
        "gamma regulates delta",
    ]
    assert records[0].extra["relations_dict"][("alpha", "alpha powers beta")] == ["beta"]
    assert records[1].extra["relations_dict"][("gamma", "gamma regulates delta")] == ["delta"]
