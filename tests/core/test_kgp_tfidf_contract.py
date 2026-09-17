from types import SimpleNamespace

import pytest

from Core.Methods.kgp import kgp_plan
from Core.Operators.chunk.merge import chunk_merge
from Core.Operators.entity.tfidf import entity_tfidf
from Core.Schema.SlotTypes import ChunkRecord, EntityRecord, SlotKind, SlotValue


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


@pytest.mark.asyncio
async def test_entity_tfidf_retries_without_stop_words_for_tiny_valid_candidates():
    ctx = SimpleNamespace(config=SimpleNamespace(top_k=2))
    candidates = SlotValue(
        kind=SlotKind.ENTITY_SET,
        data=[
            EntityRecord(entity_name="the", source_id="chunk-the"),
            EntityRecord(entity_name="and", source_id="chunk-and"),
        ],
        producer="test",
    )

    result = await entity_tfidf(
        inputs={
            "query": SlotValue(
                kind=SlotKind.QUERY_TEXT,
                data="the",
                producer="test",
            ),
            "entities": candidates,
        },
        ctx=ctx,
        params={"top_k": 2},
    )

    records = result["entities"].data
    assert len(records) == 2
    assert records[0].entity_name == "the"
    assert records[0].source_id == "chunk-the"
    assert records[0].score > records[1].score


def test_kgp_second_hop_depends_on_first_hop_selection_and_reasoning():
    plan = kgp_plan("What is crystal technology?", dataset="Demo", depth=2)
    steps = {step.step_id: step for step in plan.steps}

    expand2 = steps["hop_2_expand"].action.tools[0]
    assert expand2.inputs["entities"].from_step_id == "hop_1_rerank"

    reason2 = steps["hop_2_reason"].action.tools[0]
    assert reason2.inputs["query"].from_step_id == "hop_1_reason"
    assert reason2.inputs["chunks"].from_step_id == "hop_2_evidence"


def test_kgp_answer_uses_accumulated_intermediate_and_terminal_evidence():
    plan = kgp_plan("What is crystal technology?", dataset="Demo", depth=2)
    steps = {step.step_id: step for step in plan.steps}

    history = steps["hop_2_evidence_history"].action.tools[0]
    assert history.tool_id == "chunk.merge"
    assert history.inputs["left"].from_step_id == "hop_1_evidence"
    assert history.inputs["right"].from_step_id == "hop_2_evidence"

    answer_evidence = steps["answer_evidence"].action.tools[0]
    assert answer_evidence.tool_id == "chunk.merge"
    assert answer_evidence.inputs["left"].from_step_id == "hop_2_evidence_history"
    assert answer_evidence.inputs["right"].from_step_id == "final_evidence"

    answer = steps["answer"].action.tools[0]
    assert answer.inputs["chunks"].from_step_id == "answer_evidence"


@pytest.mark.asyncio
async def test_chunk_merge_deduplicates_by_source_id_and_preserves_best_score():
    left = SlotValue(
        kind=SlotKind.CHUNK_SET,
        data=[
            ChunkRecord(
                chunk_id="chunk-a",
                text="first evidence",
                score=0.4,
                extra={"hop": 1},
            )
        ],
        producer="hop1",
    )
    right = SlotValue(
        kind=SlotKind.CHUNK_SET,
        data=[
            ChunkRecord(
                chunk_id="chunk-a",
                text="first evidence",
                score=0.9,
                extra={"reranked": True},
            ),
            ChunkRecord(
                chunk_id="chunk-b",
                text="second evidence",
                score=0.7,
            ),
        ],
        producer="hop2",
    )

    result = await chunk_merge(
        inputs={"left": left, "right": right},
        ctx=SimpleNamespace(),
        params={},
    )

    chunks = result["chunks"].data
    assert [chunk.chunk_id for chunk in chunks] == ["chunk-a", "chunk-b"]
    assert chunks[0].score == pytest.approx(0.9)
    assert chunks[0].extra == {"hop": 1, "reranked": True}
    assert result["chunks"].metadata["unique_evidence_chunks"] == 2
