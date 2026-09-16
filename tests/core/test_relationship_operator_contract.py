from types import SimpleNamespace

import pytest

from Core.Operators.relationship.vdb import relationship_vdb
from Core.Schema.SlotTypes import SlotKind, SlotValue


class FakeRelationshipVDB:
    async def retrieval_edges(self, query, top_k, graph, need_score=False):
        assert query == "alpha beta"
        assert top_k == 3
        assert need_score is True
        return (
            [
                {
                    "src_id": "alpha",
                    "tgt_id": "beta",
                    "relation_name": "supports",
                    "description": "alpha supports beta",
                    "weight": 1.0,
                    "source_id": "chunk-1",
                },
                {
                    "src_id": "beta",
                    "tgt_id": "gamma",
                    "relation_name": "causes",
                    "description": "beta causes gamma",
                    "weight": 2.0,
                    "source_id": "chunk-2",
                },
            ],
            [0.35, 0.9],
        )


class FakeGraph:
    async def edge_degree(self, src, tgt):
        return {("alpha", "beta"): 5, ("beta", "gamma"): 2}[(src, tgt)]


@pytest.mark.asyncio
async def test_relationship_vdb_unpacks_edges_and_scores():
    ctx = SimpleNamespace(
        relations_vdb=FakeRelationshipVDB(),
        graph=FakeGraph(),
        config=SimpleNamespace(top_k=3),
    )
    inputs = {
        "query": SlotValue(kind=SlotKind.QUERY_TEXT, data="alpha beta", producer="test")
    }

    result = await relationship_vdb(inputs, ctx)

    relationships = result["relationships"]
    assert relationships.kind == SlotKind.RELATIONSHIP_SET
    assert [record.relation_name for record in relationships.data] == ["causes", "supports"]
    assert [record.score for record in relationships.data] == [0.9, 0.35]
    assert relationships.data[1].source_id == "chunk-1"
