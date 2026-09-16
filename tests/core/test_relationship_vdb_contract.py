import pytest

from Core.AgentSchema.tool_contracts import RelationshipVDBSearchInputs
from Core.AgentTools.relationship_tools import relationship_vdb_search_tool


class FakeNode:
    def __init__(self, rel_id: str, text: str):
        self.metadata = {"id": rel_id}
        self.text = text
        self.node_id = rel_id


class FakeNodeWithScore:
    def __init__(self, rel_id: str, text: str, score: float):
        self.node = FakeNode(rel_id, text)
        self.score = score


class FakeVDB:
    def __init__(self):
        self.calls = []

    async def retrieval(self, query, top_k):
        self.calls.append((query, top_k))
        return [
            FakeNodeWithScore("alpha->beta", "alpha relates to beta", 0.9),
            FakeNodeWithScore("beta->gamma", "beta relates to gamma", 0.4),
        ]


class FakeContext:
    def __init__(self, vdb):
        self.vdb = vdb

    def get_vdb_instance(self, vdb_id):
        return self.vdb if vdb_id == "demo_relations" else None


@pytest.mark.asyncio
async def test_relationship_vdb_search_uses_retrieval_api_and_threshold():
    vdb = FakeVDB()
    context = FakeContext(vdb)
    params = RelationshipVDBSearchInputs(
        vdb_reference_id="demo_relations",
        query_text="how are alpha and beta related?",
        top_k=5,
        score_threshold=0.5,
    )

    result = await relationship_vdb_search_tool(params, context)

    assert vdb.calls == [("how are alpha and beta related?", 5)]
    assert result.similar_relationships == [
        ("alpha->beta", "alpha relates to beta", 0.9)
    ]
    assert result.metadata["num_results"] == 1


@pytest.mark.asyncio
async def test_relationship_vdb_search_reports_embedding_mode_unsupported():
    context = FakeContext(FakeVDB())
    params = RelationshipVDBSearchInputs(
        vdb_reference_id="demo_relations",
        query_embedding=[0.1, 0.2],
    )

    result = await relationship_vdb_search_tool(params, context)

    assert result.similar_relationships == []
    assert "not implemented" in result.metadata["error"].lower()
