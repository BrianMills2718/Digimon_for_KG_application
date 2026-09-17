from types import SimpleNamespace

import pytest

from Core.Operators.subgraph.materialize import subgraph_materialize
from Core.Schema.SlotTypes import SlotKind, SlotValue, SubgraphRecord


class FakeGraph:
    async def get_node(self, node_id):
        return {"source_id": "chunk-opaque", "description": "Alpha"}

    async def get_edge(self, src, tgt):
        return None


class OpaqueChunks:
    async def get_data_by_key(self, chunk_id):
        return object()


@pytest.mark.asyncio
async def test_subgraph_materialize_does_not_stringify_unknown_objects_as_evidence():
    result = await subgraph_materialize(
        inputs={
            "subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=SubgraphRecord(nodes={"alpha"}, edges=[]),
                producer="test",
            )
        },
        ctx=SimpleNamespace(graph=FakeGraph(), doc_chunks=OpaqueChunks()),
        params={},
    )

    assert result["entities"].data[0].entity_name == "alpha"
    assert result["chunks"].data == []
