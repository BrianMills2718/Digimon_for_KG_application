import asyncio
from Core.Operators import OperatorContext
from Core.Operators.subgraph.materialize import subgraph_materialize
from Core.Schema.SlotTypes import SlotKind,SlotValue,SubgraphRecord
from Core.Storage.NetworkXStorage import NetworkXStorage


class PassageFixture:
    async def get_data_by_key(self, key):
        return {"text": "  Original evidence.\n", "source_ref": "source:1", "namespace_id": "demo"}


def test_materialize_preserves_original_text_and_source_scope():
    async def run():
        store=NetworkXStorage()
        await store.upsert_node("entity:a",{"source_id":"passage:1"})
        result=await subgraph_materialize({"subgraph":SlotValue(SlotKind.SUBGRAPH,SubgraphRecord({"entity:a"},[]))},OperatorContext(store,doc_chunks=PassageFixture()))
        return result["chunks"].data[0]
    chunk=asyncio.run(run())
    assert chunk.text == "  Original evidence.\n"
    assert chunk.extra["source_ref"] == "source:1"
    assert chunk.extra["namespace_id"] == "demo"
