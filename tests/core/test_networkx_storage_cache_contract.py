import pytest

from Core.Storage.NetworkXStorage import NetworkXStorage


@pytest.mark.asyncio
async def test_node_and_edge_index_caches_are_invalidated_after_mutation():
    storage = NetworkXStorage()

    await storage.upsert_node("a", {"entity_name": "a"})
    await storage.upsert_node("b", {"entity_name": "b"})
    await storage.upsert_edge("a", "b", {"src_id": "a", "tgt_id": "b"})

    assert await storage.get_node_index("b") == 1
    assert storage.get_edge_index("a", "b") == 0

    # Populate the caches, then mutate topology.
    await storage.upsert_node("c", {"entity_name": "c"})
    await storage.upsert_edge("b", "c", {"src_id": "b", "tgt_id": "c"})

    assert await storage.get_node_index("c") == 2
    assert storage.get_edge_index("b", "c") == 1


@pytest.mark.asyncio
async def test_clear_invalidates_cached_indices():
    storage = NetworkXStorage()
    await storage.upsert_node("a", {"entity_name": "a"})
    await storage.get_node_index("a")

    storage.clear()
    await storage.upsert_node("fresh", {"entity_name": "fresh"})

    assert await storage.get_node_index("fresh") == 0
    assert storage.node_list == ["fresh"]


@pytest.mark.asyncio
async def test_community_schema_without_source_ids_does_not_divide_by_zero():
    storage = NetworkXStorage()
    await storage.upsert_node(
        "solo",
        {
            "entity_name": "solo",
            "source_id": "",
            "clusters": '[{"level": 0, "cluster": "0"}]',
        },
    )

    schema = await storage.get_community_schema()

    assert "0" in schema
    assert schema["0"].nodes == ["solo"]
    assert schema["0"].chunk_ids == []
    assert schema["0"].occurrence == 0.0
