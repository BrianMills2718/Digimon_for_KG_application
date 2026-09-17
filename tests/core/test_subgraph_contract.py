from types import SimpleNamespace

import networkx as nx
import pytest

from Core.Operators.subgraph.khop_paths import subgraph_khop_paths
from Core.Operators.subgraph.materialize import subgraph_materialize
from Core.Operators.subgraph.steiner_tree import subgraph_steiner_tree
from Core.Schema.SlotTypes import (
    EntityRecord,
    SlotKind,
    SlotValue,
    SubgraphRecord,
)


class FakePathGraph:
    async def get_paths_from_sources(self, start_nodes, cutoff=5):
        return [
            [
                {"src_id": "alpha", "tgt_id": "middle"},
                {"src_id": "middle", "tgt_id": "beta"},
            ]
        ]


class FakeConcatenatedPathGraph:
    async def get_paths_from_sources(self, start_nodes, cutoff=5):
        return [
            [
                {"src_id": "alpha", "tgt_id": "x"},
                {"src_id": "x", "tgt_id": "beta"},
                {"src_id": "alpha", "tgt_id": "y"},
                {"src_id": "y", "tgt_id": "gamma"},
            ]
        ]


@pytest.mark.asyncio
async def test_khop_path_mode_parses_storage_edge_records_into_node_paths():
    result = await subgraph_khop_paths(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[EntityRecord(entity_name="alpha")],
                producer="test",
            )
        },
        ctx=SimpleNamespace(graph=FakePathGraph()),
        params={"mode": "paths", "cutoff": 3},
    )

    subgraph = result["subgraph"].data
    assert subgraph.paths == [["alpha", "middle", "beta"]]
    assert set(subgraph.edges) == {("alpha", "middle"), ("middle", "beta")}
    assert subgraph.nodes == {"alpha", "middle", "beta"}


@pytest.mark.asyncio
async def test_khop_path_mode_splits_disconnected_concatenated_paths():
    result = await subgraph_khop_paths(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[EntityRecord(entity_name="alpha")],
                producer="test",
            )
        },
        ctx=SimpleNamespace(graph=FakeConcatenatedPathGraph()),
        params={"mode": "paths", "cutoff": 3},
    )

    subgraph = result["subgraph"].data
    assert subgraph.paths == [
        ["alpha", "x", "beta"],
        ["alpha", "y", "gamma"],
    ]
    assert set(subgraph.edges) == {
        ("alpha", "x"),
        ("x", "beta"),
        ("alpha", "y"),
        ("y", "gamma"),
    }
    assert ("beta", "alpha") not in subgraph.edges


@pytest.mark.asyncio
async def test_steiner_operator_includes_required_intermediate_node():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("alpha", "middle"),
            ("middle", "beta"),
            ("alpha", "detour"),
            ("detour", "other"),
        ]
    )
    ctx = SimpleNamespace(
        graph=SimpleNamespace(_graph=SimpleNamespace(graph=graph))
    )

    result = await subgraph_steiner_tree(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[
                    EntityRecord(entity_name="alpha"),
                    EntityRecord(entity_name="beta"),
                ],
                producer="test",
            )
        },
        ctx=ctx,
        params={},
    )

    subgraph = result["subgraph"].data
    assert subgraph.nodes == {"alpha", "middle", "beta"}
    assert len(subgraph.edges) == 2


@pytest.mark.asyncio
async def test_steiner_operator_uses_best_connected_terminal_group():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("alpha", "middle"),
            ("middle", "beta"),
            ("isolated-high", "isolated-helper"),
        ]
    )
    ctx = SimpleNamespace(
        graph=SimpleNamespace(_graph=SimpleNamespace(graph=graph))
    )

    result = await subgraph_steiner_tree(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[
                    EntityRecord(entity_name="alpha", score=0.6),
                    EntityRecord(entity_name="beta", score=0.5),
                    EntityRecord(entity_name="isolated-high", score=0.99),
                ],
                producer="test",
            )
        },
        ctx=ctx,
        params={},
    )

    slot = result["subgraph"]
    assert slot.data.nodes == {"alpha", "middle", "beta"}
    assert slot.metadata["used_terminals"] == ["alpha", "beta"]
    assert slot.metadata["dropped_disconnected_terminals"] == ["isolated-high"]


class FakeMaterializeGraph:
    async def get_node(self, node_id):
        return {
            "alpha": {"source_id": "chunk-a", "description": "Alpha"},
            "beta": {"source_id": "chunk-b", "description": "Beta"},
        }.get(node_id)

    async def get_edge(self, src, tgt):
        if {src, tgt} == {"alpha", "beta"}:
            return {"source_id": "chunk-edge<SEP>chunk-a"}
        return None


class FakeChunks:
    def __init__(self):
        self.data = {
            "chunk-a": "alpha source",
            "chunk-b": "beta source",
            "chunk-edge": "relationship source",
        }

    async def get_data_by_key(self, chunk_id):
        return self.data.get(chunk_id)


@pytest.mark.asyncio
async def test_subgraph_materialization_returns_entities_and_original_source_chunks():
    ctx = SimpleNamespace(
        graph=FakeMaterializeGraph(),
        doc_chunks=FakeChunks(),
    )
    selected = SubgraphRecord(
        nodes={"alpha", "beta"},
        edges=[("alpha", "beta")],
    )

    result = await subgraph_materialize(
        inputs={
            "subgraph": SlotValue(
                kind=SlotKind.SUBGRAPH,
                data=selected,
                producer="test",
            )
        },
        ctx=ctx,
        params={},
    )

    entities = result["entities"].data
    chunks = result["chunks"].data

    assert {entity.entity_name for entity in entities} == {"alpha", "beta"}
    assert [chunk.chunk_id for chunk in chunks] == [
        "chunk-a",
        "chunk-b",
        "chunk-edge",
    ]
    assert chunks[2].text == "relationship source"
