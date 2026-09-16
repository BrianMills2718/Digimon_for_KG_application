from types import SimpleNamespace

import networkx as nx
import pytest

from Core.Operators.subgraph.khop_paths import subgraph_khop_paths
from Core.Operators.subgraph.steiner_tree import subgraph_steiner_tree
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


class FakePathGraph:
    async def get_paths_from_sources(self, start_nodes, cutoff=5):
        return [
            [
                {"src_id": "alpha", "tgt_id": "middle"},
                {"src_id": "middle", "tgt_id": "beta"},
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
