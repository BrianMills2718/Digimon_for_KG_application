import copy
from types import SimpleNamespace

import pytest

import Core.AgentTools.graph_construction_tools as tools
from Core.AgentSchema.graph_construction_tool_contracts import (
    BuildPassageGraphInputs,
    BuildTreeGraphInputs,
)


class GraphConfig:
    type = "er_graph"

    def model_copy(self, deep=False):
        return copy.deepcopy(self)


class Config:
    def __init__(self):
        self.graph = GraphConfig()
        self.working_dir = "./results"

    def model_copy(self, deep=False):
        return copy.deepcopy(self)


class Graph:
    def __init__(self):
        self._graph = SimpleNamespace(namespace=None)
        self.node_num = 2
        self.edge_num = 1
        self.build_calls = []

    async def build_graph(self, chunks, force=False):
        self.build_calls.append(force)
        return True


class Chunks:
    def __init__(self):
        self.types = []

    def get_namespace(self, dataset_name, graph_type="er_graph"):
        self.types.append(graph_type)
        return SimpleNamespace(path=f"/tmp/{dataset_name}/{graph_type}")

    async def get_chunks_for_dataset(self, dataset_name):
        return [("chunk-a", object()), ("chunk-b", object())]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("builder", "inputs", "expected_type"),
    [
        (tools.build_tree_graph, BuildTreeGraphInputs(target_dataset_name="Demo"), "tree_graph"),
        (tools.build_passage_graph, BuildPassageGraphInputs(target_dataset_name="Demo"), "passage_graph"),
    ],
)
async def test_changed_manifest_rebuilds_non_er_graph_and_invalidates_only_generic_derivatives(
    monkeypatch,
    builder,
    inputs,
    expected_type,
):
    graph = Graph()
    chunks = Chunks()
    monkeypatch.setattr(tools, "get_graph", lambda **kwargs: graph)
    monkeypatch.setattr(tools, "manifest_matches", lambda graph, chunk_pairs: False)
    monkeypatch.setattr(tools, "write_manifest", lambda graph, chunk_pairs: True)
    invalidations = []

    def capture(config, dataset_name, *, invalidate_sparse_matrices):
        invalidations.append((dataset_name, invalidate_sparse_matrices))
        return []

    monkeypatch.setattr(tools, "invalidate_after_forced_graph_rebuild", capture)

    result = await builder(
        inputs,
        Config(),
        llm_instance=object(),
        encoder_instance=object(),
        chunk_factory=chunks,
    )

    assert result.status == "success"
    assert graph.build_calls == [True]
    assert chunks.types == [expected_type]
    assert invalidations == [("Demo", False)]
    assert "source chunks changed" in result.message
