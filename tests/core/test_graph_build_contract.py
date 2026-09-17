import copy
from types import SimpleNamespace

import pytest

import Core.AgentTools.graph_construction_tools as tools
from Core.AgentSchema.graph_construction_tool_contracts import (
    BuildERGraphInputs,
    BuildPassageGraphInputs,
    BuildRKGraphInputs,
)


class FakeGraphConfig:
    def __init__(self):
        self.type = "er_graph"

    def model_copy(self, deep=False):
        return copy.deepcopy(self)


class FakeMainConfig:
    def __init__(self):
        self.graph = FakeGraphConfig()
        self.working_dir = "./results"

    def model_copy(self, deep=False):
        return copy.deepcopy(self)


class FakeGraph:
    def __init__(self, succeeds=True, node_num=3, edge_num=2):
        self._graph = SimpleNamespace(namespace=None)
        self.node_num = node_num
        self.edge_num = edge_num
        self.succeeds = succeeds
        self.build_calls = []

    async def build_graph(self, chunks, force=False):
        self.build_calls.append(force)
        return self.succeeds


class FakeChunkFactory:
    def __init__(self):
        self.namespace_types = []

    def get_namespace(self, dataset_name, graph_type="er_graph"):
        self.namespace_types.append(graph_type)
        return SimpleNamespace(
            path=f"/tmp/{dataset_name}/{graph_type}",
            get_save_path=lambda suffix=None: (
                f"/tmp/{dataset_name}/{graph_type}/{suffix}"
                if suffix
                else f"/tmp/{dataset_name}/{graph_type}"
            ),
        )

    async def get_chunks_for_dataset(self, dataset_name):
        return [("chunk-1", object())]


def stable_manifest(monkeypatch, value=True):
    monkeypatch.setattr(tools, "manifest_matches", lambda graph, chunks: value)
    monkeypatch.setattr(tools, "write_manifest", lambda graph, chunks: True)


@pytest.mark.asyncio
async def test_rk_build_does_not_report_success_when_graph_build_fails(monkeypatch):
    graph = FakeGraph(succeeds=False)
    monkeypatch.setattr(tools, "get_graph", lambda **kwargs: graph)
    stable_manifest(monkeypatch, True)

    result = await tools.build_rk_graph(
        BuildRKGraphInputs(target_dataset_name="Demo"),
        FakeMainConfig(),
        llm_instance=object(),
        encoder_instance=object(),
        chunk_factory=FakeChunkFactory(),
    )

    assert result.status == "failure"
    assert result.graph_instance is None


@pytest.mark.asyncio
async def test_matching_manifest_reuses_rk_graph(monkeypatch):
    graph = FakeGraph(succeeds=True)
    monkeypatch.setattr(tools, "get_graph", lambda **kwargs: graph)
    stable_manifest(monkeypatch, True)

    result = await tools.build_rk_graph(
        BuildRKGraphInputs(target_dataset_name="Demo"),
        FakeMainConfig(),
        llm_instance=object(),
        encoder_instance=object(),
        chunk_factory=FakeChunkFactory(),
    )

    assert result.status == "success"
    assert result.graph_instance is graph
    assert result.node_count == 3
    assert result.edge_count == 2
    assert graph.build_calls == [False]


@pytest.mark.asyncio
async def test_passage_graph_uses_canonical_namespace_name(monkeypatch):
    graph = FakeGraph(succeeds=True)
    chunks = FakeChunkFactory()
    monkeypatch.setattr(tools, "get_graph", lambda **kwargs: graph)

    result = await tools.build_passage_graph(
        BuildPassageGraphInputs(target_dataset_name="Demo"),
        FakeMainConfig(),
        llm_instance=object(),
        encoder_instance=object(),
        chunk_factory=chunks,
    )

    assert result.status == "success"
    assert chunks.namespace_types == ["passage_graph"]


@pytest.mark.asyncio
async def test_empty_forced_graph_is_failure_and_does_not_invalidate_old_artifacts(monkeypatch):
    graph = FakeGraph(succeeds=True, node_num=0, edge_num=0)
    monkeypatch.setattr(tools, "get_graph", lambda **kwargs: graph)
    monkeypatch.setattr(tools, "write_manifest", lambda graph, chunks: True)
    invalidations = []
    monkeypatch.setattr(
        tools,
        "invalidate_after_forced_graph_rebuild",
        lambda *args, **kwargs: invalidations.append((args, kwargs)),
    )

    result = await tools.build_er_graph(
        BuildERGraphInputs(target_dataset_name="Demo", force_rebuild=True),
        FakeMainConfig(),
        llm_instance=object(),
        encoder_instance=object(),
        chunk_factory=FakeChunkFactory(),
    )

    assert result.status == "failure"
    assert result.node_count == 0
    assert result.graph_instance is None
    assert invalidations == []


@pytest.mark.asyncio
async def test_successful_forced_er_rebuild_writes_manifest_then_invalidates(monkeypatch):
    graph = FakeGraph(succeeds=True, node_num=2, edge_num=1)
    monkeypatch.setattr(tools, "get_graph", lambda **kwargs: graph)
    events = []
    monkeypatch.setattr(
        tools,
        "write_manifest",
        lambda graph, chunks: events.append("manifest") or True,
    )

    def capture_invalidation(config, dataset_name, *, invalidate_sparse_matrices):
        events.append((dataset_name, invalidate_sparse_matrices))
        return []

    monkeypatch.setattr(
        tools,
        "invalidate_after_forced_graph_rebuild",
        capture_invalidation,
    )

    result = await tools.build_er_graph(
        BuildERGraphInputs(target_dataset_name="Demo", force_rebuild=True),
        FakeMainConfig(),
        llm_instance=object(),
        encoder_instance=object(),
        chunk_factory=FakeChunkFactory(),
    )

    assert result.status == "success"
    assert graph.build_calls == [True]
    assert events == ["manifest", ("Demo", True)]


@pytest.mark.asyncio
async def test_matching_er_manifest_does_not_force_or_invalidate(monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr(tools, "get_graph", lambda **kwargs: graph)
    stable_manifest(monkeypatch, True)
    invalidations = []
    monkeypatch.setattr(
        tools,
        "invalidate_after_forced_graph_rebuild",
        lambda *args, **kwargs: invalidations.append((args, kwargs)),
    )

    result = await tools.build_er_graph(
        BuildERGraphInputs(target_dataset_name="Demo", force_rebuild=False),
        FakeMainConfig(),
        llm_instance=object(),
        encoder_instance=object(),
        chunk_factory=FakeChunkFactory(),
    )

    assert result.status == "success"
    assert graph.build_calls == [False]
    assert invalidations == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("manifest_state", "message_fragment"),
    [
        (None, "manifest was missing"),
        (False, "source chunks changed"),
    ],
)
async def test_missing_or_changed_manifest_forces_er_rebuild_and_invalidation(
    monkeypatch,
    manifest_state,
    message_fragment,
):
    graph = FakeGraph()
    monkeypatch.setattr(tools, "get_graph", lambda **kwargs: graph)
    stable_manifest(monkeypatch, manifest_state)
    invalidations = []

    def capture_invalidation(config, dataset_name, *, invalidate_sparse_matrices):
        invalidations.append((dataset_name, invalidate_sparse_matrices))
        return []

    monkeypatch.setattr(
        tools,
        "invalidate_after_forced_graph_rebuild",
        capture_invalidation,
    )

    result = await tools.build_er_graph(
        BuildERGraphInputs(target_dataset_name="Demo", force_rebuild=False),
        FakeMainConfig(),
        llm_instance=object(),
        encoder_instance=object(),
        chunk_factory=FakeChunkFactory(),
    )

    assert result.status == "success"
    assert message_fragment in result.message
    assert graph.build_calls == [True]
    assert invalidations == [("Demo", True)]
