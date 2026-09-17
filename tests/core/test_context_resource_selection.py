from types import SimpleNamespace

import pytest

from Core.AgentSchema.context import (
    GraphRAGContext,
    _dataset_from_graph_id,
    _graph_type_from_graph_id,
)


def make_context():
    return GraphRAGContext.model_construct(
        request_id="test",
        target_dataset_name="mcp_session",
        main_config=object(),
        llm_provider=None,
        embedding_provider=None,
        chunk_storage_manager=None,
        graphs={
            "Alpha_ERGraph": object(),
            "Beta_ERGraph": object(),
        },
        vdbs={
            "Alpha_entities": object(),
            "Beta_entities": object(),
            "Beta_relations": object(),
            "Alpha_relations": object(),
        },
        resolved_configs={},
        active_dataset_name=None,
    )


def test_dataset_name_is_parsed_from_supported_graph_ids():
    assert _dataset_from_graph_id("Alpha_ERGraph") == "Alpha"
    assert _dataset_from_graph_id("Alpha_RKGraph") == "Alpha"
    assert _dataset_from_graph_id("Alpha_TreeGraphBalanced") == "Alpha"
    assert _dataset_from_graph_id("Alpha_TreeGraph") == "Alpha"
    assert _dataset_from_graph_id("Alpha_PassageGraph") == "Alpha"
    assert _dataset_from_graph_id("arbitrary") is None


@pytest.mark.parametrize(
    ("graph_id", "graph_type"),
    [
        ("Alpha_ERGraph", "er_graph"),
        ("Alpha_RKGraph", "rkg_graph"),
        ("Alpha_TreeGraph", "tree_graph"),
        ("Alpha_TreeGraphBalanced", "tree_graph_balanced"),
        ("Alpha_PassageGraph", "passage_graph"),
    ],
)
def test_graph_type_is_parsed_from_supported_graph_ids(graph_id, graph_type):
    assert _graph_type_from_graph_id(graph_id) == graph_type


def test_graph_lookup_prioritizes_same_dataset_vdbs_without_hiding_others():
    ctx = make_context()

    assert ctx.get_graph_instance("Beta_ERGraph") is not None
    vdb_ids = ctx.list_vdbs()

    assert vdb_ids[:2] == ["Beta_entities", "Beta_relations"]
    assert set(vdb_ids) == {
        "Alpha_entities",
        "Beta_entities",
        "Beta_relations",
        "Alpha_relations",
    }


def test_switching_graph_switches_vdb_priority():
    ctx = make_context()

    ctx.get_graph_instance("Beta_ERGraph")
    assert ctx.list_vdbs()[0].startswith("Beta_")

    ctx.get_graph_instance("Alpha_ERGraph")
    assert ctx.list_vdbs()[0].startswith("Alpha_")


def test_graph_listing_prevents_legacy_substring_match_from_binding_longer_dataset():
    ctx = GraphRAGContext.model_construct(
        request_id="test",
        target_dataset_name="mcp_session",
        main_config=object(),
        llm_provider=None,
        embedding_provider=None,
        chunk_storage_manager=None,
        graphs={
            # Insert the ambiguous longer dataset first to reproduce the old bug.
            "Test2_ERGraph": object(),
            "Test_ERGraph": object(),
            "Test_RKGraph": object(),
        },
        vdbs={},
        resolved_configs={},
        active_dataset_name=None,
    )

    graph_ids = ctx.list_graphs()
    legacy_match = next(graph_id for graph_id in graph_ids if "Test" in graph_id)

    assert legacy_match == "Test_ERGraph"
    assert graph_ids.index("Test_ERGraph") < graph_ids.index("Test2_ERGraph")


class FakeChunkFactory:
    def __init__(self):
        self.calls = []

    def get_namespace(self, dataset_name, graph_type="er_graph"):
        self.calls.append((dataset_name, graph_type))
        return SimpleNamespace(path=f"/{dataset_name}/{graph_type}")


@pytest.mark.parametrize(
    ("graph_id", "expected_type"),
    [
        ("Demo_ERGraph", "er_graph"),
        ("Demo_RKGraph", "rkg_graph"),
        ("Demo_TreeGraph", "tree_graph"),
        ("Demo_TreeGraphBalanced", "tree_graph_balanced"),
        ("Demo_PassageGraph", "passage_graph"),
    ],
)
def test_registration_restores_graph_specific_namespace(graph_id, expected_type):
    chunks = FakeChunkFactory()
    storage = SimpleNamespace(
        namespace=SimpleNamespace(path="/Demo/er_graph")
    )
    graph = SimpleNamespace(_graph=storage)
    ctx = GraphRAGContext.model_construct(
        request_id="test",
        target_dataset_name="mcp_session",
        main_config=object(),
        llm_provider=None,
        embedding_provider=None,
        chunk_storage_manager=chunks,
        graphs={},
        vdbs={},
        resolved_configs={},
        active_dataset_name=None,
    )

    ctx.add_graph_instance(graph_id, graph)

    assert chunks.calls == [("Demo", expected_type)]
    assert storage.namespace.path == f"/Demo/{expected_type}"
    assert ctx.graphs[graph_id] is graph
