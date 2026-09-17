from Core.AgentSchema.context import GraphRAGContext, _dataset_from_graph_id


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
