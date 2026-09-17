from types import SimpleNamespace

from Config.GraphConfig import GraphConfig
from Core.Graph.RKGraph import RKGraph


def test_default_graph_config_preserves_extracted_metadata():
    config = GraphConfig()

    assert config.enable_entity_description is True
    assert config.enable_entity_type is True
    assert config.enable_edge_description is True
    assert config.enable_edge_keywords is False


def test_rk_graph_enables_keyword_extraction_explicitly():
    graph_config = GraphConfig(enable_edge_keywords=False)
    full_config = SimpleNamespace(graph=graph_config)

    graph = RKGraph(config=full_config, llm=object(), encoder=object())

    assert graph.graph_config.enable_edge_keywords is True
