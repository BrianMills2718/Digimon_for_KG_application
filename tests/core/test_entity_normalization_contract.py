from types import SimpleNamespace

import networkx as nx
import pytest

from Core.Common.EntityNormalization import normalize_entity_id, normalize_graph_text
from Core.Community.LeidenCommunity import LeidenCommunity
from Core.Graph.DelimiterExtraction import DelimiterExtractionMixin
from Core.Graph.ERGraph import ERGraph
from Core.Operators.entity.link import entity_link
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


class FakeNamespace:
    def get_save_path(self, suffix=None):
        base = "/tmp/digimon-test-community"
        return f"{base}/{suffix}" if suffix else base


class DummyExtractor(DelimiterExtractionMixin):
    def __init__(self):
        self.config = SimpleNamespace(
            loaded_custom_ontology=None,
            enable_edge_keywords=False,
        )
        self.graph_config = self.config


class NoCallVDB:
    async def retrieval_nodes(self, *args, **kwargs):
        raise AssertionError("exact Unicode graph match should not call the VDB")


class UnicodeGraph:
    entity_metakey = "entity_name"

    def __init__(self):
        self.nodes = {
            "москва": {
                "entity_name": "москва",
                "source_id": "chunk-moscow",
                "entity_type": "location",
                "description": "capital city",
            }
        }

    async def get_node(self, entity_id):
        return self.nodes.get(entity_id)


def test_normalization_preserves_unicode_identity_and_semantic_text():
    assert normalize_entity_id("Москва") == "москва"
    assert normalize_entity_id("北京") == "北京"
    assert normalize_entity_id("José Álvarez") == "josé álvarez"
    assert normalize_entity_id("ACME, Inc.") == "acme inc"
    assert normalize_entity_id("  Scott\tDerrickson  ") == "scott derrickson"
    assert normalize_graph_text('"Столица России — Москва."') == "Столица России — Москва."
    assert normalize_graph_text("北京位于中国。") == "北京位于中国。"


@pytest.mark.asyncio
async def test_delimiter_extraction_preserves_unicode_identity_and_descriptions():
    extractor = DummyExtractor()

    entity = await extractor._handle_single_entity_extraction(
        ['"entity"', '"Москва"', '"LOCATION"', '"Столица России"'],
        "chunk-1",
    )
    relationship = await extractor._handle_single_relationship_extraction(
        [
            '"relationship"',
            '"Москва"',
            '"Россия"',
            '"Москва является столицей России"',
            '1.0',
        ],
        "chunk-1",
    )

    assert entity.entity_name == "москва"
    assert entity.description == "Столица России"
    assert relationship.src_id == "москва"
    assert relationship.tgt_id == "россия"
    assert relationship.description == "Москва является столицей России"


@pytest.mark.asyncio
async def test_two_step_er_tuples_preserve_unicode_entity_endpoints():
    nodes, edges = await ERGraph._build_graph_from_tuples(
        object(),
        ["Москва", "Россия", "北京"],
        [
            ["Москва", "столица", "Россия"],
            ["北京", "位于", "中国"],
        ],
        "chunk-1",
    )

    assert {"москва", "россия", "北京"}.issubset(nodes)
    assert ("москва", "россия") in edges
    assert ("北京", "中国") in edges


@pytest.mark.asyncio
async def test_exact_entity_link_uses_unicode_normalized_graph_identity():
    ctx = SimpleNamespace(graph=UnicodeGraph(), entities_vdb=NoCallVDB())
    result = await entity_link(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[EntityRecord(entity_name="Москва")],
                producer="test",
            )
        },
        ctx=ctx,
        params={"similarity_threshold": 0.99},
    )

    linked = result["entities"].data
    assert len(linked) == 1
    assert linked[0].entity_name == "москва"
    assert linked[0].score == pytest.approx(1.0)
    assert linked[0].extra["link_method"] == "exact_graph"


@pytest.mark.asyncio
async def test_singleton_leiden_mapping_preserves_unicode_identity():
    community = LeidenCommunity(
        llm=None,
        enforce_sub_communities=False,
        namespace=FakeNamespace(),
    )
    graph = nx.Graph()
    graph.add_node("МОСКВА")

    result = await community._clustering(
        graph,
        max_cluster_size=10,
        random_seed=0,
    )

    assert result == {"москва": [{"level": 0, "cluster": "0"}]}
    assert community.community_node_map.json_data == result
