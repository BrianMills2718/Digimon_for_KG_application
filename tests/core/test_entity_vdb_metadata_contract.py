from types import SimpleNamespace

import pytest

import Core.AgentTools.entity_vdb_tools as entity_vdb_tools
from Core.AgentSchema.tool_contracts import EntityVDBBuildInputs


class FakeGraph:
    async def nodes_data(self):
        return [
            {
                "content": "leaf text",
                "index": 7,
                "layer": 0,
            }
        ]


class IdentityGraph:
    async def nodes_data(self):
        return [
            {
                "entity_name": "zorathian empire",
                "entity_type": "organization",
                "description": "an interstellar civilization",
                "content": "zorathian empire: organization: an interstellar civilization",
                "source_id": "chunk-z",
            }
        ]


class FakeContext:
    def __init__(self, graph=None):
        self.embedding_provider = object()
        self.graph = graph or FakeGraph()
        self.vdbs = {}

    def get_graph_instance(self, graph_id):
        return self.graph

    def get_vdb_instance(self, vdb_id):
        return self.vdbs.get(vdb_id)

    def add_vdb_instance(self, vdb_id, vdb):
        self.vdbs[vdb_id] = vdb

    def list_vdbs(self):
        return list(self.vdbs.keys())


class CapturingFaissIndex:
    captured = None

    def __init__(self, config):
        self.config = config
        self._index = object()

    async def build_index(self, elements, meta_data, force=False):
        type(self).captured = {
            "elements": elements,
            "meta_data": list(meta_data),
            "force": force,
        }
        return True


def patch_index(monkeypatch):
    monkeypatch.setattr(entity_vdb_tools, "FaissIndex", CapturingFaissIndex)
    monkeypatch.setattr(
        entity_vdb_tools,
        "create_faiss_index_config",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )


@pytest.mark.asyncio
async def test_tree_graph_metadata_survives_entity_vdb_build(monkeypatch):
    patch_index(monkeypatch)

    ctx = FakeContext()
    result = await entity_vdb_tools.entity_vdb_build_tool(
        EntityVDBBuildInputs(
            graph_reference_id="Demo_TreeGraph",
            vdb_collection_name="Demo_entities",
        ),
        ctx,
    )

    assert result.num_entities_indexed == 1
    captured = CapturingFaissIndex.captured
    assert captured["elements"][0]["id"] == "7"
    assert captured["elements"][0]["index"] == 7
    assert captured["elements"][0]["layer"] == 0
    assert "index" in captured["meta_data"]
    assert "layer" in captured["meta_data"]
    assert "content" not in captured["meta_data"]


@pytest.mark.asyncio
async def test_entity_vdb_embeds_name_type_and_description_when_content_is_available(monkeypatch):
    patch_index(monkeypatch)

    ctx = FakeContext(graph=IdentityGraph())
    result = await entity_vdb_tools.entity_vdb_build_tool(
        EntityVDBBuildInputs(
            graph_reference_id="Demo_ERGraph",
            vdb_collection_name="Demo_entities",
        ),
        ctx,
    )

    assert result.num_entities_indexed == 1
    document = CapturingFaissIndex.captured["elements"][0]
    assert document["id"] == "zorathian empire"
    assert document["content"] == (
        "zorathian empire: organization: an interstellar civilization"
    )
    assert document["name"] == "zorathian empire"
    assert document["source_id"] == "chunk-z"
