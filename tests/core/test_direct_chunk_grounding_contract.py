from types import SimpleNamespace

import networkx as nx
import pytest

from Core.AgentSchema.tool_contracts import ChunkGetTextForEntitiesInput
from Core.AgentTools.chunk_tools import (
    chunk_from_relationships_tool,
    chunk_get_text_for_entities_tool,
)
from Core.Schema.ChunkSchema import TextChunk


class FakeChunkStorage:
    async def get_chunks_for_dataset(self, dataset_name):
        assert dataset_name == "Demo"
        return [
            (
                "chunk-real",
                TextChunk(
                    tokens=3,
                    chunk_id="chunk-real",
                    content="real source evidence",
                    doc_id="doc-real",
                    index=0,
                    title="real",
                ),
            )
        ]


class FakeContext:
    def __init__(self, graph):
        self.graph_instance = SimpleNamespace(_graph=SimpleNamespace(graph=graph))
        self.chunk_storage_manager = FakeChunkStorage()

    def get_graph_instance(self, graph_id):
        assert graph_id == "Demo_ERGraph"
        return self.graph_instance


@pytest.mark.asyncio
async def test_relationship_chunk_tool_skips_unresolved_reference_instead_of_fabricating_text():
    graph = nx.Graph()
    graph.add_edge(
        "alpha",
        "beta",
        relation_name="related_to",
        source_id="chunk-missing",
        chunks=["chunk-missing"],
    )
    ctx = FakeContext(graph)

    result = await chunk_from_relationships_tool(
        {
            "target_relationships": ["alpha->beta"],
            "document_collection_id": "Demo_ERGraph",
        },
        ctx,
    )

    assert result["relevant_chunks"] == []


@pytest.mark.asyncio
async def test_entity_chunk_tool_returns_exact_source_and_never_content_guesses_missing_id():
    graph = nx.Graph()
    graph.add_node("alpha", source_id="chunk-real")
    graph.add_node("beta", source_id="chunk-missing")
    ctx = FakeContext(graph)

    exact = await chunk_get_text_for_entities_tool(
        ChunkGetTextForEntitiesInput(
            graph_reference_id="Demo_ERGraph",
            entity_ids=["alpha"],
        ),
        ctx,
    )
    assert len(exact["retrieved_chunks"]) == 1
    assert exact["retrieved_chunks"][0]["chunk_id"] == "chunk-real"
    assert exact["retrieved_chunks"][0]["text_content"] == "real source evidence"

    missing = await chunk_get_text_for_entities_tool(
        ChunkGetTextForEntitiesInput(
            graph_reference_id="Demo_ERGraph",
            entity_ids=["beta"],
        ),
        ctx,
    )
    assert missing["retrieved_chunks"] == []
