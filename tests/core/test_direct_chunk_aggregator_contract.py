from types import SimpleNamespace

import networkx as nx
import pytest

from Core.AgentSchema.tool_contracts import ChunkRelationshipScoreAggregatorInputs
from Core.AgentTools.chunk_tools import chunk_aggregator_tool
from Core.Schema.ChunkSchema import TextChunk


class FakeStorage:
    async def get_chunks_for_dataset(self, dataset_name):
        datasets = {
            "Alpha": [
                (
                    "chunk-alpha",
                    TextChunk(
                        tokens=2,
                        chunk_id="chunk-alpha",
                        content="alpha evidence",
                        doc_id="alpha-doc",
                        index=0,
                    ),
                )
            ],
            "Beta": [
                (
                    "chunk-beta",
                    TextChunk(
                        tokens=2,
                        chunk_id="chunk-beta",
                        content="beta evidence",
                        doc_id="beta-doc",
                        index=0,
                    ),
                )
            ],
        }
        return datasets.get(dataset_name, [])


class GraphWrapper:
    def __init__(self, graph):
        self._graph = SimpleNamespace(graph=graph)


class FakeContext:
    def __init__(self):
        alpha = nx.Graph()
        alpha.add_edge(
            "a",
            "b",
            relation_name="alpha_relation",
            source_id="chunk-alpha",
        )
        beta = nx.Graph()
        beta.add_edge(
            "x",
            "y",
            relation_name="beta_relation",
            source_id="chunk-beta",
        )
        self.graphs = {
            "Alpha_ERGraph": GraphWrapper(alpha),
            "Beta_ERGraph": GraphWrapper(beta),
        }
        self.active_dataset_name = "Alpha"
        self.chunk_storage_manager = FakeStorage()

    def list_graphs(self):
        return list(self.graphs.keys())


@pytest.mark.asyncio
async def test_empty_candidate_direct_aggregator_recovers_exact_scored_relationship_chunks():
    context = FakeContext()
    result = await chunk_aggregator_tool(
        ChunkRelationshipScoreAggregatorInputs(
            chunk_candidates=[],
            relationship_scores={"x->y": 0.9},
            top_k_chunks=5,
        ),
        context,
    )

    chunks = result.ranked_aggregated_chunks
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk-beta"
    assert chunks[0].content == "beta evidence"
    assert chunks[0].relevance_score == pytest.approx(0.9)
    assert chunks[0].metadata["graph_reference_id"] == "Beta_ERGraph"


@pytest.mark.asyncio
async def test_direct_aggregator_returns_empty_when_scores_match_no_graph_relationship():
    result = await chunk_aggregator_tool(
        ChunkRelationshipScoreAggregatorInputs(
            chunk_candidates=[],
            relationship_scores={"missing->edge": 1.0},
            top_k_chunks=5,
        ),
        FakeContext(),
    )

    assert result.ranked_aggregated_chunks == []
