import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict

from Core.Common.Utils import mdhash_id
from typing import Any, Optional, Union, Type, Dict, List
from Core.Common.ContextMixin import ContextMixin
from Core.Common.Constants import GRAPH_FIELD_SEP
from pydantic import BaseModel, ConfigDict, model_validator
from Core.Graph.ChunkFactory import get_chunks
import tiktoken
from Core.Common.Memory import Memory
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Common.Utils import (split_string_by_multi_markers, clean_str)


class BaseGraph(ABC, ContextMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    chunks: Optional[str] = None
    # context: str  # all the context, including all necessary info
    llm_name_or_type: Optional[str] = None
    # working memory for constructing the graph
    working_memory: Memory = Memory()

    @model_validator(mode="after")
    def _update_context(
            cls: Type["BaseGraph"], data: "BaseGraph"
    ) -> "BaseGraph":

        cls.config = data.context.config
        cls.ENCODER = tiktoken.encoding_for_model(cls.config.token_model)
        cls._graph: NetworkXStorage = NetworkXStorage()  # Store the graph

        return data

    async def chunk_documents(self, docs: Union[str, list[Any]], is_chunked: bool = False) -> dict[str, dict[str, str]]:
        """Chunk the given documents into smaller chunks.

        Args:
        docs (Union[str, list[str]]): The documents to chunk, either as a single string or a list of strings.

        Returns:
        dict[str, dict[str, str]]: A dictionary where the keys are the MD5 hashes of the chunks, and the values are dictionaries containing the chunk content.
        """
        if isinstance(docs, str):
            docs = [docs]

        if isinstance(docs[0], dict):
            new_docs = {doc['id']: {"content": doc['content'].strip()} for doc in docs}
        else:
            new_docs = {mdhash_id(doc.strip(), prefix="doc-"): {"content": doc.strip()} for doc in docs}
        chunks = await get_chunks(new_docs, "chunking_by_seperators", self.ENCODER, is_chunked=is_chunked)
        self.chunks = chunks
        return chunks

    async def build_graph(self, chunks):
        """
        Builds or loads a graph based on the input chunks.

        Args:
            chunks: The input data chunks used to build the graph.

        Returns:
            The graph if it already exists, otherwise builds and returns the graph.
        """
        # If the graph already exists, load it
        if self._exist_graph():
            return self._load_graph()

        # Build the graph based on the input chunks
        self._build_graph(chunks)

    def _exist_graph(self):
        """
        Checks if the graph already exists.

        Returns:
            bool: True if the graph exists, False otherwise.
        """
        pass

    async def _merge_nodes_then_upsert(self, entity_name: str, nodes_data: List[Entity]) -> Dict[str, Any]:
        existing_node = await self.er_graph.get_node(entity_name)
        if existing_node:
            existing_source_ids = existing_node["source_id"].split(GRAPH_FIELD_SEP)
        else:
            existing_source_ids = []

        # Extract source_ids from nodes_data and merge with existing_source_ids
        new_source_ids = [dp.source_id for dp in nodes_data]
        merged_source_ids = list(set(new_source_ids) | set(existing_source_ids))

        # Create node_data with merged source_ids and entity_name
        source_id = GRAPH_FIELD_SEP.join(merged_source_ids)
        node_data = dict(source_id=source_id, entity_name=entity_name)

        # Upsert the node with the merged data
        await self.er_graph.upsert_node(entity_name, node_data=node_data)
        return node_data

    async def _merge_edges_then_upsert(self, src_id: str, tgt_id: str, edges_data: List[Relationship]) -> None:
        # Check if the edge exists and fetch existing data
        if await self.er_graph.has_edge(src_id, tgt_id):
            existing_edge_data = await self.er_graph.get_edge(src_id, tgt_id)
        else:
            existing_edge_data = {}

        # Calculate the new weight by summing weights from edges_data and existing edge
        new_weight = sum(dp.weight for dp in edges_data)
        total_weight = new_weight + existing_edge_data.get("weight", 0)

        # Merge source_ids from edges_data and existing edge data
        existing_source_ids = split_string_by_multi_markers(existing_edge_data.get("source_id", ""), [GRAPH_FIELD_SEP])
        new_edge_source_ids = [dp.source_id for dp in edges_data]
        merged_source_ids = list(set(new_edge_source_ids) | set(existing_source_ids))

        # Merge relation_names from edges_data and existing edge data
        existing_relation_name = existing_edge_data.get("relation_name", "")
        existing_relation_names = split_string_by_multi_markers(existing_relation_name, [GRAPH_FIELD_SEP])
        new_relation_names = [dp.relation_name for dp in edges_data]
        merged_relation_names = sorted(list(set(new_relation_names) | set(existing_relation_names)))
        relation_name = GRAPH_FIELD_SEP.join(merged_relation_names)
        keywords = None

        # If keywords is used
        if self.config.use_keywords:
            keywords = GRAPH_FIELD_SEP.join(
                sorted(set([dp.keywords for dp in edges_data] + split_string_by_multi_markers(
                    existing_edge_data.get("keywords", ""), [GRAPH_FIELD_SEP])
                           )))

        # Ensure src_id and tgt_id nodes exist
        for node_id in (src_id, tgt_id):
            if not await self.er_graph.has_node(node_id):
                # Upsert node with source_id and entity_name
                await self.er_graph.upsert_node(
                    node_id,
                    node_data=dict(source_id=merged_source_ids, entity_name=node_id)
                )

        # Create edge_data with merged data
        edge_data = dict(weight=total_weight, source_id=GRAPH_FIELD_SEP.join(merged_source_ids),
                         relation_name=relation_name, keywords=keywords)

        # Upsert the edge with the merged data
        await self.er_graph.upsert_edge(src_id, tgt_id, edge_data=edge_data)

    @abstractmethod
    def _extract_node_relationship(self, chunk_key_pair: tuple[str, TextChunk]):
        """
        Abstract method to extract relationships between nodes in the graph.

        This method should be implemented by subclasses to define how node relationships are extracted.
        """
        pass

    @abstractmethod
    def _build_graph(self, chunks):
        """
        Abstract method to build the graph based on the input chunks.

        Args:
            chunks: The input data chunks used to build the graph.

        This method should be implemented by subclasses to define how the graph is built from the input chunks.
        """
        pass

    async def _augment_graph(self, queries, similarity_threshold=0.8, similarity_top_k=100, duplicate=True):
        """
        For each entity in the graph, get its synonyms from the knowledge base
        queries: list of entity names
        """
        ranking = await self.entity_vdb.retrieve_batch(queries, top_k=similarity_top_k)
        entity_names = list(queries.values())
        kb_similarity = {}
        for key, entity_name in queries.items():
            rank = ranking.data[key]
            filtered_rank = rank[1:] if duplicate else rank
            kb_similarity[entity_name] = (
                [entity_names[r[0]] for r in filtered_rank],
                [r[2] / rank[0][2] for r in filtered_rank]
            )

        maybe_edges = defaultdict(list)
        # Refactored second part using dictionary iteration and enumerate
        for src_id, nns in kb_similarity.items():
            processed_nns = [clean_str(nn) for nn in nns[0]]
            for idx, (nn, score) in enumerate(zip(processed_nns, nns[1])):
                if score < similarity_threshold or idx >= similarity_top_k:
                    break
                if nn == src_id:
                    continue
                tgt_id = nn

                # No need source_id for this type of edges
                relationship = Relationship(src_id=clean_str(src_id),
                                            tgt_id=clean_str(tgt_id),
                                            source_id="N/A",
                                            weight=self.config.similarity_max * score, relation_name="similarity")
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        # Merge the edges
        maybe_edges_aug = defaultdict(list)
        for k, v in maybe_edges.items():
            maybe_edges_aug[tuple(sorted(k))].extend(v)

        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])
