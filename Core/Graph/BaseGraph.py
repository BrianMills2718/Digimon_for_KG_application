import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict

from Core.Common.Logger import logger
from typing import Any, Optional, Union, Type, Dict, List
from Core.Common.ContextMixin import ContextMixin
from Core.Common.Constants import GRAPH_FIELD_SEP
from pydantic import BaseModel, ConfigDict, model_validator
import tiktoken
from Core.Common.Memory import Memory
from Core.Prompt import GraphPrompt
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Common.Utils import (split_string_by_multi_markers, clean_str)
from Core.Utils.MergeER import MergeEntity, MergeRelationship


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
            logger.info("Graph already exists")
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

    def _load_graph(self):
        """
        Loads the graph from the file
        """
        pass

    async def _merge_nodes_then_upsert(self, entity_name: str, nodes_data: List[Entity]):
        existing_node = await self.er_graph.get_node(entity_name)
        merge_nodes_data = {}
        # Groups node properties by their keys for upsert operation.
        upsert_nodes_data = defaultdict(list)
        for node in nodes_data:
            for node_key, node_value in node.as_dict.items():
                upsert_nodes_data[node_key].append(node_value)
        source_id, new_entity_type, merge_description = upsert_nodes_data["source_id"], upsert_nodes_data[
            "entity_type"], upsert_nodes_data["description"]

        if existing_node:
            merge_nodes_data.update({
                "source_id": existing_node["source_id"].split(GRAPH_FIELD_SEP),
            })
            if self.config.enable_entity_description:
                merge_nodes_data.update({
                    "description": existing_node["description"].split(GRAPH_FIELD_SEP),
                })
            if self.config.enable_entity_type:
                merge_nodes_data.update({
                    "entity_type": existing_node["entity_type"].split(GRAPH_FIELD_SEP),
                })
            source_id, new_entity_type, merge_description = await MergeEntity.merge_info(upsert_nodes_data,
                                                                                         merge_nodes_data)

        description = (
            await self._handle_entity_relation_summary(entity_name, merge_description)
            if self.config.enable_entity_description
            else ""
        )

        node_data = dict(source_id=source_id, entity_name=entity_name, entity_type=new_entity_type,
                         description=description)
        # Upsert the node with the merged data
        await self.er_graph.upsert_node(entity_name, node_data=node_data)

    async def _merge_edges_then_upsert(self, src_id: str, tgt_id: str, edges_data: List[Relationship]) -> None:
        # Check if the edge exists and fetch existing data
        merge_edge_data = {}

        existing_edge_data = await self.er_graph.get_edge(src_id, tgt_id) if await self.er_graph.has_edge(src_id,
                                                                                                          tgt_id) else None

        # Groups node properties by their keys for upsert operation.
        upsert_edge_data = defaultdict(list)
        for edge in edges_data:
            for edge_key, edge_value in edge.as_dict.items():
                upsert_edge_data[edge_key].append(edge_value)
        source_id, total_weight, merge_description, keywords, relation_name = upsert_edge_data["source_id"], \
        upsert_edge_data[
            "weight"], upsert_edge_data["description"], upsert_edge_data["keywords"], upsert_edge_data["relation_name"]
        if existing_edge_data:
            merge_edge_data.update({
                "source_id": split_string_by_multi_markers(existing_edge_data["source_id"], [GRAPH_FIELD_SEP]),
                "weight": existing_edge_data["weight"]
            })
            if self.config.enable_edge_description:
                merge_edge_data.update({"description": existing_edge_data["description"]})
            if self.config.enable_keywords:
                merge_edge_data.update({"keywords": existing_edge_data["keywords"]})
            if self.config.enable_edge_name:
                merge_edge_data.update({"relation_name": existing_edge_data["relation_name"]})
            source_id, total_weight, merge_description, keywords, relation_name = await MergeRelationship.merge_info(
                upsert_edge_data,
                merge_edge_data)

        description = (
            await self._handle_entity_relation_summary((src_id, tgt_id), merge_description)
            if self.config.enable_edge_description
            else ""
        )

        # Ensure src_id and tgt_id nodes exist
        for node_id in (src_id, tgt_id):
            if not await self.er_graph.has_node(node_id):
                # Upsert node with source_id and entity_name
                await self.er_graph.upsert_node(
                    node_id,
                    node_data=dict(source_id=source_id, entity_name=node_id)
                )
        # Create edge_data with merged data
        edge_data = dict(weight=total_weight, source_id=GRAPH_FIELD_SEP.join(source_id),
                         relation_name=relation_name, keywords=keywords, description=description)

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

    async def __graph__(self, elements: list):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)
        for m_nodes, m_edges in elements:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)

        await asyncio.gather(*[self._merge_nodes_then_upsert(k, v) for k, v in maybe_nodes.items()])
        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])

    async def _handle_entity_relation_summary(self, entity_or_relation_name: str, description: str) -> str:
        tokens = self.ENCODER.encode(description)
        if len(tokens) < self.config.summary_max_tokens:
            return description

        use_description = self.ENCODER.decode(tokens[:self.llm.get_maxtokens()])
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=use_description.split(GRAPH_FIELD_SEP)
        )
        use_prompt = GraphPrompt.SUMMARIZE_ENTITY_DESCRIPTIONS.format(**context_base)
        logger.debug(f"Trigger summary: {entity_or_relation_name}")
        return await self.llm.aask(use_prompt, max_tokens=self.config.summary_max_tokens)
