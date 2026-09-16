import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List

import igraph as ig
import numpy as np
from scipy.sparse import csr_matrix

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Common.Memory import Memory
from Core.Common.Utils import (
    build_data_for_merge,
    clean_str,
    csr_from_indices,
    csr_from_indices_list,
    decode_string_by_tiktoken,
    encode_string_by_tiktoken,
)
from Core.Prompt import GraphPrompt
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Utils.MergeER import MergeEntity, MergeRelationship


class BaseGraph(ABC):
    @property
    def capabilities(self):
        """Return set of GraphCapability flags derived from config."""
        from Core.Schema.GraphCapabilities import GraphCapability

        caps = {GraphCapability.SUPPORTS_SUBGRAPH}
        cfg = self.config if self.config else None
        if cfg:
            if getattr(cfg, "enable_entity_types", False) or getattr(cfg, "extract_two_step", True) is False:
                caps.add(GraphCapability.HAS_ENTITY_TYPES)
            if getattr(cfg, "enable_edge_keywords", False):
                caps.add(GraphCapability.HAS_EDGE_KEYWORDS)
        if not getattr(self, "_is_tree_graph", False):
            caps.add(GraphCapability.HAS_DESCRIPTIONS)
            caps.add(GraphCapability.HAS_EDGE_DESCRIPTIONS)
            caps.add(GraphCapability.SUPPORTS_PPR)
        else:
            caps.add(GraphCapability.HAS_TREE_LAYERS)
        if getattr(self, "_has_communities", False):
            caps.add(GraphCapability.HAS_COMMUNITIES)
        if getattr(self, "_is_passage_graph", False):
            caps.add(GraphCapability.HAS_PASSAGES)
        return caps

    async def load_persisted_graph(self, force: bool = False) -> bool:
        if self._graph is None:
            logger.error("Graph storage object (_graph) is not initialized in BaseGraph.")
            return False
        logger.info(
            f"Attempting to load persisted graph via {self._graph.__class__.__name__}.load_graph(force={force})"
        )
        return await self._graph.load_graph(force)

    def __init__(self, config, llm, encoder):
        self.working_memory: Memory = Memory()
        self.config = config
        self.llm = llm
        self.ENCODER = encoder
        self._graph = None

    async def build_graph(self, chunks, force: bool = False) -> bool:
        logger.info("Starting build graph for the given documents")
        build_successful = False

        is_exist = await self._load_graph(force)
        if force or not is_exist:
            await self._clear()
            build_successful = await self._build_graph(chunks)
            if build_successful:
                await self._persist_graph(force)
                logger.info("Graph built successfully and persisted.")
            else:
                logger.error("Graph building failed in _build_graph, skipping persistence.")
        else:
            logger.info("Graph loaded from existing artifacts, build not forced.")
            build_successful = True

        if build_successful:
            logger.info("✅ Finished the graph building stage successfully.")
        else:
            logger.error("❌ Finished the graph building stage with errors.")
        return build_successful

    async def _load_graph(self, force: bool = False):
        return await self._graph.load_graph(force)

    @property
    def namespace(self):
        return None

    @namespace.setter
    def namespace(self, namespace):
        self._graph.namespace = namespace

    @property
    def entity_metakey(self):
        return "entity_name"

    async def _merge_nodes_then_upsert(self, entity_name: str, nodes_data: List[Entity]):
        existing_node = await self._graph.get_node(entity_name)
        existing_data = build_data_for_merge(existing_node) if existing_node else defaultdict(list)
        upsert_nodes_data = defaultdict(list)
        for node in nodes_data:
            for node_key, node_value in node.as_dict.items():
                upsert_nodes_data[node_key].append(node_value)

        merge_description = (
            MergeEntity.merge_descriptions(
                existing_data["description"], upsert_nodes_data["description"]
            )
            if getattr(self.config, "enable_entity_description", True)
            else None
        )
        description = (
            await self._handle_entity_relation_summary(entity_name, merge_description)
            if merge_description
            else ""
        )
        source_id = MergeEntity.merge_source_ids(
            existing_data["source_id"], upsert_nodes_data["source_id"]
        )
        new_entity_type = (
            MergeEntity.merge_types(
                existing_data["entity_type"], upsert_nodes_data["entity_type"]
            )
            if getattr(self.config, "enable_entity_type", True)
            else ""
        )
        node_data = dict(
            source_id=source_id,
            entity_name=entity_name,
            entity_type=new_entity_type,
            description=description,
        )
        await self._graph.upsert_node(entity_name, node_data=node_data)

    async def _merge_edges_then_upsert(
        self, src_id: str, tgt_id: str, edges_data: List[Relationship]
    ) -> None:
        existing_edge = (
            await self._graph.get_edge(src_id, tgt_id)
            if await self._graph.has_edge(src_id, tgt_id)
            else None
        )
        existing_edge_data = (
            build_data_for_merge(existing_edge) if existing_edge else defaultdict(list)
        )
        upsert_edge_data = defaultdict(list)
        for edge in edges_data:
            for edge_key, edge_value in edge.as_dict.items():
                upsert_edge_data[edge_key].append(edge_value)

        source_id = MergeRelationship.merge_source_ids(
            existing_edge_data["source_id"], upsert_edge_data["source_id"]
        )
        total_weight = MergeRelationship.merge_weight(
            existing_edge_data["weight"], upsert_edge_data["weight"]
        )
        merge_description = (
            MergeRelationship.merge_descriptions(
                existing_edge_data["description"], upsert_edge_data["description"]
            )
            if getattr(self.config, "enable_edge_description", True)
            else ""
        )
        description = (
            await self._handle_entity_relation_summary((src_id, tgt_id), merge_description)
            if getattr(self.config, "enable_edge_description", True)
            else ""
        )
        keywords = (
            MergeRelationship.merge_keywords(
                existing_edge_data["keywords"], upsert_edge_data["keywords"]
            )
            if getattr(self.config, "enable_edge_keywords", True)
            else ""
        )
        relation_name = (
            MergeRelationship.merge_relation_name(
                existing_edge_data["relation_name"], upsert_edge_data["relation_name"]
            )
            if getattr(self.config, "enable_edge_name", True)
            else ""
        )

        for node_id in (src_id, tgt_id):
            if not await self._graph.has_node(node_id):
                await self._graph.upsert_node(
                    node_id,
                    node_data=dict(
                        source_id=source_id,
                        entity_name=node_id,
                        entity_type="",
                        description="",
                    ),
                )

        edge_data = dict(
            weight=total_weight,
            source_id=source_id,
            relation_name=relation_name,
            keywords=keywords,
            description=description,
            src_id=src_id,
            tgt_id=tgt_id,
        )
        await self._graph.upsert_edge(src_id, tgt_id, edge_data=edge_data)

    @abstractmethod
    def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]):
        pass

    @abstractmethod
    def _build_graph(self, chunks):
        pass

    async def augment_graph_by_similarity_search(self, entity_vdb, duplicate=False):
        logger.info("Starting augment the existing graph with similarity edges")
        ranking = {}
        import tqdm

        nodes = await self._graph.nodes()
        for node in tqdm.tqdm(nodes, total=len(nodes)):
            ranking[node] = await entity_vdb.retrieval(
                query=node, top_k=self.config.similarity_top_k
            )

        is_euclidean_distance = False
        kb_similarity = defaultdict(list)
        for key, rank in ranking.items():
            if not rank:
                continue
            max_score = max(ns_item.score for ns_item in rank)
            for idx, ns_item in enumerate(rank):
                score = ns_item.score
                if idx == 0 and score == 0:
                    is_euclidean_distance = True
                if not duplicate and idx == 0:
                    continue
                if is_euclidean_distance:
                    adjusted = 1 - score / max_score if max_score else 0.0
                else:
                    adjusted = score / max_score if max_score else 0.0
                name = (
                    ns_item.metadata.get("entity_name")
                    or ns_item.metadata.get("name")
                    or ns_item.metadata.get("id")
                )
                if name:
                    kb_similarity[key].append((name, adjusted))

        maybe_edges = defaultdict(list)
        for src_id, nns in kb_similarity.items():
            for idx, (neighbor, score) in enumerate(nns):
                if score < self.config.similarity_threshold or idx >= self.config.similarity_top_k:
                    break
                if neighbor == src_id:
                    continue
                relationship = Relationship(
                    src_id=clean_str(src_id),
                    tgt_id=clean_str(neighbor),
                    source_id="N/A",
                    weight=self.config.similarity_max * score,
                    relation_name="similarity",
                )
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        logger.info(f"Augmenting graph with {len(maybe_edges)} edges")
        await asyncio.gather(
            *[
                self._merge_edges_then_upsert(key[0], key[1], value)
                for key, value in maybe_edges.items()
            ]
        )
        await self._persist_graph()
        logger.info("✅ Finished augmenting graph with similarity edges")

    async def __graph__(self, elements: list):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)
        for m_nodes, m_edges in elements:
            for key, value in m_nodes.items():
                maybe_nodes[key].extend(value)
            for key, value in m_edges.items():
                maybe_edges[tuple(sorted(key))].extend(value)

        await asyncio.gather(
            *[self._merge_nodes_then_upsert(key, value) for key, value in maybe_nodes.items()]
        )
        await asyncio.gather(
            *[
                self._merge_edges_then_upsert(key[0], key[1], value)
                for key, value in maybe_edges.items()
            ]
        )

    async def _handle_entity_relation_summary(
        self, entity_or_relation_name: str, description: str
    ) -> str:
        """Summarize long merged descriptions without depending on embedding tokenizer APIs."""
        if not description:
            return ""

        tokens = encode_string_by_tiktoken(description)
        summary_max_tokens = getattr(self.config, "summary_max_tokens", 500) or 500
        if len(tokens) < summary_max_tokens:
            return description

        llm_model_max_token_size = (
            getattr(self.config, "llm_model_max_token_size", None) or 32768
        )
        use_description = decode_string_by_tiktoken(
            tokens[:llm_model_max_token_size]
        )

        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=use_description.split(GRAPH_FIELD_SEP),
        )
        use_prompt = GraphPrompt.SUMMARIZE_ENTITY_DESCRIPTIONS.format(**context_base)
        logger.debug(f"Trigger summary: {entity_or_relation_name}")
        max_tokens = getattr(self.config, "summary_max_tokens", 256) or 256
        return await self.llm.aask(use_prompt, max_tokens=max_tokens)

    async def _persist_graph(self, force=False):
        await self._graph.persist(force)

    async def nodes_data(self):
        return await self._graph.get_nodes_data()

    async def edges_data(self, need_content=True):
        return await self._graph.get_edges_data(need_content)

    async def subgraphs_data(self):
        return await self._graph.get_subgraph_from_same_chunk()

    async def node_metadata(self):
        return await self._graph.get_node_metadata()

    async def edge_metadata(self):
        return await self._graph.get_edge_metadata()

    async def subgraph_metadata(self):
        return await self._graph.get_subgraph_metadata()

    async def stable_largest_cc(self):
        if isinstance(self._graph, NetworkXStorage):
            return await self._graph.get_stable_largest_cc()
        logger.error("Only NETWORKX is supported for finding the largest connected component.")
        return None

    async def cluster_data_to_subgraphs(self, cluster_data: dict):
        if isinstance(self._graph, NetworkXStorage):
            await self._graph.cluster_data_to_subgraphs(cluster_data)
            return None
        logger.error("Only NETWORKX is supported for constructing the cluster <-> node mapping.")
        return None

    async def community_schema(self):
        return await self._graph.get_community_schema()

    async def get_node(self, node_id):
        return await self._graph.get_node(node_id)

    async def get_node_by_index(self, index):
        return await self._graph.get_node_by_index(index)

    async def get_edge_by_index(self, index):
        return await self._graph.get_edge_by_index(index)

    async def get_node_by_indices(self, node_idxs):
        return await asyncio.gather(*[self.get_node_by_index(idx) for idx in node_idxs])

    async def get_edge_by_indices(self, edge_idxs):
        return await asyncio.gather(*[self.get_edge_by_index(idx) for idx in edge_idxs])

    async def get_edge(self, src, tgt):
        return await self._graph.get_edge(src, tgt)

    async def nodes(self):
        return await self._graph.nodes()

    async def edges(self):
        return await self._graph.edges()

    async def node_degree(self, node_id):
        return await self._graph.node_degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str):
        return await self._graph.edge_degree(src_id, tgt_id)

    async def get_node_edges(self, source_node_id: str):
        return await self._graph.get_node_edges(source_node_id)

    @property
    def node_num(self):
        return self._graph.get_node_num()

    @property
    def edge_num(self):
        return self._graph.get_edge_num()

    def get_induced_subgraph(self, nodes: list[str]):
        return self._graph.get_induced_subgraph(nodes)

    async def get_entities_to_relationships_map(self, is_directed=False):
        if self.node_num == 0:
            return csr_matrix((0, 0))

        node_neighbors = {
            node: list(await self._graph.neighbors(node))
            for node in await self._graph.nodes()
        }
        data = []
        for node, neighbors in node_neighbors.items():
            for neighbor in neighbors:
                edge_index = self._graph.get_edge_index(node, neighbor)
                if edge_index == -1:
                    continue
                node_index = await self._graph.get_node_index(node)
                data.append([node_index, edge_index])
                if not is_directed:
                    neighbor_index = await self._graph.get_node_index(neighbor)
                    data.append([neighbor_index, edge_index])

        return csr_from_indices(data, shape=(self.node_num, self.edge_num))

    async def get_relationships_attrs(self, key):
        if self.edge_num == 0:
            return []
        return [edge[key] for edge in await self.edges_data(False)]

    async def get_relationships_to_chunks_map(self, doc_chunk):
        raw_relationships_to_chunks = await self.get_relationships_attrs(key="source_id")
        raw_relationships_to_chunks = [
            [i for i in await doc_chunk.get_index_by_merge_key(chunk_ids) if i is not None]
            for chunk_ids in raw_relationships_to_chunks
        ]
        return csr_from_indices_list(
            raw_relationships_to_chunks,
            shape=(len(raw_relationships_to_chunks), await doc_chunk.size),
        )

    async def get_edge_weight(self, src_id: str, tgt_id: str):
        return await self._graph.get_edge_weight(src_id, tgt_id)

    async def get_node_index(self, node_key):
        return await self._graph.get_node_index(node_key)

    async def get_node_indices(self, node_keys):
        return await asyncio.gather(*[self.get_node_index(key) for key in node_keys])

    async def personalized_pagerank(self, reset_prob_chunk, damping: float = 0.1):
        pageranked_probabilities = []
        igraph_ = ig.Graph.from_networkx(self._graph.graph)
        igraph_.es["weight"] = [
            await self.get_edge_weight(edge[0], edge[1])
            for edge in list(await self.edges())
        ]

        for reset_prob in reset_prob_chunk:
            pageranked_probs = igraph_.personalized_pagerank(
                vertices=range(self.node_num),
                damping=damping,
                directed=False,
                weights="weight",
                reset=reset_prob,
                implementation="prpack",
            )
            pageranked_probabilities.append(np.array(pageranked_probs))

        pageranked_probabilities = np.array(pageranked_probabilities)
        return pageranked_probabilities[0]

    async def get_neighbors(self, node_id: str):
        return await self._graph.neighbors(node_id)

    async def get_nodes(self):
        return await self._graph.nodes()

    async def find_k_hop_neighbors_batch(self, start_nodes: list[str], k: int):
        return await self._graph.find_k_hop_neighbors_batch(start_nodes=start_nodes, k=k)

    async def get_edge_relation_name_batch(self, edges: list[tuple[str, str]]):
        return await self._graph.get_edge_relation_name_batch(edges=edges)

    async def get_neighbors_from_sources(self, start_nodes: list[str]):
        return await self._graph.get_neighbors_from_sources(start_nodes=start_nodes)

    async def get_paths_from_sources(
        self, start_nodes: list[str], cutoff: int = 5
    ) -> list[tuple[str, str, str]]:
        return await self._graph.get_paths_from_sources(start_nodes=start_nodes, cutoff=cutoff)

    async def _clear(self):
        self._graph.clear()
