import html
import json
import os
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from pydantic import model_validator

from Core.Common.Logger import logger
from Core.Storage.BaseGraphStorage import BaseGraphStorage


class NetworkXStorage(BaseGraphStorage):
    name: str = "nx_data.graphml"  # The valid file name for NetworkX
    _graph: nx.Graph = nx.Graph()

    def load_nx_graph(self):
        # Attempting to load the graph from the specified GraphML file
        logger.info(f"Attempting to load the graph from: {self.graphml_xml_file}")
        if os.path.exists(self.graphml_xml_file):
            self._graph = nx.read_graphml(self.graphml_xml_file)
        else:
            # GraphML file doesn't exist; need to construct the graph from scratch
            logger.info("GraphML file does not exist! Need to build the graph from scratch.")

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @model_validator(mode="after")
    def _register_node2emb(cls, data):
        cls._node_embed_algorithms = {
            "node2vec": data._node2vec_embed,
        }
        return data

    @property
    def graphml_xml_file(self):
        assert self.namespace is not None
        return self.namespace.get_save_path(self.name)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    async def init_graph(self):
        self.load_nx_graph()

    @property
    def graph(self):
        return self._graph

    async def _persist(self, force):
        if os.path.exists(self.graphml_xml_file) and not force:
            return
        logger.info(f"Writing graph into {self.graphml_xml_file}")
        NetworkXStorage.write_nx_graph(self.graph, self.graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def get_edge_weight(
            self, source_node_id: str, target_node_id: str
    ) -> Union[float, None]:
        edge_data = self._graph.edges.get((source_node_id, target_node_id))
        return edge_data.get("weight") if edge_data is not None else None

    async def get_edge(
            self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict):
        self._graph.add_node(node_id, **node_data)

    # TODO: not use dict for edge_data
    async def upsert_edge(
            self, source_node_id: str, target_node_id: str, edge_data: dict
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    async def persist(self, force):
        await self._persist(force)
