async def _build_ppr_context(self):
    """
    Build the context for the Personalized PageRank (PPR) query.

    This function constructs two mappings:
        1. chunk_to_edge: Maps chunks (document sources) to edge indices.
        2. edge_to_entity: Maps edges to the entities (nodes) they connect.

    The function iterates over all edges in the graph, retrieves relevant metadata for each edge,
    and updates the mappings. These mappings are essential for executing PPR queries efficiently.
    """
    self.chunk_to_edge = defaultdict(int)
    self.edge_to_entity = defaultdict(int)
    self.id_to_entity = defaultdict(int)

    nodes = list(self.er_graph.graph.nodes())
    edges = list(self.er_graph.graph.edges())

    async def _build_edge_chunk_mapping(edge) -> None:
        """
        Build mappings for the edges of a given graph.

        Args:
            edge (Tuple[str, str]): A tuple representing the edge (node1, node2).
            edges (list): List of all edges in the graph.
            nodes (list): List of all nodes in the graph.
            docs_to_facts (Dict[Tuple[int, int], int]): Mapping of document indices to fact indices.
            facts_to_phrases (Dict[Tuple[int, int], int]): Mapping of fact indices to phrase indices.
        """
        try:
            # Fetch edge data asynchronously
            edge_data = await self.er_graph.get_edge(edge[0], edge[1])
            source_ids = edge_data['source_id'].split(GRAPH_FIELD_SEP)
            for source_id in source_ids:
                # Map document to edge
                source_idx = self.chunk_key_to_idx[source_id]
                edge_idx = edges.index(edge)
                self.chunk_to_edge[(source_idx, edge_idx)] = 1

            # Map fact to phrases for both nodes in the edge
            node_idx_1 = nodes.index(edge[0])
            node_idx_2 = nodes.index(edge[1])

            self.edge_to_entity[(edge_idx, node_idx_1)] = 1
            self.edge_to_entity[(edge_idx, node_idx_2)] = 1

        except ValueError as ve:
            # Handle specific errors, such as when edge or node is not found
            logger.error(f"ValueError in edge {edge}: {ve}")
        except KeyError as ke:
            # Handle missing data in chunk_key_to_idx
            logger.error(f"KeyError in edge {edge}: {ke}")
        except Exception as e:
            # Handle general exceptions gracefully
            logger.error(f"Unexpected error processing edge {edge}: {e}")

    # Process all nodes asynchronously
    await asyncio.gather(*[_build_edge_chunk_mapping(edge) for edge in edges])

    for node in nodes:
        self.id_to_entity[nodes.index(node)] = node

    self.chunk_to_edge_mat = csr_array(([int(v) for v in self.chunk_to_edge.values()], (
        [int(e[0]) for e in self.chunk_to_edge.keys()], [int(e[1]) for e in self.chunk_to_edge.keys()])),
                                       shape=(len(self.chunk_key_to_idx.keys()), len(edges)))

    self.edge_to_entity_mat = csr_array(([int(v) for v in self.edge_to_entity.values()], (
        [e[0] for e in self.edge_to_entity.keys()], [e[1] for e in self.edge_to_entity.keys()])),
                                        shape=(len(edges), len(nodes)))

    self.chunk_to_entity_mat = self.chunk_to_edge_mat.dot(self.edge_to_entity_mat)
    self.chunk_to_entity_mat[self.chunk_to_entity_mat.nonzero()] = 1
    self.entity_doc_count = self.chunk_to_entity_mat.sum(0).T