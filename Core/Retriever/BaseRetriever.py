import asyncio
from abc import ABC
from Core.Common.Utils import truncate_list_by_token_size
from Core.Common.Logger import logger
import numpy as np
import asyncio
from Core.Retriever.RetrieverFactory import get_retriever_operator


class BaseRetriever(ABC):

    def __init__(self, config, retriever_context=None):
        self.config = config
        self.retriever_context = retriever_context

    async def retrieve_relevant_content(self, **kwargs):
        """
        Find the relevant contexts for the given query.
        """
        mode = kwargs.pop("mode")
        if mode not in self.mode_list:
            logger.warning(f"Invalid mode: {mode}")
            return None

        retrieve_fun = get_retriever_operator(self.type, mode)
        return await retrieve_fun(self, **kwargs)

    async def _construct_relationship_context(self, edge_datas: list[dict]):

        if not all([n is not None for n in edge_datas]):
            logger.warning("Some edges are missing, maybe the storage is damaged")
        edge_degree = await asyncio.gather(
            *[self.graph.edge_degree(r["src_id"], r["tgt_id"]) for r in edge_datas]
        )
        edge_datas = [
            {"src_id": v["src_id"], "tgt_id": v["tgt_id"], "rank": d, **v}
            for v, d in zip(edge_datas, edge_degree)
            if v is not None
        ]
        edge_datas = sorted(
            edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        edge_datas = truncate_list_by_token_size(
            edge_datas,
            key=lambda x: x["description"],
            max_token_size=self.config.max_token_for_global_context,
        )
        return edge_datas

    async def _run_personalized_pagerank(self, query, query_entities):
 # <-- NEWLY ADDED LINE
        # Run Personalized PageRank
        reset_prob_matrix = np.zeros(self.graph.node_num)

        if self.config.use_entity_similarity_for_ppr:
            # Here, we re-implement the key idea of the FastGraphRAG, you can refer to the source code for more details:
            # https://github.com/circlemind-ai/fast-graphrag/tree/main

            # --- BEGIN DIAGNOSTIC LOGS FOR FIRST CALL ---
            logger.debug(f"BaseRetriever: Diagnosing entities_vdb before first call in _run_personalized_pagerank:")
            if self.entities_vdb is None:
                logger.debug("BaseRetriever: self.entities_vdb IS NONE before first call!")
            else:
                logger.debug(f"BaseRetriever: type(self.entities_vdb) is {type(self.entities_vdb)}")
                logger.debug(f"BaseRetriever: hasattr(self.entities_vdb, 'retrieval_nodes_with_score_matrix') is {hasattr(self.entities_vdb, 'retrieval_nodes_with_score_matrix')}")
                if hasattr(self.entities_vdb, 'retrieval_nodes_with_score_matrix'):
                    logger.debug(f"BaseRetriever: self.entities_vdb.retrieval_nodes_with_score_matrix is {self.entities_vdb.retrieval_nodes_with_score_matrix}")
                else:
                    logger.error("BaseRetriever: self.entities_vdb has NO attribute 'retrieval_nodes_with_score_matrix' before first call!")
            # --- END DIAGNOSTIC LOGS FOR FIRST CALL ---

            # Use entity similarity to compute the reset probability matrix
            reset_prob_matrix += await self.entities_vdb.retrieval_nodes_with_score_matrix(query_entities, top_k=1,
                                                                                           graph=self.graph)

            # --- BEGIN DIAGNOSTIC LOGS FOR SECOND CALL ---
            logger.debug(f"BaseRetriever: Diagnosing entities_vdb before second call in _run_personalized_pagerank:")
            if self.entities_vdb is None:
                logger.error("BaseRetriever: self.entities_vdb IS NONE before second call!")
            else:
                logger.debug(f"BaseRetriever: type(self.entities_vdb) is {type(self.entities_vdb)}")
                logger.debug(f"BaseRetriever: hasattr(self.entities_vdb, 'retrieval_nodes_with_score_matrix') is {hasattr(self.entities_vdb, 'retrieval_nodes_with_score_matrix')}")
                if hasattr(self.entities_vdb, 'retrieval_nodes_with_score_matrix'):
                    logger.debug(f"BaseRetriever: self.entities_vdb.retrieval_nodes_with_score_matrix is {self.entities_vdb.retrieval_nodes_with_score_matrix}")
                else:
                    logger.error("BaseRetriever: self.entities_vdb has NO attribute 'retrieval_nodes_with_score_matrix' before second call!")
            # --- END DIAGNOSTIC LOGS FOR SECOND CALL ---

            # Run Personalized PageRank on the linked entities       
            reset_prob_matrix += await self.entities_vdb.retrieval_nodes_with_score_matrix(query,
                                                                                           top_k=self.config.top_k_entity_for_ppr,
                                                                                           graph=self.graph)
        else:
            # Set the weight of the retrieved documents based on the number of documents they appear in
            # Please refer to the HippoRAG code for more details: https://github.com/OSU-NLP-Group/HippoRAG/tree/main
            if not hasattr(self, "entity_chunk_count"):
                # Register the entity-chunk count matrix into the class when you first use it.
                # Check if entities_to_relationships exists before using it
                if hasattr(self, "entities_to_relationships") and self.entities_to_relationships is not None:
                    e2r = await self.entities_to_relationships.get()
                    r2c = await self.relationships_to_chunks.get()
                    c2e = e2r.dot(r2c).T
                    c2e[c2e.nonzero()] = 1
                    self.entity_chunk_count = c2e.sum(0).T
                else:
                    # Fallback: create a dummy entity_chunk_count if e2r/r2c not available
                    logger.warning("entities_to_relationships not available, using default entity_chunk_count")
                    # Get total number of nodes to create proper sized array
                    num_nodes = len(await self.graph.get_nodes())
                    self.entity_chunk_count = np.ones(num_nodes)  # Default to 1 for all entities

            for entity in query_entities:
    
                entity_idx = await self.graph.get_node_index(entity["entity_name"])
                if self.config.node_specificity:
                    if self.entity_chunk_count[entity_idx] == 0:
                        weight = 1
                    else:
                        weight = 1 / float(self.entity_chunk_count[entity_idx])
                    reset_prob_matrix[entity_idx] = weight
                else:
                    reset_prob_matrix[entity_idx] = 1.0
        # TODO: as a method in our NetworkXGraph class or directly use the networkx graph
        # Transform the graph to igraph format 
        return await self.graph.personalized_pagerank([reset_prob_matrix])

    async def link_query_entities(self, query_entities):

        entities = []
        for query_entity in query_entities:
            node_datas = await self.entities_vdb.retrieval_nodes(query_entity, top_k=1, graph=self.graph)
            # For entity link, we only consider the top-ranked entity
            entities.append(node_datas[0])

        return entities
