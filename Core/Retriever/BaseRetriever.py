import asyncio
from Core.Config2 import Config
from Core.Common.ContextMixin import ContextMixin
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
  
        
    def __init__(self, config):
          self.config = config

    def reset(self):
        self.memory.clear()

    @abstractmethod
    async def find_relevant_contexts(self, query, top_k=10, **context):
         """
        Find the top-k relevant contexts for the given query.
        :param query: The query string.
        :param top_k: The number of top-k relevant contexts to return.
        :return: A list of tuples, where each tuple contains the document id and the context text.
        """
        
      async def _construct_relationship_context(self, edge_datas: list[dict]):

        if not all([n is not None for n in edge_datas]):
            logger.warning("Some edges are missing, maybe the storage is damaged")
        edge_degree = await asyncio.gather(
            *[self.graph.edge_degree(r["src_id"], r["tgt_id"]) for r in edge_datas]
        )
        edge_datas = [
            {"src_id": v["src_id"], "tgt_id": v["tgt_id"], "rank": d, **v}
            for v, d in zip( edge_datas, edge_degree)
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
        # ✅
        assert self.config.use_entities_vdb
        # Run Personalized PageRank
        reset_prob_matrix = np.zeros(self.graph.node_num)

        if self.config.use_entity_similarity_for_ppr:
            # Here, we re-implement the key idea of the FastGraphRAG, you can refer to the source code for more details:
            # https://github.com/circlemind-ai/fast-graphrag/tree/main

            # Use entity similarity to compute the reset probability matrix
            reset_prob_matrix += await self.entities_vdb.retrieval_nodes_with_score_matrix(query_entities, top_k=1, graph = self.graph)
            # Run Personalized PageRank on the linked entities      
            reset_prob_matrix += await self.entities_vdb.retrieval_nodes_with_score_matrix(query, top_k=self.config.top_k_entity_for_ppr, graph = self.graph)     
        else:
            # Set the weight of the retrieved documents based on the number of documents they appear in
            # Please refer to the HippoRAG code for more details: https://github.com/OSU-NLP-Group/HippoRAG/tree/main
            if not hasattr(self, "entity_chunk_count"):
                    # Register the entity-chunk count matrix into the class when you first use it.
                    e2r = await self._entities_to_relationships.get()
                    r2c = await self._relationships_to_chunks.get()
                    c2e= e2r.dot(r2c).T
                    c2e[c2e.nonzero()] = 1
                    self.entity_chunk_count = c2e.sum(0).T
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
        #TODO: as a method in our NetworkXGraph class or directly use the networkx graph
        # Transform the graph to igraph format 
        return await self.graph.personalized_pagerank([reset_prob_matrix])
        
    