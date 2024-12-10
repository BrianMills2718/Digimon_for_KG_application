from Core.Common.Logger import logger
from Core.Retriever.BaseRetriever import BaseRetriever
import numpy as np
import asyncio
from collections import defaultdict
from Core.Common.Utils import truncate_list_by_token_size

from Core.Retriever.RetrieverFactory import register_retriever_method
class EntityRetriever(BaseRetriever):
    def __init__(self, **kwargs):

        config = kwargs.pop("config")
        super().__init__(config)
        self.mode_list = ["ppr", "vdb", "from_relation"]
        self.type = "entity"
        for key, value in kwargs.items():
            setattr(self, key, value)
           
    
    @register_retriever_method(type = "entity", method_name = "ppr")    
    async def _find_relevant_entities_by_ppr(self, query, seed_entities: list[dict]):
 
        if len(seed_entities) == 0:
            return None
        # Create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
        ppr_node_matrix = await self._run_personalized_pagerank(query, seed_entities)
        topk_indices = np.argsort(ppr_node_matrix)[-self.config.top_k:]
        nodes = await self.graph.get_node_by_indices(topk_indices)
 
        return nodes, ppr_node_matrix
    
    @register_retriever_method(type = "entity", method_name = "vdb")    
    async def _find_relevant_entities_vdb(self, seed):
        try:           
            node_datas = await self.entities_vdb.retrieval_nodes(seed, self.config.top_k, self.graph)
                    
            if not len(node_datas):
                return None
            if not all([n is not None for n in node_datas]):
                logger.warning("Some nodes are missing, maybe the storage is damaged")
            node_degrees = await asyncio.gather(
                *[self.graph.node_degree(node["entity_name"]) for node in node_datas]
            )
            node_datas = [
                {**n, "entity_name": n["entity_name"], "rank": d}
                for n, d in zip( node_datas, node_degrees)
                if n is not None
            ]
  
            return node_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant entities_vdb: {e}")
    


   
    

    async def _find_relevant_entities_by_relation_agent(self, query: str, current_entity_relations_list: list[dict],
                                                        relations_dict: defaultdict[list], width=3):
        """
        Use agent to select the top-K relations based on the input query and entities
        Args:
            query: str, the query to be processed.
            current_entity_relations_list: list,  whose element is {"entity": entity_name, "relation": relation, "score": score, "head": bool}
            relations_dict: defaultdict[list], key is (src, rel), value is tar
        Returns:
            flag: bool,  indicator that shows whether to reason or not
            relations, heads
            cluster_chain_of_entities: list[list], reasoning paths
            candidates: list[str], entity candidates
            relations: list[str], related relation
            heads: list[bool]
        """
        # ✅
        try:
            from Core.Prompt.TogPrompt import  score_entity_candidates_prompt
            total_candidates = []
            total_scores = []
            total_relations = []
            total_topic_entities = []
            total_head = []

            for index, entity in enumerate(current_entity_relations_list):
                candidate_list = relations_dict[(entity["entity"], entity["relation"])]

                # score these candidate entities
                if len(candidate_list) == 1:
                    scores = [entity["score"]]
                elif len(candidate_list) == 0:
                    scores = [0.0]
                else:
                    # agent
                    prompt = score_entity_candidates_prompt.format(query, entity["relation"]) + '; '.join(
                        candidate_list) + ';' + '\nScore: '
                    result = await self.llm.aask(msg=[
                        {"role": "user",
                         "content": prompt}
                    ])

                    # clean
                    import re
                    scores = re.findall(r'\d+\.\d+', result)
                    scores = [float(number) for number in scores]
                    if len(scores) != len(candidate_list):
                        logger.info("All entities are created with equal scores.")
                        scores = [1 / len(candidate_list)] * len(candidate_list)

                # update
                if len(candidate_list) == 0:
                    candidate_list.append("[FINISH]")
                candidates_relation = [entity['relation']] * len(candidate_list)
                topic_entities = [entity['entity']] * len(candidate_list)
                head_num = [entity['head']] * len(candidate_list)
                total_candidates.extend(candidate_list)
                total_scores.extend(scores)
                total_relations.extend(candidates_relation)
                total_topic_entities.extend(topic_entities)
                total_head.extend(head_num)

            # entity_prune
            zipped = list(zip(total_relations, total_candidates, total_topic_entities, total_head, total_scores))
            sorted_zipped = sorted(zipped, key=lambda x: x[4], reverse=True)
            sorted_relations = list(map(lambda x: x[0], sorted_zipped))
            sorted_candidates = list(map(lambda x: x[1], sorted_zipped))
            sorted_topic_entities = list(map(lambda x: x[2], sorted_zipped))
            sorted_head = list(map(lambda x: x[3], sorted_zipped))
            sorted_scores = list(map(lambda x: x[4], sorted_zipped))

            # prune according to width
            relations = sorted_relations[:width]
            candidates = sorted_candidates[:width]
            topics = sorted_topic_entities[:width]
            heads = sorted_head[:width]
            scores = sorted_scores[:width]

            # merge and output
            merged_list = list(zip(relations, candidates, topics, heads, scores))
            filtered_list = [(rel, ent, top, hea, score) for rel, ent, top, hea, score in merged_list if score != 0]
            if len(filtered_list) == 0:
                return False, [], [], [], []
            relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))
            cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
            return True, cluster_chain_of_entities, candidates, relations, heads
        except Exception as e:
            logger.exception(f"Failed to find relevant entities by relation agent: {e}")
      
     
    @register_retriever_method(type = "entity", method_name = "from_relation")    
    async def _find_relevant_entities_by_relationships(self, seed):
        entity_names = set()
        for e in seed:
            entity_names.add(e["src_id"])
            entity_names.add(e["tgt_id"])

        node_datas = await asyncio.gather(
            *[self.graph.get_node(entity_name) for entity_name in entity_names]
        )

        node_degrees = await asyncio.gather(
            *[self.graph.node_degree(entity_name) for entity_name in entity_names]
        )
        node_datas = [
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(entity_names, node_datas, node_degrees)
        ]

        node_datas = truncate_list_by_token_size(
            node_datas,
            key=lambda x: x["description"],
            max_token_size = self.config.max_token_for_local_context,
        )

        return node_datas 
    
    async def _find_relevant_tree_nodes_vdb(self, query, top_k=5):
        # ✅
        try:
            assert self.config.use_entities_vdb
            node_datas = await self.entities_vdb.retrieval(query, top_k)
            import pdb
            pdb.set_trace()             
            if not len(node_datas):
                return None
            if not all([n is not None for n in node_datas]):
                logger.warning("Some nodes are missing, maybe the storage is damaged")
            node_degrees = await asyncio.gather(
                *[self.graph.node_degree(node["entity_name"]) for node in node_datas]
            )
            node_datas = [
                {**n, "entity_name": n["entity_name"], "rank": d}
                for n, d in zip( node_datas, node_degrees)
                if n is not None
            ]
  
            return node_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant entities_vdb: {e}")
   
   
   





    async def _link_entities(self, query_entities):

        entities = []
        for query_entity in query_entities: 
            node_datas = await self.entities_vdb.retrieval_nodes(query_entity, top_k=1, graph = self.graph)
            # For entity link, we only consider the top-ranked entity
            entities.append(node_datas[0]) 
 
        return entities
