from Core.Common.Logger import logger
from Core.Retriever.BaseRetriever import BaseRetriever
import numpy as np
import asyncio
from collections import defaultdict
from Core.Common.Utils import truncate_list_by_token_size
from Core.Index.TFIDFStore import TFIDFIndex
from Core.Retriever.RetrieverFactory import register_retriever_method


class EntityRetriever(BaseRetriever):
    def __init__(self, **kwargs):

        config = kwargs.pop("config")
        super().__init__(config)
        self.mode_list = ["ppr", "vdb", "from_relation", "tf_df", "all", "by_neighbors", "link_entity", "get_all", "from_relation_by_agent"]
        self.type = "entity"
        for key, value in kwargs.items():
            setattr(self, key, value)

    @register_retriever_method(type="entity", method_name="ppr")
    async def _find_relevant_entities_by_ppr(self, query, seed_entities: list[dict], link_entity=False):

        if len(seed_entities) == 0:
            return None
        if link_entity:
            seed_entities = await self.link_query_entities(seed_entities)
        # Create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
        ppr_node_matrix = await self._run_personalized_pagerank(query, seed_entities)
        topk_indices = np.argsort(ppr_node_matrix)[-self.config.top_k:]
        nodes = await self.graph.get_node_by_indices(topk_indices)

        return nodes, ppr_node_matrix

    @register_retriever_method(type="entity", method_name="vdb")
    @register_retriever_method(type="entity", method_name="vdb")
    async def _find_relevant_entities_vdb(self, seed, tree_node=False, top_k=None):
        logger.info(f"ENTITY_VDB_RETRIEVAL: Querying VDB with seed: '{seed}' for top_k: {top_k}")
        try:
            if top_k is None:
                top_k = self.config.top_k

            raw_vdb_results = await self.entities_vdb.retrieval_nodes(
                query=seed,
                top_k=top_k,
                graph=self.graph,
                tree_node=tree_node,
                need_score=True
            )

            # Detailed logging for VDB results
            if isinstance(raw_vdb_results, tuple) and len(raw_vdb_results) == 2:
                raw_nodes, raw_scores = raw_vdb_results
                logger.info(f"ENTITY_VDB_RETRIEVAL: Raw VDB nodes count: {len(raw_nodes) if raw_nodes else 0}")
                if raw_nodes:
                    logger.debug(f"ENTITY_VDB_RETRIEVAL: First raw VDB node (with score if available): {raw_nodes[0]}, Score: {raw_scores[0] if raw_scores else 'N/A'}")
            else:
                logger.info(f"ENTITY_VDB_RETRIEVAL: Raw VDB nodes count: {len(raw_vdb_results) if raw_vdb_results else 0}")
                if raw_vdb_results:
                    logger.debug(f"ENTITY_VDB_RETRIEVAL: First raw VDB node: {raw_vdb_results[0]}")

            if isinstance(raw_vdb_results, tuple) and len(raw_vdb_results) == 2:
                retrieved_nodes_data, scores = raw_vdb_results
            else:
                retrieved_nodes_data = raw_vdb_results
                scores = None

            logger.info(f"ENTITY_VDB_RETRIEVAL: Processed node_datas count: {len(retrieved_nodes_data) if retrieved_nodes_data else 0}")
            if retrieved_nodes_data:
                logger.debug(f"ENTITY_VDB_RETRIEVAL: First processed node_data: {retrieved_nodes_data[0]}")
            else:
                logger.warning("ENTITY_VDB_RETRIEVAL: No processed node_datas to return.")
                logger.warning("ENTITY_VDB_RETRIEVAL: No processed node_datas to return.")

            nodes_with_metadata = []
            for i, node_data_item in enumerate(retrieved_nodes_data):
                if node_data_item is None:
                    logger.warning(f"Encountered a None node_data_item at index {i}. Skipping.")
                    continue

                if tree_node:
                    node_id = node_data_item.get('id', node_data_item.get('index', i))
                    node_layer = node_data_item.get('layer', 0)
                    node_text = node_data_item.get('text', '')

                    if node_id is None:
                        logger.warning(f"Tree node item at index {i} is missing 'id' or 'index'. Using loop index. Item: {node_data_item}")
                        node_id = i
                else:
                    node_id = node_data_item.get(self.graph.entity_metakey, f"missing_id_{i}")
                    node_layer = node_data_item.get("layer", 0) # Assuming 'layer' might be a direct attribute or in metadata
                    node_text = node_data_item.get("content", "")

                vdb_score = scores[i] if scores and scores[i] is not None else "N/A"
                
                current_node_data = {
                    "text": node_text,
                    "id": node_id,
                    "entity_name": node_id,  # Ensure this is present for downstream consumers
                    "layer": node_layer,
                    "vdb_score": vdb_score,
                    "source_id": node_data_item.get("source_id", "Unknown"),
                    "description": node_data_item.get("description", ""),
                    "entity_type": node_data_item.get("entity_type", "")
                }
                logger.debug(f"ENTITY_VDB_CONSTRUCT: Constructed current_node_data: {current_node_data}")
                if "entity_name" not in current_node_data:
                    logger.warning(f"ENTITY_VDB_CONSTRUCT: 'entity_name' key MISSING in current_node_data for VDB metadata: {getattr(node_data_item, 'metadata', node_data_item) if node_data_item else 'N/A'}")
                elif not current_node_data["entity_name"]:
                    logger.warning(f"ENTITY_VDB_CONSTRUCT: 'entity_name' IS EMPTY/None in current_node_data for VDB metadata: {getattr(node_data_item, 'metadata', node_data_item) if node_data_item else 'N/A'}. Full dict: {current_node_data}")
                # --- BEGIN: Attach clusters from community_node_map if available ---
                if hasattr(self, 'community') and self.community and hasattr(self.community, 'community_node_map') and self.community.community_node_map:
                    try:
                        entity_cluster_info = await self.community.community_node_map.get_by_id(node_id)  # node_id is the entity_name
                        if entity_cluster_info:
                            current_node_data["clusters"] = entity_cluster_info  # Should be a list of dicts
                            logger.debug(f"ENTITY_VDB_CLUSTERS: Attached clusters for entity '{node_id}': {entity_cluster_info}")
                        else:
                            logger.debug(f"ENTITY_VDB_CLUSTERS: No cluster info found in community_node_map for entity '{node_id}'.")
                    except Exception as e:
                        logger.warning(f"ENTITY_VDB_CLUSTERS: Exception while fetching clusters for entity '{node_id}': {e}")
                else:
                    logger.debug(f"ENTITY_VDB_CLUSTERS: Community or community_node_map not available/configured. Skipping cluster attachment for entity '{node_id}'.")
                # --- END: Attach clusters ---
                # --- DIAGNOSTIC LOGGING ---
                logger.debug(f"ENTITY_VDB_DIAG: About to check for community and community_node_map for entity '{node_id}'")
                # Log type and repr for first entity only
                if i == 0:
                    logger.debug(f"ENTITY_VDB_DIAG: self.community type: {type(getattr(self, 'community', None))}, repr: {repr(getattr(self, 'community', None))}")
                    cm_map = getattr(getattr(self, 'community', None), 'community_node_map', None)
                    logger.debug(f"ENTITY_VDB_DIAG: self.community.community_node_map type: {type(cm_map)}, repr: {repr(cm_map)}")
                # --- BEGIN: Attach clusters from community_node_map if available ---
                if hasattr(self, 'community') and self.community and hasattr(self.community, 'community_node_map') and self.community.community_node_map:
                    try:
                        entity_cluster_info = await self.community.community_node_map.get_by_id(node_id)  # node_id is the entity_name
                        if entity_cluster_info:
                            current_node_data["clusters"] = entity_cluster_info  # Should be a list of dicts
                            logger.debug(f"ENTITY_VDB_CLUSTERS: Attached clusters for entity '{node_id}': {entity_cluster_info}")
                        else:
                            logger.debug(f"ENTITY_VDB_CLUSTERS: No cluster info found in community_node_map for entity '{node_id}'.")
                    except Exception as e:
                        logger.warning(f"ENTITY_VDB_CLUSTERS: Exception while fetching clusters for entity '{node_id}': {e}")
                else:
                    logger.debug(f"ENTITY_VDB_CLUSTERS: Community or community_node_map not available/configured. Skipping cluster attachment for entity '{node_id}'.")
                # --- END: Attach clusters ---
                logger.debug(f"ENTITY_VDB_DIAG: Finished cluster attachment attempt for entity '{node_id}'")
                # Log the state of current_node_data AFTER attempting to add clusters
                if "clusters" in current_node_data:
                    logger.debug(f"ENTITY_VDB_POST_CLUSTER_ATTACH: Entity '{current_node_data.get('entity_name', node_id)}' HAS 'clusters' attribute: {current_node_data['clusters']}")
                else:
                    logger.warning(f"ENTITY_VDB_POST_CLUSTER_ATTACH: Entity '{current_node_data.get('entity_name', node_id)}' still MISSING 'clusters' attribute after attachment attempt.")
                nodes_with_metadata.append(current_node_data)

            logger.debug(f"ENTITY_VDB_FINAL_LIST: Final list of processed_node_datas before assignment (first item if any): {nodes_with_metadata[0] if nodes_with_metadata else 'Empty list'}")
            for idx, p_node in enumerate(nodes_with_metadata):
                if not isinstance(p_node, dict) or "entity_name" not in p_node:
                    logger.error(f"ENTITY_VDB_FINAL_LIST_CHECK: Item at index {idx} is problematic: {p_node}")

            # Add the 'rank' key based on node degree
            if nodes_with_metadata:
                entity_names_for_degree = [node.get("entity_name") for node in nodes_with_metadata if node.get("entity_name")]
                if entity_names_for_degree:
                    try:
                        node_degrees = await asyncio.gather(
                            *[self.graph.node_degree(name) for name in entity_names_for_degree]
                        )
                        name_to_degree_map = dict(zip(entity_names_for_degree, node_degrees))
                        for node_dict in nodes_with_metadata:
                            entity_name = node_dict.get("entity_name")
                            if entity_name in name_to_degree_map:
                                node_dict["rank"] = name_to_degree_map[entity_name]
                            else:
                                node_dict["rank"] = 0
                                logger.warning(f"ENTITY_VDB_RANKING: Could not determine rank for node: {node_dict.get('id', 'Unknown ID')}, entity_name: {entity_name}")
                        logger.info(f"ENTITY_VDB_RANKING: Added 'rank' (node degree) to {len(nodes_with_metadata)} entities.")
                        if nodes_with_metadata:
                            logger.debug(f"ENTITY_VDB_RANKING: First entity after adding rank: {nodes_with_metadata[0]}")
                    except Exception as e:
                        logger.error(f"ENTITY_VDB_RANKING: Error calculating node degrees or adding rank: {e}")
                        for node_dict in nodes_with_metadata:
                            node_dict["rank"] = 0
                else:
                    logger.warning("ENTITY_VDB_RANKING: No valid entity names found in processed_node_datas to calculate degrees.")
                    for node_dict in nodes_with_metadata:
                        node_dict["rank"] = 0
            else:
                logger.info("ENTITY_VDB_RANKING: No processed_node_datas to add rank to.")

            logger.info(f"Processed {len(nodes_with_metadata)} nodes with metadata for RAPTOR.")
            return nodes_with_metadata

        except Exception as e:
            logger.exception(f"Failed to find relevant entities_vdb: {e}")
            return None

        try:
            if top_k is None:
                top_k = self.config.top_k
            node_datas = await self.entities_vdb.retrieval_nodes(query=seed, top_k=top_k, graph=self.graph,
                                                                 tree_node=tree_node)

            if not len(node_datas):
                return None
            if not all([n is not None for n in node_datas]):
                logger.warning("Some nodes are missing, maybe the storage is damaged")
            if tree_node:
                # node_datas is already a list of dicts [{id, text, layer}, ...] or tuple (list_of_dicts, scores)
                retrieved_data = node_datas
                if not retrieved_data:
                    logger.warning(f"No tree_node data returned from VDB for seed: {seed}")
                    return []
                if not retrieved_data: # Check if retrieved_data itself is None or empty
                    logger.warning(f"No data returned from self.entities_vdb.retrieval_nodes for seed: {seed}")
                    return []
                
                if isinstance(retrieved_data, tuple) and len(retrieved_data) == 2:
                    nodes_with_metadata, scores = retrieved_data
                    logger.info(f"Scores retrieved from VDB (tuple): {scores}") # LOG THE SCORES
                else: 
                    # This path implies retrieval_nodes did not return scores as a tuple
                    nodes_with_metadata = retrieved_data # Assuming it's just the list of nodes
                    scores = [None] * len(nodes_with_metadata) if nodes_with_metadata else []
                    logger.warning(f"VDB did not return scores as a tuple. Scores set to: {scores}") # LOG THE SCORES
                
                if not nodes_with_metadata: # Check after potential unpacking
                    logger.warning(f"Empty node_with_metadata list after VDB retrieval for seed: {seed}")
                    return []

                # Ensure nodes_with_metadata is a list of mutable dicts
                processed_nodes_with_metadata = []
                for i, node_data_item in enumerate(nodes_with_metadata):
                    # If node_data_item is not a dict (e.g., a TextNode object still), convert or handle
                    if not isinstance(node_data_item, dict):
                        logger.warning(f"Node data item is not a dict: {type(node_data_item)}. Attempting to use as is or convert if known type.")
                        current_node_meta = dict(node_data_item) if hasattr(node_data_item, '__dict__') else {}
                        if not current_node_meta and isinstance(node_data_item, dict): # if it was already a dict
                            current_node_meta = node_data_item
                    else:
                        current_node_meta = dict(node_data_item) # Make a mutable copy if it's already a dict

                    if i < len(scores) and scores[i] is not None: 
                        current_node_meta['vdb_score'] = scores[i]
                    else:
                        current_node_meta['vdb_score'] = "N/A" # Explicitly set to N/A if no score
                    processed_nodes_with_metadata.append(current_node_meta)
                
                logger.info(f"Retrieved {len(processed_nodes_with_metadata)} tree nodes with metadata for RAPTOR.")
                return processed_nodes_with_metadata

            node_degrees = await asyncio.gather(
                *[self.graph.node_degree(node["entity_name"]) for node in node_datas]
            )
            node_datas = [
                {**n, "entity_name": n["entity_name"], "rank": d}
                for n, d in zip(node_datas, node_degrees)
                if n is not None
            ]

            return node_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant entities_vdb: {e}")

    @register_retriever_method(type="entity", method_name="tf_df")
    async def _find_relevant_entities_tf_df(self, seed, corpus, top_k, candidates_idx):
        try:
            graph_nodes = list(await self.graph.get_nodes())
            corpus = dict({id: (await self.graph.get_node(id))['description'] for id in graph_nodes})
            candidates_idx = list(id for id in graph_nodes)
            index = TFIDFIndex()

            index._build_index_from_list([corpus[_] for _ in candidates_idx])
            idxs = index.query(query_str=seed, top_k=top_k)

            new_candidates_idx = [candidates_idx[_] for _ in idxs]
            cur_contexts = [corpus[_] for _ in new_candidates_idx]

            return cur_contexts, new_candidates_idx

        except Exception as e:
            logger.exception(f"Failed to find relevant entities_vdb: {e}")

    @register_retriever_method(type="entity", method_name="all")
    async def _find_relevant_entities_all(self, key):
        graph_nodes = list(await self.graph.get_nodes())
        corpus = dict({id: (await self.graph.get_node(id))[key] for id in graph_nodes})
        candidates_idx = list(id for id in graph_nodes)
        return corpus, candidates_idx

    @register_retriever_method(type="entity", method_name="link_entity")
    async def _link_entities(self, query_entities):
        # For entity link, we only consider the top-ranked entity
        entities = await asyncio.gather(
            *[self.entities_vdb.retrieval_nodes(query_entity, top_k=1, graph=self.graph) for query_entity in
              query_entities]
        )
        entities = list(map(lambda x: x[0], entities))
        return entities

    @register_retriever_method(type="entity", method_name="get_all")
    async def _get_all_entities(self):
        nodes = await self.graph.nodes_data()
        return nodes

    @register_retriever_method(type="entity", method_name="from_relation_by_agent")
    async def _find_relevant_entities_by_relationships_agent(self, query: str, total_entity_relation_list: list[dict],
                                                             total_relations_dict: defaultdict[list], width=3):
        """
        Use agent to select the top-K relations based on the input query and entities
        Args:
            query: str, the query to be processed.
            total_entity_relation_list: list,  whose element is {"entity": entity_name, "relation": relation, "score": score, "head": bool}
            total_relations_dict: defaultdict[list], key is (src, rel), value is tar
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
            from Core.Prompt.TogPrompt import score_entity_candidates_prompt
            total_candidates = []
            total_scores = []
            total_relations = []
            total_topic_entities = []
            total_head = []

            for index, entity in enumerate(total_entity_relation_list):
                candidate_list = total_relations_dict[(entity["entity"], entity["relation"])]

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

            # entity_prune according to width
            zipped = list(zip(total_relations, total_candidates, total_topic_entities, total_head, total_scores))
            sorted_zipped = sorted(zipped, key=lambda x: x[4], reverse=True)
            relations = list(map(lambda x: x[0], sorted_zipped))[:width]
            candidates = list(map(lambda x: x[1], sorted_zipped))[:width]
            topics = list(map(lambda x: x[2], sorted_zipped))[:width]
            heads = list(map(lambda x: x[3], sorted_zipped))[:width]
            scores = list(map(lambda x: x[4], sorted_zipped))[:width]

            # merge and output
            merged_list = list(zip(relations, candidates, topics, heads, scores))
            filtered_list = [(rel, ent, top, hea, score) for rel, ent, top, hea, score in merged_list if score != 0]
            if len(filtered_list) == 0:
                return False, []
            else:
                return True, filtered_list
        except Exception as e:
            logger.exception(f"Failed to find relevant entities by relation agent: {e}")

    @register_retriever_method(type="entity", method_name="from_relation")
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
        for k, n, d in zip(entity_names, node_datas, node_degrees):
            if "description" not in n:
                n['description'] = ""

            node_datas = [
                {**n, "entity_name": k, "rank": d}
            ]
     
        node_datas = truncate_list_by_token_size(
            node_datas,
            key=lambda x: x["description"],
            max_token_size=self.config.max_token_for_local_context,
        )

        return node_datas

    @register_retriever_method(type="entity", method_name="by_neighbors")
    async def _find_relevant_entities_by_neighbor(self, seed):
        return list(await self.graph.get_neighbors(seed))