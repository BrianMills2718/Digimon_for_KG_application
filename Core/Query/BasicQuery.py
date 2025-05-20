from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Common.Utils import list_to_quoted_csv_string, truncate_list_by_token_size, combine_contexts
from Core.Prompt import QueryPrompt, RaptorPrompt


class BasicQuery(BaseQuery):
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)

    async def _retrieve_relevant_contexts(self, query):
        logger.info(f"BASIC_QUERY_MAIN_RETRIEVE: Entered _retrieve_relevant_contexts for query: '{query}'")
        logger.info(f"BASIC_QUERY_MAIN_RETRIEVE: Config - tree_search: {self.config.tree_search}, use_global_query: {self.config.use_global_query}, use_community: {self.config.use_community}, use_keywords: {self.config.use_keywords}, enable_local: {self.config.enable_local}, enable_hybrid_query: {self.config.enable_hybrid_query}")
        logger.info("BASIC_QUERY_MAIN_RETRIEVE: Checking for tree_search path...")
        if self.config.tree_search:
            logger.info(f"RAPTOR Mode: Retrieving tree nodes with metadata for query: '{query}'")
            tree_nodes_data = await self._retriever.retrieve_relevant_content(
                seed=query,
                tree_node=True, 
                type=Retriever.ENTITY, 
                mode="vdb"
            )

            if not tree_nodes_data:
                logger.warning("RAPTOR Mode: No relevant tree nodes with metadata found.")
                return QueryPrompt.FAIL_RESPONSE 

            logger.info(f"RAPTOR Mode: Retrieved {len(tree_nodes_data)} tree nodes with metadata.")

            # --- START OF NEW RE-RANKING LOGIC ---
            # --- START OF NEW RE-RANKING LOGIC ---
            try:
                max_tree_layer = 0  # Default if graph info is not available
                # Access the graph object from the retriever's context
                # self._retriever.context should have 'graph' (TreeGraphStorage instance)
                # and 'config' (QueryConfig instance for retriever options)
                graph_storage = None
                # Accept both dict-like and attribute access for context
                context_obj = self._retriever.context
                if isinstance(context_obj, dict):
                    graph_storage = context_obj.get('graph')
                else:
                    graph_storage = getattr(context_obj, 'graph', None)

                if graph_storage and hasattr(graph_storage, 'num_layers'):
                    # num_layers is the count of layers. If 1 layer, its index is 0.
                    # If num_layers is 0 (e.g., empty graph), max_tree_layer should be -1 or handled.
                    # Let's ensure num_layers is at least 1 for a valid max_tree_layer index.
                    if graph_storage.num_layers > 0:
                        max_tree_layer = graph_storage.num_layers - 1
                    else:
                        max_tree_layer = 0
                    logger.info(f"RAPTOR Re-ranking: Determined max_tree_layer: {max_tree_layer} (from graph.num_layers: {graph_storage.num_layers})")
                else:
                    logger.warning("RAPTOR Re-ranking: Could not determine max_tree_layer from graph context. Defaulting to 0.")
                    # max_tree_layer remains 0 as initialized

                # Configurable: positive boosts nodes with higher layer numbers (more abstract).
                # Layer 0 is assumed to be the leaf layer. Higher layer numbers are summaries.
                layer_boost_factor = 0.1

                processed_nodes_for_ranking = []
                for i, node in enumerate(tree_nodes_data):
                    try:
                        original_vdb_score = float(node.get("vdb_score", 0.0))
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse vdb_score '{node.get('vdb_score')}' for node {node.get('id')}. Using 0.0.")
                        original_vdb_score = 0.0
                    node_layer = int(node.get("layer", 0))

                    # Re-ranking formula:
                    # Boosts nodes with higher layer numbers (more abstract).
                    # If all nodes are layer 0 (e.g. single-layer tree), node_layer will be 0,
                    # and the boost term (layer_boost_factor * node_layer) will be 0.
                    # So, final_score will equal original_vdb_score. This is expected for single-layer trees.
                    # The re-ranking effect will be visible with multi-layered trees.
                    final_score = original_vdb_score * (1 + (layer_boost_factor * node_layer))

                    node['final_score'] = final_score
                    node['original_vdb_score'] = original_vdb_score
                    processed_nodes_for_ranking.append(node)
                    logger.debug(f"RAPTOR Re-ranking: Node ID {node.get('id')}, Layer {node_layer}, VDB Score {original_vdb_score:.4f}, Final Score {final_score:.4f}")

                tree_nodes_data = sorted(processed_nodes_for_ranking, key=lambda x: x['final_score'], reverse=True)
                logger.info("RAPTOR Re-ranking: Nodes re-ranked based on final_score.")
            except Exception as e:
                logger.exception(f"Error during RAPTOR node re-ranking: {e}. Proceeding with original VDB ranking.")
            # --- END OF NEW RE-RANKING LOGIC ---

            final_context_parts = []
            logger.info(f"RAPTOR Mode: Top {len(tree_nodes_data)} retrieved tree nodes after potential re-ranking:")
            for i, node_info in enumerate(tree_nodes_data):
                node_text_snippet = node_info.get("text", "")[:200] + "..."
                log_msg = (
                    f"RAPTOR Node {i+1}/{len(tree_nodes_data)} "
                    f"ID: {node_info.get('id')}, Layer: {node_info.get('layer')}, "
                    f"VDB Score: {node_info.get('original_vdb_score', 'N/A')}, "
                    f"Final Score: {node_info.get('final_score', 'N/A')}, "
                    f"Content (snippet): {node_text_snippet}"
                )
                logger.info(log_msg)
                final_context_parts.append(node_info.get("text", ""))

            if not final_context_parts:
                return QueryPrompt.FAIL_RESPONSE

            # Use top_k_for_llm from config if available, else fallback to retrieve_top_k
            top_k_for_llm = getattr(self.config, 'retrieve_top_k', len(tree_nodes_data))
            final_context_for_llm = "\n\n---\n\n".join([node.get("text","") for node in tree_nodes_data[:top_k_for_llm]])

            logger.info(f"RAPTOR Mode: Generating answer with retrieved tree context ({len(tree_nodes_data[:top_k_for_llm])} nodes).")
            return final_context_for_llm


        entities_context, relations_context, text_units_context, communities_context = None, None, None, None
        logger.info("BASIC_QUERY_MAIN_RETRIEVE: Checking for global query with community path...")
        if self.config.use_global_query and self.config.use_community:
            return await self._retrieve_relevant_contexts_global(query)
        logger.info("BASIC_QUERY_MAIN_RETRIEVE: Checking for global query with keywords path...")
        if self.config.use_keywords and self.config.use_global_query:
            entities_context, relations_context, text_units_context = await self._retrieve_relevant_contexts_global_keywords(
                query)
        logger.info("BASIC_QUERY_MAIN_RETRIEVE: Checking for local or hybrid query path (this should lead to _retrieve_relevant_contexts_local)...")
        if self.config.enable_local or self.config.enable_hybrid_query: 
            entities_context, relations_context, text_units_context, communities_context = await self._retrieve_relevant_contexts_local(
                query)
        if self.config.enable_hybrid_query:
            hl_entities_context, hl_relations_context, hl_text_units_context = await self._retrieve_relevant_contexts_global_keywords(
                query)
            entities_context, relations_context, text_units_context = combine_contexts(
                entities=[entities_context, hl_entities_context], relationships=[relations_context, hl_relations_context],
                sources=[text_units_context, hl_text_units_context])
        
        if entities_context is None and relations_context is None and text_units_context is None and communities_context is None:
             logger.warning("BASIC_QUERY_MAIN_RETRIEVE: No specific retrieval path matched, proceeding to default/fallback.")
             logger.warning("No context retrieved for non-RAPTOR path.")
             return QueryPrompt.FAIL_RESPONSE

        results = f"""
            -----Entities-----
            ```csv
            {entities_context if entities_context else "N/A"}
            ```
            -----Relationships-----
            ```csv
            {relations_context if relations_context else "N/A"}
            ```
            -----Sources-----
            ```csv
            {text_units_context if text_units_context else "N/A"}
            ```
            """

        if self.config.use_community and communities_context is not None:
            results = f"""
            -----Communities-----
            ```csv
            {communities_context}
            ```
            {results}
            """
        return results

    async def _retrieve_relevant_contexts_local(self, query):
        logger.info(f"BASIC_QUERY_LOCAL: Entered _retrieve_relevant_contexts_local for query: '{query}'")
        use_communities = None  # Initialize to avoid UnboundLocalError
        if self.config.use_keywords:
            query = await self.extract_query_keywords(query)

        node_datas = await self._retriever.retrieve_relevant_content(seed=query, type=Retriever.ENTITY, mode="vdb")
        if node_datas:
            logger.info(f"BASIC_QUERY_LOCAL: Retrieved {len(node_datas)} entities for query '{query}'.")
            logger.debug(f"BASIC_QUERY_LOCAL: First entity object received: {node_datas[0]}")
            if not isinstance(node_datas[0], dict):
                logger.error(f"BASIC_QUERY_LOCAL: First entity is NOT a dict: {type(node_datas[0])}")
            elif "entity_name" not in node_datas[0]:
                logger.error(f"BASIC_QUERY_LOCAL: 'entity_name' key MISSING in first entity. Keys: {list(node_datas[0].keys())}")
            elif not node_datas[0]["entity_name"]:
                logger.warning(f"BASIC_QUERY_LOCAL: 'entity_name' IS EMPTY/None in first entity: {node_datas[0]}")
        else:
            logger.warning(f"BASIC_QUERY_LOCAL: No entities retrieved by VDB for query '{query}'.")

        if self.config.use_community_info:
            use_communities = await self._retriever.retrieve_relevant_content(seed=node_datas, type=Retriever.COMMUNITY,
                                                                              mode="from_entity")
        use_relations = await self._retriever.retrieve_relevant_content(seed=node_datas, type=Retriever.RELATION,
                                                                        mode="from_entity")
        logger.info(f"BASIC_QUERY_LOCAL: Attempting to retrieve text_units using 'entity_occurrence' mode.")
        if not node_datas:
            logger.warning("BASIC_QUERY_LOCAL: Skipping text_unit retrieval as no entities were found.")
        use_text_units = await self._retriever.retrieve_relevant_content(node_datas=node_datas, type=Retriever.CHUNK,
                                                                        mode="entity_occurrence")
        if use_text_units:
            logger.info(f"BASIC_QUERY_LOCAL: Retrieved {len(use_text_units)} text units. First unit (first 50 chars): {use_text_units[0][:50] if use_text_units else 'None'}...")
        else:
            logger.warning(f"BASIC_QUERY_LOCAL: No text units retrieved by 'entity_occurrence'.")
        logger.info(
            f"Using {len(node_datas)} entities, {len(use_relations)} relations, {len(use_text_units)} text units"
        )

        if self.config.use_community_info:
            logger.info(f"Using {len(use_communities)} communities")
        entites_section_list = [["id", "entity", "type", "description", "rank"]]
        for i, n in enumerate(node_datas):
            entites_section_list.append(
                [
                    i,
                    n["entity_name"],
                    n.get("entity_type", "UNKNOWN"),
                    n.get("description", "UNKNOWN"),
                    n["rank"],
                ]
            )
        entities_context = list_to_quoted_csv_string(entites_section_list)

        relations_section_list = [
            ["id", "source", "target", "description", "weight", "rank"]
        ] if not self.config.use_keywords else ["id", "source", "target", "keywords", "description", "weight", "rank"]

        for i, e in enumerate(use_relations):
            row = [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
            ]
            if self.config.use_keywords:
                row.append(e["keywords"])
            row.extend([e["weight"], e["rank"]])
            relations_section_list.append(row)

        relations_context = list_to_quoted_csv_string(relations_section_list)
        communities_context = None
        if self.config.use_community:
            if use_communities:
                communities_section_list = [["id", "content"]]
                for i, c in enumerate(use_communities):
                    communities_section_list.append([i, c["report_string"]])
                communities_context = list_to_quoted_csv_string(communities_section_list)
                logger.info(f"BASIC_QUERY_LOCAL: Formatted {len(use_communities)} communities into context.")
            else:
                logger.warning(f"BASIC_QUERY_LOCAL: 'use_community' is True, but 'use_communities' was not populated (e.g., 'use_communiy_info' might be False or no communities were retrieved). Setting communities_context to None or empty.")
                communities_context = None  # Or: list_to_quoted_csv_string([["id", "content"]])

        text_units_section_list = [["id", "content"]]
        for i, t in enumerate(use_text_units):
            text_units_section_list.append([i, t])
        text_units_context = list_to_quoted_csv_string(text_units_section_list)

        return entities_context, relations_context, text_units_context, communities_context

    async def _retrieve_relevant_contexts_global(self, query):

        community_datas = await self._retriever.retrieve_relevant_content(type=Retriever.COMMUNITY, mode="from_level")
        map_communities_points = await self._map_global_communities(
            query, community_datas
        )
        final_support_points = []
        for i, mc in enumerate(map_communities_points):
            for point in mc:
                if "description" not in point:
                    continue
                final_support_points.append(
                    {
                        "analyst": i,
                        "answer": point["description"],
                        "score": point.get("score", 1),
                    }
                )
        final_support_points = [p for p in final_support_points if p["score"] > 0]
        if not len(final_support_points):
            return QueryPrompt.FAIL_RESPONSE
        final_support_points = sorted(
            final_support_points, key=lambda x: x["score"], reverse=True
        )
        final_support_points = truncate_list_by_token_size(
            final_support_points,
            key=lambda x: x["answer"],
            max_token_size=self.config.global_max_token_for_community_report,
        )
        points_context = []
        for dp in final_support_points:
            points_context.append(
                f"""----Analyst {dp['analyst']}----
    Importance Score: {dp['score']}
    {dp['answer']}
    """
            )
        points_context = "\n".join(points_context)
        return points_context

    async def _retrieve_relevant_contexts_global_keywords(self, query):
        query = await self.extract_query_keywords(query, "high")
        edge_datas = await self._retriever.retrieve_relevant_content(seed=query, type=Retriever.RELATION, mode="vdb")
        use_entities = await self._retriever.retrieve_relevant_content(seed=edge_datas, type=Retriever.ENTITY,
                                                                       mode="from_relation")
        use_text_units = await self._retriever.retrieve_relevant_content(seed=edge_datas, type=Retriever.CHUNK,
                                                                         mode="from_relation")
        logger.info(
            f"Global query uses {len(use_entities)} entities, {len(edge_datas)} relations, {len(use_text_units)} text units"
        )
        relations_section_list = [
            ["id", "source", "target", "description", "keywords", "weight", "rank"]
        ]
        for i, e in enumerate(edge_datas):
            relations_section_list.append(
                [
                    i,
                    e["src_id"],
                    e["tgt_id"],
                    e["description"],
                    e["keywords"],
                    e["weight"],
                    e["rank"],
                ]
            )
        relations_context = list_to_quoted_csv_string(relations_section_list)

        entites_section_list = [["id", "entity", "type", "description", "rank"]]
        for i, n in enumerate(use_entities):
            entites_section_list.append(
                [
                    i,
                    n["entity_name"],
                    n.get("entity_type", "UNKNOWN"),
                    n.get("description", "UNKNOWN"),
                    n["rank"],
                ]
            )
        entities_context = list_to_quoted_csv_string(entites_section_list)

        text_units_section_list = [["id", "content"]]
        for i, t in enumerate(use_text_units):
            text_units_section_list.append([i, t])
        text_units_context = list_to_quoted_csv_string(text_units_section_list)
        return entities_context, relations_context, text_units_context

    async def generation_qa(self, query, context):
        if context is None or context == QueryPrompt.FAIL_RESPONSE:
            logger.warning("generation_qa: No context provided or failed to retrieve context.")
            return QueryPrompt.FAIL_RESPONSE

        if self.config.tree_search: 
            try:
                instruction = RaptorPrompt.ANSWER_QUESTION.format(context=context, question=query)
            except ImportError:
                logger.error("Could not import RaptorPrompt. Using generic prompt.")
                instruction = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            except AttributeError:
                logger.error("ANSWER_QUESTION not found in RaptorPrompt. Using generic prompt.")
                instruction = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
                
            logger.info("RAPTOR Mode: Generating answer with retrieved tree context.")
            response = await self.llm.aask(msg=instruction)
            return response
        
        elif self.config.community_information and self.config.use_global_query: 
            sys_prompt_temp = QueryPrompt.GLOBAL_REDUCE_RAG_RESPONSE
            instruction = query 
            system_msgs = [sys_prompt_temp.format(
                report_data=context, response_type=self.config.response_type
            )]
            logger.info("Global Mode (Communities): Generating answer.")
            response = await self.llm.aask(
                msg=instruction, 
                system_msgs=system_msgs
            )
            return response
        
        else:
            if not self.config.community_information and self.config.use_keywords: 
                sys_prompt_temp = QueryPrompt.RAG_RESPONSE
            elif self.config.community_information and not self.config.use_keywords and self.config.enable_local: 
                sys_prompt_temp = QueryPrompt.LOCAL_RAG_RESPONSE
            else:
                logger.warning(f"generation_qa: Unhandled config combination for prompts. Using default. tree_search={self.config.tree_search}, community_info={self.config.community_information}, use_global_query={self.config.use_global_query}, use_keywords={self.config.use_keywords}, enable_local={self.config.enable_local}")
                instruction = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
                response = await self.llm.aask(msg=instruction)
                return response

            instruction = query 
            system_msgs = [sys_prompt_temp.format(
                context_data=context, response_type=self.config.response_type
            )]
            logger.info("Standard RAG Mode: Generating answer.")
            response = await self.llm.aask(
                msg=instruction, 
                system_msgs=system_msgs
            )
            return response

    async def generation_summary(self, query, context):

        if context is None:
            return QueryPrompt.FAIL_RESPONSE

        if self.config.community_information and self.config.use_global_query: 
            sys_prompt_temp = QueryPrompt.GLOBAL_REDUCE_RAG_RESPONSE
        elif not self.config.community_information and self.config.use_keywords: 
            sys_prompt_temp = QueryPrompt.RAG_RESPONSE
        elif self.config.community_information and not self.config.use_keywords and self.config.enable_local: 
            sys_prompt_temp = QueryPrompt.LOCAL_RAG_RESPONSE
        else:
            logger.error("Invalid query configuration")
            return QueryPrompt.FAIL_RESPONSE
        response = await self.llm.aask(
            query,
            system_msgs=[sys_prompt_temp.format(
                report_data=context, response_type=self.config.response_type
            )],
        )
        return response