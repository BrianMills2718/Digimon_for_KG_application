from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Common.Utils import list_to_quoted_csv_string, truncate_list_by_token_size, combine_contexts
from Core.Prompt import QueryPrompt, RaptorPrompt


class BasicQuery(BaseQuery):
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)

    async def _retrieve_relevant_contexts(self, query):
        if self.config.tree_search:
            logger.info(f"RAPTOR Mode: Retrieving tree nodes for query: '{query}'")
            tree_nodes_texts = await self._retriever.retrieve_relevant_content(
                seed=query,
                tree_node=True, 
                type=Retriever.ENTITY, 
                mode="vdb"
            )

            if not tree_nodes_texts:
                logger.warning("RAPTOR Mode: No relevant tree nodes found.")
                return QueryPrompt.FAIL_RESPONSE 

            logger.info(f"RAPTOR Mode: Retrieved {len(tree_nodes_texts)} tree node texts.")
            for i, node_text in enumerate(tree_nodes_texts):
                logger.info(f"RAPTOR Node {i+1}/{len(tree_nodes_texts)} Content (snippet): {node_text[:150].replace(chr(10), ' ')}...")

            context_string = "\n\n---\n\n".join(tree_nodes_texts) 
            return context_string

        entities_context, relations_context, text_units_context, communities_context = None, None, None, None
        if self.config.use_global_query and self.config.use_community:
            return await self._retrieve_relevant_contexts_global(query)
        if self.config.use_keywords and self.config.use_global_query:
            entities_context, relations_context, text_units_context = await self._retrieve_relevant_contexts_global_keywords(
                query)
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
        if self.config.use_keywords:
            query = await self.extract_query_keywords(query)

        node_datas = await self._retriever.retrieve_relevant_content(seed=query, type=Retriever.ENTITY, mode="vdb")

        if self.config.use_communiy_info:
            use_communities = await self._retriever.retrieve_relevant_content(seed=node_datas, type=Retriever.COMMUNITY,
                                                                              mode="from_entity")
        use_relations = await self._retriever.retrieve_relevant_content(seed=node_datas, type=Retriever.RELATION,
                                                                        mode="from_entity")
        use_text_units = await self._retriever.retrieve_relevant_content(node_datas=node_datas, type=Retriever.CHUNK,
                                                                         mode="entity_occurrence")
        logger.info(
            f"Using {len(node_datas)} entities, {len(use_relations)} relations, {len(use_text_units)} text units"
        )

        if self.config.use_communiy_info:
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
            communities_section_list = [["id", "content"]]
            for i, c in enumerate(use_communities):
                communities_section_list.append([i, c["report_string"]])
            communities_context = list_to_quoted_csv_string(communities_section_list)

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