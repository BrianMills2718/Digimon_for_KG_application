import re
import asyncio
import json
from collections import defaultdict, Counter
from typing import Union, Any
from Core.Graph.BaseGraph import BaseGraph
from Core.Common.Logger import logger
from Core.Common.Utils import (
    clean_str,
    split_string_by_multi_markers,
    is_float_regex,
    list_to_quoted_csv_string,
    prase_json_from_response,
    truncate_list_by_token_size,
    mdhash_id
)
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.Message import Message
from Core.Prompt import EntityPrompt
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Storage.JsonKVStorage import JsonKVStorage
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Common.Constants import (
    GRAPH_FIELD_SEP, 
    DEFAULT_ENTITY_TYPES, 
    DEFAULT_RECORD_DELIMITER, 
    DEFAULT_COMPLETION_DELIMITER,
    DEFAULT_TUPLE_DELIMITER
)
from Core.Common.Memory import Memory
from Core.Community.ClusterFactory import get_community_instance
from Core.Common.QueryConfig import QueryConfig
from Core.Prompt import QueryPrompt
from Core.Common.QueryConfig import query_config
from pydantic import model_validator
from Core.Index import (
    get_rag_embedding
)    
from Core.Index.Schema import (
    FAISSIndexConfig
)
from Core.Index.VectorIndex import VectorIndex
class ERGraph(BaseGraph):
   
    text_chunks: JsonKVStorage = JsonKVStorage()
    er_graph: NetworkXStorage = NetworkXStorage()
    
    @model_validator(mode="after")
    def _init_vectordb(cls, data):
        index_config = FAISSIndexConfig(persist_path="./storage", embed_model = get_rag_embedding())
        cls.entity_vdb = VectorIndex(index_config)
        cls.relationship_vdb = VectorIndex(index_config)
        return data

    async def _construct_graph(self, chunks: dict[str, TextChunk]):
        try:
            filtered_keys = await self.text_chunks.filter_keys(list(chunks.keys()))
            inserting_chunks = {key: value for key, value in chunks.items() if key in filtered_keys}
            ordered_chunks = list(inserting_chunks.items())
            async def _process_single_content(chunk_key_dp: tuple[str, TextChunk]):
                chunk_key, chunk_dp = chunk_key_dp
                records = await self._extract_records_from_chunk(chunk_dp)
                return await self._organize_records(records, chunk_key)

            results = await asyncio.gather(*[_process_single_content(c) for c in ordered_chunks])
            await self._update_graph(results)
            if self.config.use_community:
                # ---------- update clusterings of graph
                logger.info("Starting [Community Report]")
                community_ins = get_community_instance(self.config.graph_cluster_algorithm, enforce_sub_communities = self.config.enforce_sub_communities, llm = self.llm)
                cluster_node_map = await community_ins._clustering_(self.er_graph.graph, self.config.max_graph_cluster_size, self.config.graph_cluster_seed)
                self.er_graph._cluster_data_to_subgraphs(cluster_node_map)
                await community_ins._generate_community_report_(self.er_graph)
                logger.info("[Community Report]  Finished")
          
        
            # # ---------- commit upsertings and indexing
            # await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
            self.query_config = query_config
                # await self.global_query("who are you", community_ins)
            await self.local_query("who are you", community_ins)
        finally:
            logger.info("Consturcting graph finisihed")

    
    async def query(self, query: str, param: QueryConfig):
        if param.mode == "local" and not self.config.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.config.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode == "local":
            response = await self.local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param
            )
        elif param.mode == "global":
            response = await self.global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        # await self._query_done()
        return response
    async def local_query(self, query:str, community_instance):
        context = await self._build_local_query_context(
            query,
            community_instance.community_reports
        )
        if self.config.only_need_context:
            return context
        if context is None:
            return QueryPrompt.FAIL_RESPONSE
        sys_prompt_temp = QueryPrompt.LOCAL_RAG_RESPONSE
        sys_prompt = sys_prompt_temp.format(
            context_data=context, response_type = self.query_config.response_type
        )
        response = await self.llm.aask(
            query,
            system_msgs = [sys_prompt]
        )
        return response
    
    async def _build_local_query_context(self, query,  community_reports):
        results = await self.entity_vdb.retrieval(query, top_k=self.query_config.top_k)
        import pdb
        pdb.set_trace()
        if not len(results):
            return None
        node_datas = await asyncio.gather(
            *[self.er_graph.get_node(r.metadata["entity_name"]) for r in results]
        )
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
        node_degrees = await asyncio.gather(
            *[self.er_graph.node_degree(r.metadata["entity_name"]) for r in results]
        )
        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(results, node_datas, node_degrees)
            if n is not None
        ]
        use_communities = await self._find_most_related_community_from_entities(node_datas, community_reports)
        use_text_units = await self._find_most_related_text_unit_from_entities(node_datas)
        use_relations = await self._find_most_related_edges_from_entities(node_datas)
        logger.info(
            f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
        )
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
        ]
        for i, e in enumerate(use_relations):
            relations_section_list.append(
                [
                    i,
                    e["src_tgt"][0],
                    e["src_tgt"][1],
                    e["description"],
                    e["weight"],
                    e["rank"],
                ]
            )
        relations_context = list_to_quoted_csv_string(relations_section_list)

        communities_section_list = [["id", "content"]]
        for i, c in enumerate(use_communities):
            communities_section_list.append([i, c["report_string"]])
        communities_context = list_to_quoted_csv_string(communities_section_list)

        text_units_section_list = [["id", "content"]]
        for i, t in enumerate(use_text_units):
            text_units_section_list.append([i, t["content"]])
        text_units_context = list_to_quoted_csv_string(text_units_section_list)
        return f"""
            -----Reports-----
            ```csv
            {communities_context}
            ```
            -----Entities-----
            ```csv
            {entities_context}
            ```
            -----Relationships-----
            ```csv
            {relations_context}
            ```
            -----Sources-----
            ```csv
            {text_units_context}
            ```
            """
    async def _find_most_related_community_from_entities(self, node_datas: list[dict], community_reports):
        related_communities = []
        for node_d in node_datas:
            if "clusters" not in node_d:
                continue
            related_communities.extend(json.loads(node_d["clusters"]))
        related_community_dup_keys = [
            str(dp["cluster"])
            for dp in related_communities
            if dp["level"] <= self.query_config.level
        ]
        related_community_keys_counts = dict(Counter(related_community_dup_keys))
        _related_community_datas = await asyncio.gather(
            *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
        )
        related_community_datas = {
            k: v
            for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
            if v is not None
        }
        related_community_keys = sorted(
            related_community_keys_counts.keys(),
            key=lambda k: (
                related_community_keys_counts[k],
                related_community_datas[k]["report_json"].get("rating", -1),
            ),
            reverse=True,
        )
        sorted_community_datas = [
            related_community_datas[k] for k in related_community_keys
        ]

        use_community_reports = truncate_list_by_token_size(
            sorted_community_datas,
            key=lambda x: x["report_string"],
            max_token_size= self.query_config.local_max_token_for_community_report,
        )
        if self.query_config.local_community_single_one:
            use_community_reports = use_community_reports[:1]
        return use_community_reports
    async def _find_most_related_text_unit_from_entities(self, node_datas: list[dict]):
        text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
            for dp in node_datas
        ]
        edges = await asyncio.gather(
            *[self.er_graph.get_node_edges(dp["entity_name"]) for dp in node_datas]
        )
        all_one_hop_nodes = set()
        for this_edges in edges:
            if not this_edges:
                continue
            all_one_hop_nodes.update([e[1] for e in this_edges])
        all_one_hop_nodes = list(all_one_hop_nodes)
        all_one_hop_nodes_data = await asyncio.gather(
            *[self.er_grap.get_node(e) for e in all_one_hop_nodes]
        )
        all_one_hop_text_units_lookup = {
            k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
            for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
            if v is not None
        }
        all_text_units_lookup = {}
        for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
            for c_id in this_text_units:
                if c_id in all_text_units_lookup:
                    continue
                relation_counts = 0
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1
                all_text_units_lookup[c_id] = {
                    "data": await self.text_chunks.get_by_id(c_id),
                    "order": index,
                    "relation_counts": relation_counts,
                }
        if any([v is None for v in all_text_units_lookup.values()]):
            logger.warning("Text chunks are missing, maybe the storage is damaged")
        all_text_units = [
            {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
        ]
        all_text_units = sorted(
            all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
        )
        all_text_units = truncate_list_by_token_size(
            all_text_units,
            key=lambda x: x["data"]["content"],
            max_token_size = self.query_config.local_max_token_for_text_unit,
        )
        all_text_units = [t["data"] for t in all_text_units]
        return all_text_units
    
    async def _find_most_related_edges_from_entities(self, node_datas: list[dict]):
        all_related_edges = await asyncio.gather(
            *[self.er_graph.get_node_edges(dp["entity_name"]) for dp in node_datas]
        )
        all_edges = set()
        for this_edges in all_related_edges:
            all_edges.update([tuple(sorted(e)) for e in this_edges])
        all_edges = list(all_edges)
        all_edges_pack = await asyncio.gather(
            *[self.er_graph.get_edge(e[0], e[1]) for e in all_edges]
        )
        all_edges_degree = await asyncio.gather(
            *[self.er_graph.edge_degree(e[0], e[1]) for e in all_edges]
        )
        all_edges_data = [
            {"src_tgt": k, "rank": d, **v}
            for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
            if v is not None
        ]
        all_edges_data = sorted(
            all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        all_edges_data = truncate_list_by_token_size(
            all_edges_data,
            key=lambda x: x["description"],
            max_token_size= self.query_config.max_token_for_local_context,
        )
        return all_edges_data
    async def _map_global_communities(
            self,
            query: str,
            communities_data
        ):
            
            community_groups = []
            while len(communities_data):
                this_group = truncate_list_by_token_size(
                    communities_data,
                    key=lambda x: x["report_string"],
                    max_token_size=self.query_config.global_max_token_for_community_report,
                )
                community_groups.append(this_group)
                communities_data = communities_data[len(this_group) :]

            async def _process(community_truncated_datas: list[Any]) -> dict:
                communities_section_list = [["id", "content", "rating", "importance"]]
                for i, c in enumerate(community_truncated_datas):
                    communities_section_list.append(
                        [
                            i,
                            c["report_string"],
                            c["report_json"].get("rating", 0),
                            c['community_info'].occurrence,
                        ]
                    )
                community_context = list_to_quoted_csv_string(communities_section_list)
                sys_prompt_temp = QueryPrompt.GLOBAL_MAP_RAG_POINTS
                sys_prompt = sys_prompt_temp.format(context_data=community_context)
       
                response = await self.llm.aask(
                    query,
                    system_msgs = [sys_prompt]
                )

                data = prase_json_from_response(response)
                return data.get("points", [])

            logger.info(f"Grouping to {len(community_groups)} groups for global search")
            responses = await asyncio.gather(*[_process(c) for c in community_groups])
            return responses

    async def global_query(
            self,
            query,
            community_instance
        ) -> str:
            community_schema = community_instance.community_schema
            community_schema = {
                k: v for k, v in community_schema.items() if v.level <= self.query_config.level
            }
            if not len(community_schema):
                return QueryPrompt.FAIL_RESPONSE

            sorted_community_schemas = sorted(
                community_schema.items(),
                key=lambda x: x[1].occurrence,
                reverse=True,
            )
     
            sorted_community_schemas = sorted_community_schemas[
                : self.query_config.global_max_consider_community
            ]
            community_datas = await community_instance.community_reports.get_by_ids( ###
                [k[0] for k in sorted_community_schemas]
            )
      
            community_datas = [c for c in community_datas if c is not None]
            community_datas = [
                c
                for c in community_datas
                if c["report_json"].get("rating", 0) >= self.query_config.global_min_community_rating
            ]
            community_datas = sorted(
                community_datas,
                key=lambda x: (x['community_info'].occurrence, x["report_json"].get("rating", 0)),
                reverse=True,
            )
            logger.info(f"Revtrieved {len(community_datas)} communities")

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
                max_token_size=self.query_config.global_max_token_for_community_report,
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
            if self.query_config.only_need_context:
                return points_context
            sys_prompt_temp = QueryPrompt.GLOBAL_REDUCE_RAG_RESPONSE
            response = await self.llm.aask(
                query,
                system_msgs= [sys_prompt_temp.format(
                    report_data=points_context, response_type=self.query_config.response_type
                )],
            )                
            return response
    async def _extract_records_from_chunk(self, chunk_dp: TextChunk):
        context = self._build_context_for_entity_extraction(chunk_dp["content"])
        prompt_template = None
        if self.config.use_keywords:
            prompt_template = EntityPrompt.ENTITY_EXTRACTION_KEYWORD
        else:
            prompt_template = EntityPrompt.ENTITY_EXTRACTION
        prompt = prompt_template.format(**context)
        
        working_memory = Memory()

        working_memory.add(Message(content = prompt, role = "user"))
        final_result = await self.llm.aask(prompt)
        working_memory.add(Message(content = final_result, role = "assistant"))

        for glean_idx in range(self.config.max_gleaning):
            working_memory.add(Message(content=EntityPrompt.ENTITY_CONTINUE_EXTRACTION, role="user"))
            glean_result = await self.llm.aask(working_memory.get())
            working_memory.add(Message(content=glean_result, role="assistant"))
            final_result += glean_result

            if glean_idx == self.config.max_gleaning - 1:
                break

            working_memory.add(Message(content=EntityPrompt.ENTITY_IF_LOOP_EXTRACTION, role="user"))
            if_loop_result = await self.llm.aask(working_memory.get())
            if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
                break
        working_memory.clear()
        return split_string_by_multi_markers(final_result, [
            DEFAULT_RECORD_DELIMITER, DEFAULT_COMPLETION_DELIMITER
        ])

    async def _organize_records(self, records: list[str], chunk_key: str):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        for record in records:
            match = re.search(r"\((.*)\)", record)
            if match is None:
                continue

            record_attributes = split_string_by_multi_markers(match.group(1), [DEFAULT_TUPLE_DELIMITER])
            entity = await self._handle_single_entity_extraction(record_attributes, chunk_key)

            if entity is not None:
                maybe_nodes[entity.entity_name].append(entity)
                continue
       
            relationship = await self._handle_single_relationship_extraction(record_attributes, chunk_key)
         
            if relationship is not None:
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        return dict(maybe_nodes), dict(maybe_edges)

    async def _handle_single_entity_extraction(self, record_attributes: list[str], chunk_key: str) -> Union[Entity, None]:
        if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
            return None

        entity_name = clean_str(record_attributes[1].upper())
        if not entity_name.strip():
            return None

        return Entity(
            entity_name = entity_name,
            entity_type = clean_str(record_attributes[2].upper()),
            description = clean_str(record_attributes[3]),
            source_id = chunk_key
        )

    async def _handle_single_relationship_extraction(self, record_attributes: list[str], chunk_key: str) -> Union[Relationship, None]:
        if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
            return None

        return Relationship(
            src_id=clean_str(record_attributes[1].upper()),
            tgt_id=clean_str(record_attributes[2].upper()),
            weight=float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0,
            description=clean_str(record_attributes[3]),
            source_id=chunk_key,
            keywords = clean_str(record_attributes[4]) if self.config.use_keywords else None
        )

    async def _update_graph(self, results: list, use_keywords: bool = False):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)
        for m_nodes, m_edges in results:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)
        import pdb
        pdb.set_trace()
        entities = await asyncio.gather(*[self._merge_nodes_then_upsert(k, v) for k, v in maybe_nodes.items()])
        if self.entity_vdb is not None:
            data_for_vdb = {
                mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + dp["description"],
                    "entity_name": dp["entity_name"],
                }
                for dp in entities
            }
            await self.entity_vdb.upsert(data_for_vdb)
        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])

    async def _merge_nodes_then_upsert(self, entity_name: str, nodes_data: list[Entity]):
        existing_node = await self.er_graph.get_node(entity_name)
        existing_data = [[],[],[]] if existing_node is None else [
            existing_node["entity_type"],
            split_string_by_multi_markers(existing_node["source_id"], [GRAPH_FIELD_SEP]),
            existing_node["description"]
        ]

        entity_type = sorted(
            Counter(dp.entity_type for dp in nodes_data + existing_data[0]).items(),
            key=lambda x: x[1], reverse=True
        )[0][0]

        description = GRAPH_FIELD_SEP.join(
            sorted(set(dp.description for dp in nodes_data) | set(existing_data[2]))
        )
        source_id = GRAPH_FIELD_SEP.join(
            set(dp.source_id for dp in nodes_data) | set(existing_data[1])
        )
        import pdb
        pdb.set_trace()
        description = await self._handle_entity_relation_summary(entity_name, description)

        node_data = dict(entity_type=entity_type, description=description, source_id=source_id)
 
        await self.er_graph.upsert_node(entity_name, node_data=node_data)
        return {**node_data, "entity_name": entity_name}


    async def _merge_edges_then_upsert(self, src_id: str, tgt_id: str, edges_data: list[Relationship]):
        existing_edge_data = {}

        if await self.er_graph.has_edge(src_id, tgt_id):
            existing_edge_data = await self.er_graph.get_edge(src_id, tgt_id)

        #NOTE: For the nano-rag, it supports DSpy 
        weight = sum(dp.weight for dp in edges_data) + existing_edge_data.get("weight", 0)

        description = GRAPH_FIELD_SEP.join(
            sorted(set(dp.description for dp in edges_data) | {existing_edge_data.get("description", "")})
        )
        source_id = GRAPH_FIELD_SEP.join(
            set([dp.source_id for dp in edges_data] + split_string_by_multi_markers(existing_edge_data.get("source_id", ""), [GRAPH_FIELD_SEP])
        ))
        if self.config.use_keywords:
            keywords = GRAPH_FIELD_SEP.join(
                sorted(set([dp.keywords for dp in edges_data] + split_string_by_multi_markers(existing_edge_data.get("keywords", ""), [GRAPH_FIELD_SEP])
            )))
            
        for need_insert_id in (src_id, tgt_id):
            if not await self.er_graph.has_node(need_insert_id):
                await self.er_graph.upsert_node(
                    need_insert_id,
                    node_data=dict(source_id=source_id, description=description, entity_type='"UNKNOWN"')
                )

        description = await self._handle_entity_relation_summary((src_id, tgt_id), description)

        await self.er_graph.upsert_edge(src_id, tgt_id, edge_data=dict(weight=weight, description=description, source_id=source_id, keywords = keywords))

    async def _handle_entity_relation_summary(self, entity_or_relation_name: str, description: str) -> str:

        tokens = self.ENCODER.encode(description)
        if len(tokens) < self.config.summary_max_tokens:
            return description

        use_description = self.ENCODER.decode(tokens[:self.llm.get_maxtokens()])
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=use_description.split(GRAPH_FIELD_SEP)
        )
        use_prompt = EntityPrompt.SUMMARIZE_ENTITY_DESCRIPTIONS.format(**context_base)
        logger.debug(f"Trigger summary: {entity_or_relation_name}")
        return await self.llm.aask(use_prompt, max_tokens = self.config.summary_max_tokens)
        

    def _build_context_for_entity_extraction(self, content: str) -> dict:
        return dict(
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter = DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_text=content
        )

    # 


    async def local_query_lightrag(
        self,
        query
    ) -> str:
        context = None
        # use_model_func = global_config["llm_model_func"]
        kw_prompt = QueryPrompt.KEYWORDS_EXTRACTION.format(query=query)
        result = await self.llm.aask(kw_prompt)

    
        keywords_data = prase_json_from_response(result)
        keywords = keywords_data.get("low_level_keywords", [])
        keywords = ", ".join(keywords)
   
        if keywords:
            context = await self._build_local_query_context_with_keywords(keywords)
        if self.query_config.only_need_context:
            return context
        if context is None:
            return QueryPrompt.FAIL_RESPONSE
        sys_prompt_temp = QueryPrompt.RAG_RESPONSE
        sys_prompt = sys_prompt_temp.format(
            context_data=context, response_type= self.query_config.response_type
        )
        response = await self.llm.aask(
            query,
            system_msgs=[sys_prompt]
        )
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        return response
    
    async def global_query_lightrag(self, query):
        context = None
        kw_prompt = QueryPrompt.KEYWORDS_EXTRACTION.format(query=query)
        result = await self.llm.aask(kw_prompt)

        
        keywords_data = prase_json_from_response(result)
        keywords = keywords_data.get("high_level_keywords", [])
        keywords = ", ".join(keywords)
       
        if keywords:
            context = await self._build_global_query_context(keywords)

        if self.query_config.only_need_context:
            return context
        if context is None:
            return QueryPrompt.FAIL_RESPONSE

        sys_prompt_temp = QueryPrompt.RAG_RESPONSE
        sys_prompt = sys_prompt_temp.format(
            context_data=context, response_type=self.query_config.response_type
        )
        response = await self.llm.aask(
            query,
            system_msgs=[sys_prompt]
        )
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        return response

    def _extract_node(self):
        pass

    def _extract_relationship(self):
        pass

    def _exist_graph(self):
        pass


    async def _build_local_query_context_with_keywords(self, query):
        results = await self.entity_vdb.retrieval(query, top_k=self.query_config.top_k)

        if not len(results):
            return None
        node_datas = await asyncio.gather(
            *[self.er_graph.get_node(r["entity_name"]) for r in results]
        )
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
        node_degrees = await asyncio.gather(
            *[self.er_graph.node_degree(r["entity_name"]) for r in results]
        )
        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(results, node_datas, node_degrees)
            if n is not None
        ]  
        # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
        use_text_units = await self._find_most_related_text_unit_from_entities(node_datas)
        use_relations = await self._find_most_related_edges_from_entities(node_datas)
        logger.info(
            f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
        )
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
            ["id", "source", "target", "description", "keywords", "weight", "rank"]
        ]
        for i, e in enumerate(use_relations):
            relations_section_list.append(
                [
                    i,
                    e["src_tgt"][0],
                    e["src_tgt"][1],
                    e["description"],
                    e["keywords"],
                    e["weight"],
                    e["rank"],
                ]
            )
        relations_context = list_to_quoted_csv_string(relations_section_list)

        text_units_section_list = [["id", "content"]]
        for i, t in enumerate(use_text_units):
            text_units_section_list.append([i, t["content"]])
        text_units_context = list_to_quoted_csv_string(text_units_section_list)
        return f"""
            -----Entities-----
            ```csv
            {entities_context}
            ```
            -----Relationships-----
            ```csv
            {relations_context}
            ```
            -----Sources-----
            ```csv
            {text_units_context}
            ```
        """
    
    async def _build_global_query_context(self, keywords):
        results = await self.relationship_vdb.retrieval(keywords, top_k=self.query_config.top_k)

        if not len(results): 
            return None

        edge_datas = await asyncio.gather(
            *[self.er_graph.get_edge(r["src_id"], r["tgt_id"]) for r in results]
        )

        if not all([n is not None for n in edge_datas]):
            logger.warning("Some edges are missing, maybe the storage is damaged")
        edge_degree = await asyncio.gather(
            *[self.er_graph.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
        )
        edge_datas = [
            {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
            for k, v, d in zip(results, edge_datas, edge_degree)
            if v is not None
        ]
        edge_datas = sorted(
            edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        edge_datas = truncate_list_by_token_size(
            edge_datas,
            key=lambda x: x["description"],
            max_token_size = self.query_config.max_token_for_global_context,
        )

        use_entities = await self._find_most_related_entities_from_relationships(edge_datas)
        use_text_units = await self._find_related_text_unit_from_relationships(edge_datas)
        logger.info(
            f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
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
            text_units_section_list.append([i, t["content"]])
        text_units_context = list_to_quoted_csv_string(text_units_section_list)

        return f"""
    -----Entities-----
    ```csv
    {entities_context}
    ```
    -----Relationships-----
    ```csv
    {relations_context}
    ```
    -----Sources-----
    ```csv
    {text_units_context}
    ```
    """

    async def _find_most_related_entities_from_relationships(self, edge_datas: list[dict]):
        entity_names = set()
        for e in edge_datas:
            entity_names.add(e["src_id"])
            entity_names.add(e["tgt_id"])

        node_datas = await asyncio.gather(
            *[self.er_graph.get_node(entity_name) for entity_name in entity_names]
        )

        node_degrees = await asyncio.gather(
            *[self.er_graph.node_degree(entity_name) for entity_name in entity_names]
        )
        node_datas = [
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(entity_names, node_datas, node_degrees)
        ]

        node_datas = truncate_list_by_token_size(
            node_datas,
            key=lambda x: x["description"],
            max_token_size = self.query_config.max_token_for_local_context,
        )

        return node_datas


    async def _find_related_text_unit_from_relationships(
        self,
        edge_datas: list[dict]
    ):
        text_units = [
            split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
            for dp in edge_datas
        ]

        all_text_units_lookup = {}

        for index, unit_list in enumerate(text_units):
            for c_id in unit_list:
                if c_id not in all_text_units_lookup:
                    all_text_units_lookup[c_id] = {
                        "data": await self.text_chunks.get_by_id(c_id),
                        "order": index,
                    }

        if any([v is None for v in all_text_units_lookup.values()]):
            logger.warning("Text chunks are missing, maybe the storage is damaged")
        all_text_units = [
            {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
        ]
        all_text_units = sorted(all_text_units, key=lambda x: x["order"])
        all_text_units = truncate_list_by_token_size(
            all_text_units,
            key=lambda x: x["data"]["content"],
            max_token_size = self.query_config.max_token_for_text_unit,
        )
        all_text_units = [t["data"] for t in all_text_units]

        return all_text_units
    

    async def hybrid_query(self, query):
        low_level_context = None
        high_level_context = None

        kw_prompt_temp = QueryPrompt.KEYWORDS_EXTRACTION
        kw_prompt = kw_prompt_temp.format(query=query)

        result = await self.llm.aask(kw_prompt)
        json_text = prase_json_from_response(result)
        try:
            keywords_data = json.loads(json_text)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        except json.JSONDecodeError:
            try:
                result = (
                    result.replace(kw_prompt[:-1], "")
                    .replace("user", "")
                    .replace("model", "")
                    .strip()
                )
                result = "{" + result.split("{")[1].split("}")[0] + "}"
                keywords_data = json.loads(result)
                hl_keywords = keywords_data.get("high_level_keywords", [])
                ll_keywords = keywords_data.get("low_level_keywords", [])
                hl_keywords = ", ".join(hl_keywords)
                ll_keywords = ", ".join(ll_keywords)
            # Handle parsing error
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return QueryPrompt.FAIL_RESPONSE

        if ll_keywords:
            low_level_context = await self._build_local_query_context(
                ll_keywords
            
            )

        if hl_keywords:
            high_level_context = await self._build_global_query_context(
                hl_keywords
             
            )

        context = self.combine_contexts(high_level_context, low_level_context)

        if self.query_config.only_need_context:
            return context
        if context is None:
            return QueryPrompt.FAIL_RESPONSE


        sys_prompt_temp = QueryPrompt.RAG_RESPONSE
        sys_prompt = sys_prompt_temp.format(
            context_data=context, response_type=self.query_config.response_type
        )
        response = await self.llm.aask(
            query,
            system_msgs=[sys_prompt]
        )
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )
        return response