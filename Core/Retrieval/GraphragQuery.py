import asyncio
from GraphRAG.Core.Query.BaseQuery import CustomQuery
from Config2 import config
from QueryConfig import QueryConfig
from Core.Common.Utils import truncate_list_by_token_size, list_to_quoted_csv_string, prase_json_from_response
from Core.Prompt import QueryPrompt
from Core.Common.Logger import logger


class GraphragQuery(CustomQuery):
    def __init__(self, community_report):
        super(GraphragQuery,self).__init__()
        self.community_report = community_report

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
                        c["occurrence"],
                    ]
                )
            community_context = list_to_quoted_csv_string(communities_section_list)
            sys_prompt_temp = QueryPrompt.GLOBAL_MAP_RAG_POINTS
            sys_prompt = sys_prompt_temp.format(context_data=community_context)
            response = await self.llm.aask(
                query,
                system_prompt=sys_prompt,
                **self.query_config.global_special_community_map_llm_kwargs,
            )
            data = prase_json_from_response(response)
            return data.get("points", [])

        logger.info(f"Grouping to {len(community_groups)} groups for global search")
        responses = await asyncio.gather(*[_process(c) for c in community_groups])
        return responses

    async def global_query(
        self,
        query,
        community_reports: CommunitySchema,
    ) -> str:
        community_schema = await community_reports._community_schema_()
        community_schema = {
            k: v for k, v in community_schema.items() if v["level"] <= self.query_config.level
        }
        if not len(community_schema):
            return QueryPrompt.FAIL_RESPONSE

        sorted_community_schemas = sorted(
            community_schema.items(),
            key=lambda x: x[1]["occurrence"],
            reverse=True,
        )
        sorted_community_schemas = sorted_community_schemas[
            : self.query_config.global_max_consider_community
        ]
        community_datas = await community_reports.get_by_ids( ###
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
            key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
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
            sys_prompt_temp.format(
                report_data=points_context, response_type=self.query_config.response_type
            ),
        )
        return response