from Core.Common.Logger import logger
from Core.Retriever.BaseRetriever import BaseRetriever
import asyncio
import json
from Core.Common.Utils import truncate_list_by_token_size
from collections import Counter
from Core.Retriever.RetrieverFactory import register_retriever_method
from Core.Prompt import QueryPrompt


class CommunityRetriever(BaseRetriever):
    def __init__(self, **kwargs):

        config = kwargs.pop("config")
        super().__init__(config)
        self.mode_list = ["from_entity", "from_level"]
        self.type = "community"
        for key, value in kwargs.items():
            setattr(self, key, value)

    @register_retriever_method(type="community", method_name="from_entity")
    async def _find_relevant_community_from_entities(self, seed: list[dict]):
        logger.info(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Entered with {len(seed) if seed else 0} seed entities.")
        if seed:
            logger.debug(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: First seed entity details: {seed[0]}")
        else:
            logger.warning("COMMUNITY_RETRIEVER_FROM_ENTITIES: Called with no seed entities.")
            return []

        community_reports = self.community.community_reports
        related_communities = []
        for node_d in seed:
            logger.debug(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Processing entity: {node_d.get('entity_name')}, Keys: {list(node_d.keys())}")
            if "clusters" in node_d and node_d["clusters"]:
                cluster_data = node_d["clusters"]
                if isinstance(cluster_data, str):
                    try:
                        logger.warning(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: 'clusters' for entity {node_d.get('entity_name')} is a string. Attempting json.loads.")
                        cluster_data = json.loads(cluster_data)
                    except Exception as e:
                        logger.error(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Error decoding 'clusters' string for entity {node_d.get('entity_name')}: {e}")
                        continue
                if isinstance(cluster_data, list):
                    related_communities.extend(cluster_data)
                    logger.debug(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Extended related_communities from entity {node_d.get('entity_name')} with: {cluster_data}")
                else:
                    logger.warning(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: 'clusters' data for entity {node_d.get('entity_name')} is not a list or decodable string: {cluster_data}")
            else:
                logger.debug(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Entity {node_d.get('entity_name')} has no 'clusters' attribute or it's empty.")
        logger.debug(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: All extracted related_communities: {related_communities}")
        query_config_level = self.retriever_context.context["query_config"].level
        logger.debug(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Using query_config.level: {query_config_level}")
        related_community_dup_keys = [
            str(dp["cluster"])
            for dp in related_communities
            if dp.get("level") <= query_config_level
        ]
        logger.info(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Found {len(set(related_community_dup_keys))} unique community keys from entities. Keys: {list(set(related_community_dup_keys))}")
        from collections import Counter
        related_community_keys_counts = dict(Counter(related_community_dup_keys))
        logger.debug(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Community key counts: {related_community_keys_counts}")
        _related_community_datas = await asyncio.gather(
            *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
        )
        logger.debug(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Data retrieved from community_reports storage for keys {list(related_community_keys_counts.keys())}: {_related_community_datas}")
        logger.info(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Retrieved {len(_related_community_datas) if _related_community_datas else 0} community data objects from storage.")
        if _related_community_datas:
            logger.debug(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: First retrieved community data object (raw): {_related_community_datas[0]}")
        related_community_datas = {
            k: v
            for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
            if v is not None
        }
        logger.info(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Filtered to {len(related_community_datas)} non-None community data objects.")
        related_community_keys = sorted(
            related_community_keys_counts.keys(),
            key=lambda k: (
                related_community_keys_counts[k],
                related_community_datas[k]["report_json"].get("rating", -1) if k in related_community_datas else -1,
            ),
            reverse=True,
        )
        sorted_community_datas = [
            related_community_datas[k] for k in related_community_keys if k in related_community_datas
        ]
        logger.info(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Sorted {len(sorted_community_datas)} community datas.")
        if sorted_community_datas:
            logger.debug(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: First sorted community data (before truncation): {sorted_community_datas[0]}")

        use_community_reports = truncate_list_by_token_size(
            sorted_community_datas,
            key=lambda x: x["report_string"],
            max_token_size=self.retriever_context.context["query_config"].local_max_token_for_community_report,
        )
        logger.info(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: Returning {len(use_community_reports) if use_community_reports else 0} community reports after truncation.")
        if use_community_reports:
            logger.debug(f"COMMUNITY_RETRIEVER_FROM_ENTITIES: First returned report (sample): {str(use_community_reports[0])[:200]}")
        if self.retriever_context.context["query_config"].local_community_single_one:
            use_community_reports = use_community_reports[:1]
        return use_community_reports

    @register_retriever_method(type="community", method_name="from_level")
    async def find_relevant_community_by_level(self, seed=None):
        community_schema = self.community.community_schema
        community_schema = {
            k: v for k, v in community_schema.items() if v.level <= self.config.level
        }
        if not len(community_schema):
            return QueryPrompt.FAIL_RESPONSE

        sorted_community_schemas = sorted(
            community_schema.items(),
            key=lambda x: x[1].occurrence,
            reverse=True,
        )

        sorted_community_schemas = sorted_community_schemas[
                                   : self.config.global_max_consider_community
                                   ]
        community_datas = await self.community.community_reports.get_by_ids(  ###
            [k[0] for k in sorted_community_schemas]
        )

        community_datas = [c for c in community_datas if c is not None]
        community_datas = [
            c
            for c in community_datas
            if c["report_json"].get("rating", 0) >= self.config.global_min_community_rating
        ]
        community_datas = sorted(
            community_datas,
            key=lambda x: (x['community_info']['occurrence'], x["report_json"].get("rating", 0)),
            reverse=True,
        )
        logger.info(f"Retrieved {len(community_datas)} communities")
        return community_datas
