"""Leiden community clustering/report generation for DIGIMON graphs."""

import asyncio
from collections import defaultdict

from graspologic.partition import hierarchical_leiden

from Core.Common.EntityNormalization import normalize_entity_id
from Core.Common.Logger import logger
from Core.Common.Utils import (
    community_report_from_json,
    encode_string_by_tiktoken,
    list_to_quoted_csv_string,
    truncate_list_by_token_size,
)
from Core.Community.BaseCommunity import BaseCommunity
from Core.Community.ClusterFactory import register_community
from Core.Graph.BaseGraph import BaseGraph
from Core.Prompt import CommunityPrompt
from Core.Schema.CommunitySchema import CommunityReportsResult, LeidenInfo
from Core.Storage.JsonKVStorage import JsonKVStorage


@register_community(name="leiden")
class LeidenCommunity(BaseCommunity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._community_reports = JsonKVStorage(self.namespace, "community_report")
        self._community_node_map = JsonKVStorage(self.namespace, "community_node_map")
        self._communities_schema: dict[str, LeidenInfo] = defaultdict(LeidenInfo)

    @property
    def community_reports(self):
        return self._community_reports

    @property
    def community_node_map(self):
        return self._community_node_map

    async def clustering(self, largest_cc, max_cluster_size, random_seed):
        return await self._clustering(largest_cc, max_cluster_size, random_seed)

    async def _clustering(self, largest_cc, max_cluster_size, random_seed):
        if largest_cc is None:
            logger.warning(
                "No largest connected component found; skipping Leiden clustering."
            )
            return None

        nodes = list(largest_cc.nodes())
        if not nodes:
            logger.warning("Largest connected component is empty; skipping Leiden clustering.")
            return None

        if len(nodes) == 1:
            node_id = normalize_entity_id(nodes[0])
            node_communities = {node_id: [{"level": 0, "cluster": "0"}]}
            logger.info("Singleton graph: created one level-0 community without Leiden.")
            await self._community_node_map.upsert(node_communities)
            return node_communities

        community_mapping = hierarchical_leiden(
            largest_cc,
            max_cluster_size=max_cluster_size,
            random_seed=random_seed,
        )
        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        levels = defaultdict(set)

        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            # stable_largest_connected_component normalizes a copied graph to
            # uppercase for deterministic clustering. Normalize back using the
            # same Unicode-safe graph-identity policy as extraction/linking.
            node_communities[normalize_entity_id(partition.node)].append(
                {"level": level_key, "cluster": str(cluster_id)}
            )
            levels[level_key].add(cluster_id)

        result = dict(node_communities)
        logger.info(
            f"Each level has communities: "
            f"{ {key: len(value) for key, value in levels.items()} }"
        )
        await self._community_node_map.upsert(result)
        return result

    @property
    def community_schema(self):
        return self._communities_schema

    async def _generate_community_report(self, er_graph):
        await er_graph.cluster_data_to_subgraphs(self._community_node_map.json_data)
        self._communities_schema = await er_graph.community_schema()
        if self._communities_schema is None:
            logger.warning("No community schema found; skipping report generation.")
            return None

        community_keys = list(self._communities_schema.keys())
        community_values = list(self._communities_schema.values())
        levels = sorted({community.level for community in community_values}, reverse=True)
        logger.info(f"Generating by levels: {levels}")
        community_datas = {}

        for level in levels:
            level_pairs = [
                (key, value)
                for key, value in zip(community_keys, community_values)
                if value.level == level
            ]
            if not level_pairs:
                continue
            this_keys, this_values = zip(*level_pairs)
            reports = await asyncio.gather(
                *[
                    self._form_single_community_report(
                        er_graph,
                        community,
                        community_datas,
                    )
                    for community in this_values
                ]
            )
            community_datas.update(
                {
                    key: {
                        "report_string": community_report_from_json(report),
                        "report_json": report,
                        **community.as_dict,
                    }
                    for key, report, community in zip(this_keys, reports, this_values)
                }
            )

        await self._community_reports.upsert(community_datas)

    async def _form_single_community_report(
        self,
        er_graph,
        community,
        already_reports: dict[str, CommunityReportsResult],
    ) -> dict:
        describe = await self._pack_single_community_describe(
            er_graph,
            community,
            already_reports=already_reports,
        )
        prompt = CommunityPrompt.COMMUNITY_REPORT.format(input_text=describe)
        response = await self.llm.aask(prompt, format="json")

        if isinstance(response, str):
            import json as _json

            try:
                return _json.loads(response)
            except _json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse community report JSON, using fallback: "
                    f"{response[:200]}"
                )
                return {
                    "title": "Community Report",
                    "summary": response,
                    "findings": [],
                }
        return response

    @staticmethod
    async def _pack_single_community_by_sub_communities(
        community,
        max_token_size: int,
        already_reports: dict[str, CommunityReportsResult],
    ):
        all_sub_communities = [
            already_reports[key]
            for key in community.sub_communities
            if key in already_reports
        ]
        all_sub_communities.sort(
            key=lambda value: value["occurrence"],
            reverse=True,
        )
        truncated = truncate_list_by_token_size(
            all_sub_communities,
            key=lambda value: value["report_string"],
            max_token_size=max_token_size,
        )
        sub_fields = ["id", "report", "rating", "importance"]
        describe = list_to_quoted_csv_string(
            [sub_fields]
            + [
                [
                    index,
                    value["report_string"],
                    value["report_json"].get("rating", -1),
                    value["occurrence"],
                ]
                for index, value in enumerate(truncated)
            ]
        )
        already_nodes = set()
        already_edges = set()
        for value in truncated:
            already_nodes.update(value["nodes"])
            already_edges.update(tuple(edge) for edge in value["edges"])
        return (
            describe,
            len(encode_string_by_tiktoken(describe)),
            already_nodes,
            already_edges,
        )

    async def _pack_single_community_describe(
        self,
        er_graph: BaseGraph,
        community: LeidenInfo,
        max_token_size: int = 12000,
        already_reports=None,
    ) -> str:
        if already_reports is None:
            already_reports = {}

        nodes_in_order = sorted(community.nodes)
        edges_in_order = sorted(community.edges, key=lambda edge: edge[0] + edge[1])
        nodes_data = await asyncio.gather(
            *[er_graph.get_node(node) for node in nodes_in_order]
        )
        edges_data = await asyncio.gather(
            *[er_graph.get_edge(src, tgt) for src, tgt in edges_in_order]
        )

        node_fields = ["id", "entity", "type", "description", "degree"]
        edge_fields = ["id", "source", "target", "description", "rank"]

        nodes_list_data = [
            [
                index,
                node_name,
                (node_data or {}).get("entity_type", "UNKNOWN"),
                (node_data or {}).get("description", "UNKNOWN"),
                await er_graph.node_degree(node_name),
            ]
            for index, (node_name, node_data) in enumerate(
                zip(nodes_in_order, nodes_data)
            )
        ]
        nodes_list_data.sort(key=lambda value: value[-1], reverse=True)
        nodes_truncated = truncate_list_by_token_size(
            nodes_list_data,
            key=lambda value: value[3],
            max_token_size=max_token_size // 2,
        )

        edges_list_data = [
            [
                index,
                edge_name[0],
                edge_name[1],
                (edge_data or {}).get("description", "UNKNOWN"),
                await er_graph.edge_degree(*edge_name),
            ]
            for index, (edge_name, edge_data) in enumerate(
                zip(edges_in_order, edges_data)
            )
        ]
        edges_list_data.sort(key=lambda value: value[-1], reverse=True)
        edges_truncated = truncate_list_by_token_size(
            edges_list_data,
            key=lambda value: value[3],
            max_token_size=max_token_size // 2,
        )

        was_truncated = (
            len(nodes_list_data) > len(nodes_truncated)
            or len(edges_list_data) > len(edges_truncated)
        )
        report_describe = ""
        use_sub_communities = (
            was_truncated
            and bool(community.sub_communities)
            and bool(already_reports)
        )

        if use_sub_communities or self.enforce_sub_communities:
            logger.info(
                f"Community {community.title} exceeds the limit or "
                "sub-community usage is forced"
            )
            (
                report_describe,
                report_size,
                contain_nodes,
                contain_edges,
            ) = await self._pack_single_community_by_sub_communities(
                community,
                max_token_size,
                already_reports,
            )
            exclude_nodes = [value for value in nodes_list_data if value[1] not in contain_nodes]
            include_nodes = [value for value in nodes_list_data if value[1] in contain_nodes]
            exclude_edges = [
                value
                for value in edges_list_data
                if (value[1], value[2]) not in contain_edges
            ]
            include_edges = [
                value
                for value in edges_list_data
                if (value[1], value[2]) in contain_edges
            ]
            remaining = max(0, max_token_size - report_size)
            nodes_truncated = truncate_list_by_token_size(
                exclude_nodes + include_nodes,
                key=lambda value: value[3],
                max_token_size=remaining // 2,
            )
            edges_truncated = truncate_list_by_token_size(
                exclude_edges + include_edges,
                key=lambda value: value[3],
                max_token_size=remaining // 2,
            )

        nodes_describe = list_to_quoted_csv_string([node_fields] + nodes_truncated)
        edges_describe = list_to_quoted_csv_string([edge_fields] + edges_truncated)
        return f"""-----Reports-----
            ```csv
            {report_describe}
            ```
            -----Entities-----
            ```csv
            {nodes_describe}
            ```
            -----Relationships-----
            ```csv
            {edges_describe}
        ```"""

    async def _load_community_report(self, graph, force) -> bool:
        if force:
            logger.info("Force regeneration of community report requested.")
            return True
        await self._community_reports.load()
        if await self._community_reports.is_empty():
            logger.error("Failed to load community report.")
            return False
        self._communities_schema = await graph.community_schema()
        logger.info("Successfully loaded community report.")
        return True

    async def _persist_community(self):
        try:
            await self._community_reports.persist()
        except Exception as exc:
            logger.exception(f"Failed to persist community report: {exc}")

    async def _load_cluster_map(self, force):
        if force:
            return False
        await self._community_node_map.load()
        if await self._community_node_map.is_empty():
            logger.error("Failed to load community <-> node map.")
            return False
        logger.info("Successfully loaded community <-> node map.")
        return True

    async def _persist_cluster_map(self):
        try:
            await self._community_node_map.persist()
        except Exception as exc:
            logger.exception(f"Failed to persist community <-> node map: {exc}")
