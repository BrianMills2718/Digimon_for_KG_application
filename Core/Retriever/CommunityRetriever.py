 async def _find_relevant_community_from_entities(self, node_datas: list[dict], community_reports):
        # ✅
        related_communities = []
        for node_d in node_datas:
            if "clusters" not in node_d:
                continue
            related_communities.extend(json.loads(node_d["clusters"]))
        related_community_dup_keys = [
            str(dp["cluster"])
            for dp in related_communities
            if dp["level"] <= self.config.level
        ]
        import pdb
        pdb.set_trace()
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
            max_token_size= self.config.local_max_token_for_community_report,
        )
        if self.config.local_community_single_one:
            use_community_reports = use_community_reports[:1]

        return use_community_reports