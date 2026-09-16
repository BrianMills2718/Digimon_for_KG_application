import asyncio
from collections import defaultdict
from itertools import combinations
from typing import Any, List

import requests

from Core.Common.Constants import GCUBE_TOKEN, GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Graph.BaseGraph import BaseGraph
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Utils.WAT import WATAnnotation


class PassageGraph(BaseGraph):
    """Passage graph whose nodes are chunks linked by shared WAT entities."""

    def __init__(self, config, llm, encoder):
        from Core.Common.TokenizerWrapper import TokenizerWrapper

        graph_config = config.graph if hasattr(config, "graph") else config
        super().__init__(graph_config, llm, TokenizerWrapper())
        self.encoder = encoder
        self.k = 30
        self.k_nei = 3
        self._graph = NetworkXStorage()

    @staticmethod
    async def _wat_entity_linking(text: str):
        wat_url = "https://wat.d4science.org/wat/tag/tag"
        payload = [
            ("gcube-token", GCUBE_TOKEN),
            ("text", text),
            ("lang", "en"),
            ("tokenizer", "nlp4j"),
            ("debug", 9),
            (
                "method",
                "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,"
                "prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,"
                "voting:relatedness=lm,ranker:model=0046.model,"
                "confidence:model=pruner-wiki.linear",
            ),
        ]

        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    requests.get,
                    wat_url,
                    params=payload,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                return [
                    WATAnnotation(**annotation)
                    for annotation in data.get("annotations", [])
                ]
            except (requests.RequestException, ValueError, KeyError) as exc:
                logger.warning(
                    f"WAT entity linking attempt {attempt + 1}/3 failed: {exc}"
                )

        logger.error("WAT entity linking failed after 3 attempts")
        return []

    async def _extract_entity_relationship(
        self, chunk_key_pair: tuple[str, TextChunk]
    ) -> Any:
        chunk_key, chunk_info = chunk_key_pair
        annotations = await self._wat_entity_linking(chunk_info.content)
        return await self._build_graph_from_wat(annotations, chunk_key)

    async def _build_graph(self, chunk_list: List[Any]):
        """Extract shared entities and build the passage graph.

        The previous implementation contained a hard-coded cold-start index and
        loaded generic checkpoint files from the process working directory. That
        made small/fresh datasets silently skip extraction and allowed cross-run
        contamination. Build directly from the supplied dataset instead.
        """
        try:
            concurrency = max(1, min(8, len(chunk_list)))
            semaphore = asyncio.Semaphore(concurrency)

            async def extract(chunk_pair):
                async with semaphore:
                    return await self._extract_entity_relationship(chunk_pair)

            results = await asyncio.gather(
                *(extract(chunk_pair) for chunk_pair in chunk_list)
            )
            await self.__passage_graph__(results, chunk_list)
            return True
        except Exception as exc:
            logger.exception(f"Error building passage graph: {exc}")
            return False
        finally:
            logger.info("Constructing passage graph finished")

    async def __passage_graph__(self, elements, chunk_list: List[Any]):
        merge_wikis = defaultdict(list)
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        for wiki_chunks in elements:
            for wiki_title, chunks in wiki_chunks.items():
                merge_wikis[wiki_title].extend(chunks)

        for chunk_key, chunk in chunk_list:
            maybe_nodes[chunk_key].append(
                Entity(
                    entity_name=chunk_key,
                    entity_type="passage",
                    description=chunk.content,
                    source_id=chunk_key,
                )
            )

        seen_edges = set()
        for wiki_title, chunks in merge_wikis.items():
            for chunk1, chunk2 in combinations(sorted(set(chunks)), 2):
                src_id, tgt_id = sorted((chunk1, chunk2))
                edge_key = (src_id, tgt_id)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                maybe_edges[edge_key].append(
                    Relationship(
                        src_id=src_id,
                        tgt_id=tgt_id,
                        relation_name=wiki_title,
                        source_id=GRAPH_FIELD_SEP.join([chunk1, chunk2]),
                    )
                )

        await asyncio.gather(
            *(self._merge_nodes_then_upsert(key, value) for key, value in maybe_nodes.items())
        )
        await asyncio.gather(
            *(
                self._merge_edges_then_upsert(src, tgt, value)
                for (src, tgt), value in maybe_edges.items()
            )
        )

    async def _build_graph_from_wat(self, wat_annotations, chunk_key):
        wiki_to_chunks = defaultdict(set)
        prior_prob = getattr(self.config, "prior_prob", 0.0)
        for wiki in wat_annotations:
            if wiki.wiki_title and wiki.prior_prob > prior_prob:
                wiki_to_chunks[wiki.wiki_title].add(chunk_key)
        return dict(wiki_to_chunks)

    @property
    def entity_metakey(self):
        return "entity_name"
