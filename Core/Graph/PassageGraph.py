import asyncio
from typing import List, Any

from Core.Graph.BaseGraph import BaseGraph
from Core.Schema.ChunkSchema import TextChunk
from Core.Common.Utils import logger

from Core.Schema.EntityRelation import Entity, Relationship
from collections import defaultdict
from itertools import combinations
import requests
from Core.Common.Constants import GCUBE_TOKEN, GRAPH_FIELD_SEP
from Core.Storage.NetworkXStorage import NetworkXStorage

from Core.Utils.WAT import WATAnnotation


class PassageGraph(BaseGraph):
    """
    PassageGraph represents a graph where each node corresponds to a passage (chunk) in a document.

    Key Features:
    - Each node in the graph represents a passage or chunk of text from the document.
    - The graph structure facilitates the propagation of knowledge across passages.

    For more details, please refer to:
    1. The original code implementation on GitHub: https://github.com/YuWVandy/KG-LLM-MDQA
    2. The associated research paper: https://arxiv.org/abs/2308.11730
    """
    def __init__(self, config, llm, encoder):
        super().__init__(config, llm, encoder)
        self.k: int = 30
        self.k_nei: int = 3
        self._graph = NetworkXStorage()

    @staticmethod
    async def _wat_entity_linking(text: str):
        # Main method, text annotation with WAT entity linking system
        wat_url = 'https://wat.d4science.org/wat/tag/tag'
        payload = [("gcube-token", GCUBE_TOKEN),
                   ("text", text),
                   ("lang", 'en'),
                   ("tokenizer", "nlp4j"),
                   ('debug', 9),
                   ("method",
                    "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear")]
        # TODO: maybe config it
        retry_count = 3
        for attempt in range(retry_count):
            try:
                response = requests.get(wat_url, params=payload)
                return [WATAnnotation(**annotation) for annotation in response.json()['annotations']]
            except requests.exceptions.RequestException as e:
                logger.error(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt == retry_count - 1:
                    logger.error("All retry attempts failed. Exiting.")
            return []

    async def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]) -> Any:
        chunk_key, chunk_info = chunk_key_pair  # Unpack the chunk key and information
        chunk_info = chunk_info.content

        # Entity linking by WAT system
        logger.info("Linking Entity by WAT system for chunk {chunk_key}".format(chunk_key=chunk_key))
        wat_annotations = await self._wat_entity_linking(chunk_info)
        return await self._build_graph_from_wat(wat_annotations, chunk_key)

    async def _build_graph(self, chunk_list: List[Any]):
        try:
            results = await asyncio.gather(
                *[self._extract_entity_relationship(chunk) for chunk in chunk_list])
            # Build graph based on the relationship of chunks
            await self.__passage_graph__(results, chunk_list)
        except Exception as e:
            logger.exception(f"Error building graph: {e}")
        finally:
            logger.info("Constructing graph finished")

    async def __passage_graph__(self, elements, chunk_list: List[Any]):
        # Initialize dictionaries to hold aggregated edge information
        merge_wikis = defaultdict(list)
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)
        # Iterate through each wiki-title
        for kw_chunk in elements:
            # Aggregate  information
            for k, v in kw_chunk.items():
                merge_wikis[k].extend(v)
        for chunk_pair in chunk_list:
          
            node_data = Entity(entity_name=chunk_pair[0], description=chunk_pair[1].content, source_id=chunk_pair[0])
            maybe_nodes[chunk_pair[0]].append(node_data)
        # Merge edge information
        # If two nodes contain the same wiki entity, then there is an edge between these two nodes
        for wiki_key, chunks in merge_wikis.items():
            # Use itertools.combinations to generate all possible pairs of chunk-keys
            for chunk1, chunk2 in combinations(chunks, 2):
                src_id, tgt_id = tuple(sorted((chunk1, chunk2)))
                edge_data = Relationship(src_id=src_id, tgt_id=tgt_id, relation_name=wiki_key,
                                         source_id=GRAPH_FIELD_SEP.join([chunk1, chunk2]))
                maybe_edges[(src_id, tgt_id)].append(edge_data)
   
        # Asynchronously merge and upsert nodes
        await asyncio.gather(*[self._merge_nodes_then_upsert(k, v) for k, v in maybe_nodes.items()])
        # Asynchronously merge and upsert edges
        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])

    async def _build_graph_from_wat(self, wat_annotations, chunk_key):
        kw2chunk = defaultdict(set)
        for wiki in wat_annotations:
            if wiki.wiki_title != '' and wiki.prior_prob > self.config.prior_prob:
                kw2chunk[wiki.wiki_title].add(chunk_key)

        return dict(kw2chunk)
    @property
    def entity_metakey(self):
        return "entity_name"