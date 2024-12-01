from Core.Graph.BaseGraph import BaseGraph
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Common.Utils import logger
from Core.Index.TFIDFStore import TFIDFIndex
from Core.Prompt import KGP_QUERY_PROMPT

from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
from itertools import chain
import concurrent.futures
import requests
from Core.Common.Constants import GCUBE_TOKEN

from Core.Utils.WAT import WATAnnotation


class PassageGraph(BaseGraph):
    """
    The Passage of Graph, where each node denotes a passage (chunk) in the document, which is used for KGP
    1. Please refer to the code link: https://github.com/YuWVandy/KG-LLM-MDQA
    2. You can see the paper: https://arxiv.org/abs/2308.11730
    """
    k: int = 30
    k_nei: int = 3

    def _wat_entity_linking(self, text):
        # Main method, text annotation with WAT entity linking system
        wat_url = 'https://wat.d4science.org/wat/tag/tag'
        payload = [("gcube-token", GCUBE_TOKEN),
                   ("text", text),
                   ("lang", 'en'),
                   ("tokenizer", "nlp4j"),
                   ('debug', 9),
                   ("method",
                    "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear")]

        response = requests.get(wat_url, params=payload)

        return [WATAnnotation(**annotation) for annotation in response.json()['annotations']]

    def wiki_kw_extract_chunk(self, chunk, prior_prob=0.8):

        wat_annotations = self._wat_entity_linking(chunk)
        json_list = [wat.as_dict for wat in wat_annotations]
        kw2chunk = defaultdict(set)
        chunk2kw = defaultdict(set)

        for wiki in json_list:
            if wiki['wiki_title'] != '' and wiki['prior_prob'] > prior_prob:
                kw2chunk[wiki['wiki_title']].add(chunk)
                chunk2kw[chunk].add(wiki['wiki_title'])

        return kw2chunk, chunk2kw

    async def _construct_graph(self, chunks: dict[str, TextChunk], prior_prob=0.8):
        kw2chunk = defaultdict(set)  # kw1 -> [chunk1, chunk2, ...]
        chunk2kw = defaultdict(set)  # chunk -> [kw1, kw2, ...]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_chunk = {
                executor.submit(self.wiki_kw_extract_chunk, value['content'], prior_prob): value['content'] for
                key, value in chunks.items()}
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk, inv_chunk = future.result()
                for key, value in chunk.items():
                    kw2chunk[key].update(value)
                for key, value in inv_chunk.items():
                    chunk2kw[key].update(value)

        for key in kw2chunk:
            kw2chunk[key] = list(kw2chunk[key])

        for key in chunk2kw:
            chunk2kw[key] = list(chunk2kw[key])

        chunk2id = {}
        for key, value in chunks.items():
            await self.kgp_graph.upsert_node(node_id=key, node_data=dict(chunk=value['content']))
            chunk2id[value['content']] = key

        for kw, chunk_list in kw2chunk.items():
            for i in range(len(chunk_list)):
                for j in range(i + 1, len(chunk_list)):
                    # logger.info("{src_id} ---{kw}---> {tgt_id}".format(src_id = chunk2id[chunk_list[i]], tgt_id = chunk2id[chunk_list[j]], kw = kw))
                    await self.kgp_graph.upsert_edge(source_node_id=chunk2id[chunk_list[i]],
                                                     target_node_id=chunk2id[chunk_list[j]], edge_data=dict(kw=kw))
