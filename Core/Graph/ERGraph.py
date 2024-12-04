import asyncio
import json
import re
from collections import defaultdict
from typing import Any, List
from Core.Graph.BaseGraph import BaseGraph
from Core.Common.Logger import logger
from Core.Common.Utils import (
    clean_str,
    prase_json_from_response
)
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.Message import Message
from Core.Prompt import GraphPrompt
from Core.Prompt.Base import TextPrompt
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Common.Constants import (
    NODE_PATTERN,
    REL_PATTERN
)


class ERGraph(BaseGraph):

    async def _named_entity_recognition(self, passage: str):
        ner_messages = GraphPrompt.NER.format(user_input=passage)

        response_content = await self.llm.aask(ner_messages)
        entities = prase_json_from_response(response_content)

        if 'named_entities' not in entities:
            entities = []
        else:
            entities = entities['named_entities']
        return entities

    async def _openie_post_ner_extract(self, chunk, entities):
        named_entity_json = {"named_entities": entities}
        openie_messages = GraphPrompt.OPENIE_POST_NET.format(passage=chunk,
                                                             named_entity_json=json.dumps(named_entity_json))
        response_content = await self.llm.aask(openie_messages)
        triples = prase_json_from_response(response_content)
        try:
            triples = triples["triples"]
        except:
            return []

        return triples

    async def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]) -> Any:
        chunk_key, chunk_info = chunk_key_pair  # Unpack the chunk key and information
        chunk_info = chunk_info.content
        if self.config.extract_two_step:
            # Extract entities and relationships using OPEN-IE for HippoRAG
            # Refer to: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/
            entities = await self._named_entity_recognition(chunk_info)
            triples = await self._openie_post_ner_extract(chunk_info, entities)
            return await self._build_graph_from_tuples(entities, triples, chunk_key)
        else:
            # Use KGAgent from camel for one-step entity and relationship extraction (used in MedicalRAG)
            # Refer to: https://github.com/SuperMedIntel/Medical-Graph-RAG
            graph_element = await self._kg_agent(chunk_info)
            return await self._build_graph_by_regular_matching(graph_element, chunk_key)

    async def _kg_agent(self, chunk_info):
        knowledge_graph_prompt = TextPrompt(GraphPrompt.KG_AGNET)
        knowledge_graph_generation = knowledge_graph_prompt.format(
            task=chunk_info
        )

        knowledge_graph_generation_msg = Message(role="Graphify", content=knowledge_graph_generation)

        content = await self.llm.aask(knowledge_graph_generation_msg.content)

        return content

    async def _build_graph(self, chunk_list: List[Any]):
        try:
            results = await asyncio.gather(
                *[self._extract_entity_relationship(chunk) for chunk in chunk_list])
            # Build graph based on the extracted entities and triples
            await self.__graph__(results)
        except Exception as e:
            logger.exception(f"Error building graph: {e}")
        finally:
            logger.info("Constructing graph finished")

    @staticmethod
    async def _build_graph_by_regular_matching(content: str, chunk_key):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        # Extract nodes
        for match in re.finditer(NODE_PATTERN, content):
            entity_name, entity_type = match.groups()
            entity_name = clean_str(entity_name)
            if entity_name not in maybe_nodes:
                entity = Entity(entity_name=entity_name, entity_type=entity_type, source_id=chunk_key)
                maybe_nodes[entity_name].append(entity)

        # Extract relationships

        for match in re.finditer(REL_PATTERN, content):
            src_id, _, tgt_id, _, rel_type = match.groups()
            if src_id in maybe_nodes and tgt_id in maybe_nodes:
                relationship = Relationship(
                    src_id=clean_str(src_id), tgt_id=clean_str(tgt_id), source_id=chunk_key,
                    relation_type=clean_str(rel_type)
                )
                maybe_edges[(src_id, tgt_id)].append(relationship)
        return maybe_nodes, maybe_edges

    @staticmethod
    async def _build_graph_from_tuples(entities, triples, chunk_key):
        """
           Build a graph structure from entities and triples.

           This function takes a list of entities and triples, and constructs a graph's nodes and edges
           based on this data. Each entity and triple is cleaned and processed before being added to
           the corresponding node or edge.

           Args:
               entities (List[str]): A list of entity strings.
               triples (List[Tuple[str, str, str]]): A list of triples, where each triple contains three strings (source entity, relation, target entity).
               chunk_key (str): A key used to identify the data chunk.
           """
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        for _entity in entities:
            entity_name = clean_str(_entity)
            entity = Entity(entity_name=entity_name, source_id=chunk_key)
            maybe_nodes[entity_name].append(entity)

        for triple in triples:
            if len(triple) != 3:
                logger.warning(f"triples length is not 3, triple is: {triple}, len is {len(triple)}, so skip it")
                continue
            relationship = Relationship(src_id=clean_str(triple[0]),
                                        tgt_id=clean_str(triple[2]),
                                        weight=1.0, source_id=chunk_key,
                                        relation_name=clean_str(triple[1]))
            maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        return dict(maybe_nodes), dict(maybe_edges)
