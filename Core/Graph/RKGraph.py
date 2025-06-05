import re
import asyncio
import json
from collections import defaultdict
from typing import Union, List, Any
from Core.Graph.BaseGraph import BaseGraph
from Core.Common.Logger import logger
from Core.Common.Utils import (
    clean_str,
    split_string_by_multi_markers,
    is_float_regex
)
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.Message import Message
from Core.Prompt import GraphPrompt
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Common.Constants import (
    DEFAULT_RECORD_DELIMITER,
    DEFAULT_COMPLETION_DELIMITER,
    DEFAULT_TUPLE_DELIMITER,
    DEFAULT_ENTITY_TYPES
)
from Core.Common.Memory import Memory
from Core.Storage.NetworkXStorage import NetworkXStorage


class RKGraph(BaseGraph):

    def __init__(self, config, llm, encoder):
        # Create a tokenizer wrapper for BaseGraph compatibility
        from Core.Common.TokenizerWrapper import TokenizerWrapper
        tokenizer = TokenizerWrapper()
        
        super().__init__(config, llm, tokenizer)  # Pass tokenizer instead of encoder
        self._graph = NetworkXStorage()
        # Handle both full config and graph config
        self.graph_config = config.graph if hasattr(config, 'graph') else config
        # Keep encoder for potential future use
        self.encoder = encoder

    async def _handle_single_entity_extraction(self, record_attributes: list[str], chunk_key: str) -> 'Entity | None':
        if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
            return None

        entity_name = clean_str(record_attributes[1])
        if not entity_name.strip():
            return None

        custom_ontology = getattr(self.graph_config, 'loaded_custom_ontology', None)
        entity_attributes = {}
        final_entity_type = clean_str(record_attributes[2])
        if custom_ontology and custom_ontology.get('entities'):
            for entity_def in custom_ontology['entities']:
                if entity_def.get('name') == final_entity_type:
                    final_entity_type = entity_def['name']
                    if 'properties' in entity_def:
                        for prop_def in entity_def['properties']:
                            prop_name = prop_def.get('name')
                            # If property is in record_attributes, add to attributes
                            if prop_name in record_attributes:
                                idx = record_attributes.index(prop_name)
                                if idx + 1 < len(record_attributes):
                                    entity_attributes[prop_name] = record_attributes[idx + 1]
                                else:
                                    entity_attributes[prop_name] = None
                    break
        entity = Entity(
            entity_name=entity_name,
            entity_type=final_entity_type,
            description=clean_str(record_attributes[3]),
            source_id=chunk_key,
            attributes=entity_attributes
        )
        return entity

    async def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]):
        chunk_key, chunk_info = chunk_key_pair
        records = await self._extract_records_from_chunk(chunk_info)
        return await self._build_graph_from_records(records, chunk_key)

    async def _build_graph(self, chunk_list: List[Any]):
        try:
            elements = await asyncio.gather(
                *[self._extract_entity_relationship(chunk) for chunk in chunk_list])
            # Build graph based on the extracted entities and triples
            await self.__graph__(elements)
            return True  # Return success
        except Exception as e:
            logger.exception(f"Error building graph: {e}")
            return False  # Return failure
        finally:
            logger.info("Constructing graph finished")

    async def _extract_records_from_chunk(self, chunk_info: TextChunk):
        """
        Extract entity and relationship from chunk, which is used for the GraphRAG.
        Please refer to the following references:
        1. https://github.com/gusye1234/nano-graphrag
        2. https://github.com/HKUDS/LightRAG/tree/main
        """
        context = self._build_context_for_entity_extraction(chunk_info.content)
        prompt_template = GraphPrompt.ENTITY_EXTRACTION_KEYWORD if getattr(self.graph_config, 'enable_edge_keywords', False) else GraphPrompt.ENTITY_EXTRACTION
        prompt = prompt_template.format(**context)

        working_memory = Memory()

        working_memory.add(Message(content=prompt, role="user"))
        final_result = await self.llm.aask(prompt)
        working_memory.add(Message(content=final_result, role="assistant"))

        for glean_idx in range(getattr(self.graph_config, 'max_gleaning', 1)):
            working_memory.add(Message(content=GraphPrompt.ENTITY_CONTINUE_EXTRACTION, role="user"))
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in working_memory.get())
            glean_result = await self.llm.aask(context)
            working_memory.add(Message(content=glean_result, role="assistant"))
            final_result += glean_result
            logger.info(f"Gleaning step {glean_idx + 1}: Accumulated LLM output so far: {glean_result[:500]}...")

            if glean_idx == getattr(self.graph_config, 'max_gleaning', 1) - 1:
                break

            working_memory.add(Message(content=GraphPrompt.ENTITY_IF_LOOP_EXTRACTION, role="user"))
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in working_memory.get())
            if_loop_result = await self.llm.aask(context)
            if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
                break
        logger.info(f"Raw LLM output for chunk {chunk_info.chunk_id} before splitting: >>>\n{final_result}\n<<<")
        working_memory.clear()
        extracted_records = split_string_by_multi_markers(final_result, [
            DEFAULT_RECORD_DELIMITER, DEFAULT_COMPLETION_DELIMITER
        ])
        logger.info(f"Split records for chunk {chunk_info.chunk_id}: {extracted_records}")
        return extracted_records

    async def _build_graph_from_records(self, records: list[str], chunk_key: str):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        for record in records:
            logger.info(f"Processing record: '{record}'")
            match = re.search(r"\((.*)\)", record)
            if match is None:
                continue

            record_attributes = split_string_by_multi_markers(match.group(1), [DEFAULT_TUPLE_DELIMITER])
            logger.info(f"Record attributes after splitting by tuple delimiter: {record_attributes}")
            entity = await self._handle_single_entity_extraction(record_attributes, chunk_key)

            if entity is not None:
                logger.info(f"Successfully extracted entity: {json.dumps(entity.as_dict, indent=2)}")
                maybe_nodes[entity.entity_name].append(entity)
                continue

            relationship = await self._handle_single_relationship_extraction(record_attributes, chunk_key)

            if relationship is not None:
                logger.info(f"Successfully extracted relationship: {json.dumps(relationship.as_dict, indent=2)}")
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        return dict(maybe_nodes), dict(maybe_edges)

    async def _handle_single_relationship_extraction(self, record_attributes: list[str], chunk_key: str) -> 'Relationship | None':
        if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
            return None

        #custom_ontology = getattr(self.config.graph_config, 'loaded_custom_ontology', None)
        custom_ontology = getattr(self.graph_config, 'loaded_custom_ontology', None)
        relation_attributes = {}
        final_relation_name = clean_str(record_attributes[0])
        if custom_ontology and custom_ontology.get('relations'):
            for relation_def in custom_ontology['relations']:
                if relation_def.get('name') == final_relation_name:
                    final_relation_name = relation_def['name']
                    if 'properties' in relation_def:
                        for prop_def in relation_def['properties']:
                            prop_name = prop_def.get('name')
                            if prop_name in record_attributes:
                                idx = record_attributes.index(prop_name)
                                if idx + 1 < len(record_attributes):
                                    relation_attributes[prop_name] = record_attributes[idx + 1]
                                else:
                                    relation_attributes[prop_name] = None
                    break
        return Relationship(
            src_id=clean_str(record_attributes[1]),
            tgt_id=clean_str(record_attributes[2]),
            weight=float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0,
            description=clean_str(record_attributes[3]),
            source_id=chunk_key,
            keywords=clean_str(record_attributes[4]) if getattr(self.graph_config, 'enable_edge_keywords', False) else "",
            relation_name=final_relation_name,
            attributes=relation_attributes
        )

    @classmethod
    def _build_context_for_entity_extraction(self, content: str) -> dict:
        return dict(
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_text=content
        )
        
    @property
    def entity_metakey(self):
        return "entity_name"