"""
Shared mixin for ENTITY_EXTRACTION-based (delimiter-delimited) entity/relationship parsing.

Used by both ERGraph (extract_two_step=False) and RKGraph to avoid code duplication.
The extraction calls the LLM with ENTITY_EXTRACTION or ENTITY_EXTRACTION_KEYWORD prompts
and parses the delimiter-separated records into Entity and Relationship objects.
"""

import re
import json
from collections import defaultdict
from typing import Any, List, Optional, Tuple

from Core.Common.EntityNormalization import normalize_entity_id
from Core.Common.Logger import logger
from Core.Common.Utils import clean_str, split_string_by_multi_markers, is_float_regex
from Core.Common.Constants import (
    DEFAULT_RECORD_DELIMITER,
    DEFAULT_COMPLETION_DELIMITER,
    DEFAULT_TUPLE_DELIMITER,
    DEFAULT_ENTITY_TYPES,
)
from Core.Common.Memory import Memory
from Core.Prompt import GraphPrompt
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Schema.Message import Message


class DelimiterExtractionMixin:
    """
    Mixin that provides ENTITY_EXTRACTION prompt-based extraction and
    delimiter-based record parsing for building knowledge graphs.

    Requirements on the host class:
      - self.llm            — LLM instance with .aask()
      - self.graph_config   — graph configuration (GraphConfig or equivalent)
                              OR self.config with the relevant fields
    """

    @staticmethod
    def _build_context_for_entity_extraction(content: str) -> dict:
        return dict(
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_text=content,
        )

    async def _extract_records_from_chunk(self, chunk_info: TextChunk) -> List[str]:
        """Call the LLM and return delimiter-split extraction records."""
        graph_cfg = getattr(self, "graph_config", self.config)

        context = self._build_context_for_entity_extraction(chunk_info.content)
        prompt_template = (
            GraphPrompt.ENTITY_EXTRACTION_KEYWORD
            if getattr(graph_cfg, "enable_edge_keywords", False)
            else GraphPrompt.ENTITY_EXTRACTION
        )
        prompt = prompt_template.format(**context)

        working_memory = Memory()
        working_memory.add(Message(content=prompt, role="user"))
        final_result = await self.llm.aask(prompt)
        working_memory.add(Message(content=final_result, role="assistant"))

        for glean_idx in range(getattr(graph_cfg, "max_gleaning", 1)):
            working_memory.add(Message(content=GraphPrompt.ENTITY_CONTINUE_EXTRACTION, role="user"))
            context_str = "\n".join(
                f"{msg.sent_from}: {msg.content}" for msg in working_memory.get()
            )
            glean_result = await self.llm.aask(context_str)
            working_memory.add(Message(content=glean_result, role="assistant"))
            final_result += glean_result
            logger.info(f"Gleaning step {glean_idx + 1}: {glean_result[:500]}...")

            if glean_idx == getattr(graph_cfg, "max_gleaning", 1) - 1:
                break

            working_memory.add(Message(content=GraphPrompt.ENTITY_IF_LOOP_EXTRACTION, role="user"))
            context_str = "\n".join(
                f"{msg.sent_from}: {msg.content}" for msg in working_memory.get()
            )
            if_loop_result = await self.llm.aask(context_str)
            if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
                break

        logger.info(
            f"Raw LLM output for chunk {chunk_info.chunk_id} before splitting: >>>\n{final_result}\n<<<"
        )
        working_memory.clear()

        extracted_records = split_string_by_multi_markers(
            final_result, [DEFAULT_RECORD_DELIMITER, DEFAULT_COMPLETION_DELIMITER]
        )
        logger.info(f"Split records for chunk {chunk_info.chunk_id}: {extracted_records}")
        return extracted_records

    async def _build_graph_from_records(
        self, records: List[str], chunk_key: str
    ) -> Tuple[dict, dict]:
        maybe_nodes: dict[str, list] = defaultdict(list)
        maybe_edges: dict[tuple, list] = defaultdict(list)

        for record in records:
            logger.info(f"Processing record: '{record}'")
            match = re.search(r"\((.*)\)", record)
            if match is None:
                continue

            record_attributes = split_string_by_multi_markers(
                match.group(1), [DEFAULT_TUPLE_DELIMITER]
            )
            logger.info(f"Record attributes after splitting: {record_attributes}")

            entity = await self._handle_single_entity_extraction(record_attributes, chunk_key)
            if entity is not None:
                logger.info(f"Extracted entity: {json.dumps(entity.as_dict, indent=2)}")
                maybe_nodes[entity.entity_name].append(entity)
                continue

            relationship = await self._handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if relationship is not None:
                logger.info(f"Extracted relationship: {json.dumps(relationship.as_dict, indent=2)}")
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        return dict(maybe_nodes), dict(maybe_edges)

    async def _handle_single_entity_extraction(
        self, record_attributes: List[str], chunk_key: str
    ) -> Optional[Entity]:
        if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
            return None

        # Graph identity is Unicode-safe. Free-text type/description normalization
        # remains unchanged to minimize the scope of this identity migration.
        entity_name = normalize_entity_id(record_attributes[1])
        if not entity_name or not entity_name.strip():
            return None

        graph_cfg = getattr(self, "graph_config", self.config)
        custom_ontology = getattr(graph_cfg, "loaded_custom_ontology", None)
        entity_attributes: dict = {}
        final_entity_type = clean_str(record_attributes[2])

        if custom_ontology and custom_ontology.get("entities"):
            for entity_def in custom_ontology["entities"]:
                if entity_def.get("name") == final_entity_type:
                    final_entity_type = entity_def["name"]
                    if "properties" in entity_def:
                        for prop_def in entity_def["properties"]:
                            prop_name = prop_def.get("name")
                            if prop_name in record_attributes:
                                idx = record_attributes.index(prop_name)
                                if idx + 1 < len(record_attributes):
                                    entity_attributes[prop_name] = record_attributes[idx + 1]
                    break

        return Entity(
            entity_name=entity_name,
            entity_type=final_entity_type,
            description=clean_str(record_attributes[3]),
            source_id=chunk_key,
            attributes=entity_attributes,
        )

    async def _handle_single_relationship_extraction(
        self, record_attributes: List[str], chunk_key: str
    ) -> Optional[Relationship]:
        if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
            return None

        graph_cfg = getattr(self, "graph_config", self.config)
        custom_ontology = getattr(graph_cfg, "loaded_custom_ontology", None)
        relation_attributes: dict = {}
        final_relation_name = clean_str(record_attributes[0])

        if custom_ontology and custom_ontology.get("relations"):
            for relation_def in custom_ontology["relations"]:
                if relation_def.get("name", "").lower() == final_relation_name.lower():
                    final_relation_name = relation_def["name"]
                    if "properties" in relation_def:
                        for prop_def in relation_def["properties"]:
                            prop_name = prop_def.get("name")
                            if prop_name in record_attributes:
                                idx = record_attributes.index(prop_name)
                                if idx + 1 < len(record_attributes):
                                    relation_attributes[prop_name] = record_attributes[idx + 1]
                    break

        enable_keywords = getattr(graph_cfg, "enable_edge_keywords", False)
        src_id = normalize_entity_id(record_attributes[1])
        tgt_id = normalize_entity_id(record_attributes[2])
        if not src_id or not tgt_id:
            return None

        return Relationship(
            src_id=src_id,
            tgt_id=tgt_id,
            weight=float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0,
            description=clean_str(record_attributes[3]),
            source_id=chunk_key,
            keywords=clean_str(record_attributes[4]) if enable_keywords else "",
            relation_name=final_relation_name,
            attributes=relation_attributes,
        )
