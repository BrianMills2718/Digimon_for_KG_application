import asyncio
import json
import re
from collections import defaultdict
from typing import Any, List, Dict, Optional, Tuple, Union
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
from Core.Storage.NetworkXStorage import NetworkXStorage


class ERGraph(BaseGraph):

    def __init__(self, config, llm, encoder, storage_instance=None):
        """
        Args:
            config: GraphConfig
            llm: LLM instance
            encoder: encoder instance
            storage_instance: Optional[NetworkXStorage], if provided will be used as the graph storage
        """
        super().__init__(config, llm, encoder)
        self._graph = storage_instance if storage_instance is not None else NetworkXStorage()

    async def _named_entity_recognition(self, passage: str):
        from Core.Common.Logger import logger
        ner_messages = GraphPrompt.NER.format(user_input=passage)
        llm_output_str = await self.llm.aask(ner_messages, format="json")
        
        parsed_output = None
        if isinstance(llm_output_str, str):
            try:
                # Clean markdown code fences before parsing
                processed_output_str = llm_output_str.strip()
                if processed_output_str.startswith("```json"):
                    processed_output_str = processed_output_str[7:]  # Remove ```json
                elif processed_output_str.startswith("```"):  # Handle if just ``` was used
                    processed_output_str = processed_output_str[3:]
                if processed_output_str.endswith("```"):
                    processed_output_str = processed_output_str[:-3]
                processed_output_str = processed_output_str.strip()
                
                # First try using the utility function if available
                try:
                    from Core.Common.Utils import prase_json_from_response
                    parsed_output = prase_json_from_response(processed_output_str)
                except (ImportError, Exception) as utility_error:
                    # Fall back to direct json.loads if utility fails
                    logger.debug(f"Falling back to direct json.loads: {utility_error}")
                    parsed_output = json.loads(processed_output_str)
            except json.JSONDecodeError as e:
                logger.error(f"NER JSONDecodeError: {e} - Output: {llm_output_str[:500]}")
                return [] # Return empty list on parsing failure
        elif isinstance(llm_output_str, dict):
            parsed_output = llm_output_str
        else:
            logger.error(f"NER - Unexpected LLM output type: {type(llm_output_str)}")
            return []

        if not isinstance(parsed_output, dict) or 'named_entities' not in parsed_output or not isinstance(parsed_output.get('named_entities'), list):
            logger.warning(f"NER - 'named_entities' key missing or not a list in parsed output: {parsed_output}")
            return []
        
        return parsed_output['named_entities']

    async def _openie_post_ner_extract(self, chunk, entities):
        from Core.Common.Logger import logger
        named_entity_json = {"named_entities": entities}
        openie_messages = GraphPrompt.OPENIE_POST_NET.format(passage=chunk,
                                                         named_entity_json=json.dumps(named_entity_json))
        llm_output_str = await self.llm.aask(openie_messages, format="json")
        
        parsed_output = None
        if isinstance(llm_output_str, str):
            try:
                # Clean markdown code fences before parsing
                processed_output_str = llm_output_str.strip()
                if processed_output_str.startswith("```json"):
                    processed_output_str = processed_output_str[7:]  # Remove ```json
                elif processed_output_str.startswith("```"):  # Handle if just ``` was used
                    processed_output_str = processed_output_str[3:]
                if processed_output_str.endswith("```"):
                    processed_output_str = processed_output_str[:-3]
                processed_output_str = processed_output_str.strip()
                
                # First try using the utility function if available
                try:
                    from Core.Common.Utils import prase_json_from_response
                    parsed_output = prase_json_from_response(processed_output_str)
                except (ImportError, Exception) as utility_error:
                    # Fall back to direct json.loads if utility fails
                    logger.debug(f"Falling back to direct json.loads: {utility_error}")
                    parsed_output = json.loads(processed_output_str)
            except json.JSONDecodeError as e:
                logger.error(f"OpenIE JSONDecodeError: {e} - Output: {llm_output_str[:500]}")
                return [] # Return empty list on parsing failure
        elif isinstance(llm_output_str, dict):
            parsed_output = llm_output_str
        else:
            logger.error(f"OpenIE - Unexpected LLM output type: {type(llm_output_str)}")
            return []

        if not isinstance(parsed_output, dict) or 'triples' not in parsed_output or not isinstance(parsed_output.get('triples'), list):
            logger.warning(f"OpenIE - 'triples' key missing or not a list in parsed output: {parsed_output}")
            return []
        
        return parsed_output['triples']

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
        from Core.Common.Logger import logger
        knowledge_graph_prompt = TextPrompt(GraphPrompt.KG_AGNET)
        knowledge_graph_generation = knowledge_graph_prompt.format(
            task=chunk_info
        )

        knowledge_graph_generation_msg = Message(role="Graphify", content=knowledge_graph_generation)
        content = await self.llm.aask(knowledge_graph_generation_msg.content)

        return content

    async def _build_graph(self, chunk_list: List[Any]) -> bool:
        from Core.Common.Logger import logger
        try:
            results = await asyncio.gather(
                *[self._extract_entity_relationship(chunk) for chunk in chunk_list])
            await self.__graph__(results)
            logger.info("Successfully built graph")
            return True
        except Exception as e:
            logger.exception(f"Error building graph: {e}")
            return False
        finally:
            logger.info("Constructing graph finished")

    async def _build_graph_by_regular_matching(self, content: str, chunk_key):
        from Core.Common.Logger import logger
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        # Extract nodes
        matches = re.findall(NODE_PATTERN, content)
        for match in matches:
            entity_name, entity_type = match
            entity_name = clean_str(entity_name)
            entity_type = clean_str(entity_type)
            if entity_name not in maybe_nodes:
                custom_ontology = getattr(self.config, 'loaded_custom_ontology', None)
                entity_attributes = {}
                final_entity_type = entity_type
                if custom_ontology and custom_ontology.get('entities'):
                    for entity_def in custom_ontology['entities']:
                        if entity_def.get('name') == entity_type:
                            final_entity_type = entity_def['name']
                            if 'properties' in entity_def:
                                for prop_def in entity_def['properties']:
                                    prop_name = prop_def.get('name')
                                    # No additional properties in this context, but placeholder
                            break
                entity = Entity(entity_name=entity_name, entity_type=final_entity_type, source_id=chunk_key, attributes=entity_attributes)
                maybe_nodes[entity_name].append(entity)

        # Extract relationships
        matches = re.findall(REL_PATTERN, content)
        for match in matches:
            src_id, _, tgt_id, _, rel_type = match
            src_id = clean_str(src_id)
            tgt_id = clean_str(tgt_id)
            rel_type = clean_str(rel_type)
            if src_id in maybe_nodes and tgt_id in maybe_nodes:
                custom_ontology = getattr(self.config, 'loaded_custom_ontology', None)
                relation_attributes = {}
                final_relation_name = clean_str(rel_type)
                if custom_ontology and custom_ontology.get('relations'):
                    for relation_def in custom_ontology['relations']:
                        if relation_def.get('name') == rel_type:
                            final_relation_name = relation_def['name']
                            if 'properties' in relation_def:
                                for prop_def in relation_def['properties']:
                                    prop_name = prop_def.get('name')
                                    # No additional properties in this context, but placeholder
                            break
                relationship = Relationship(
                    src_id=clean_str(src_id), tgt_id=clean_str(tgt_id), source_id=chunk_key,
                    relation_name=final_relation_name,
                    attributes=relation_attributes
                )
                maybe_edges[(src_id, tgt_id)].append(relationship)

        return maybe_nodes, maybe_edges

    async def _build_graph_from_tuples(self, entities, triples, chunk_key):
        # Import logger locally to ensure it's available
        from Core.Common.Logger import logger
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
        # Initialize dictionaries to hold node and edge information
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        # Process entities
        for _entity in entities:
            # Clean the entity name
            entity_name = clean_str(_entity)
            if entity_name == '':
                logger.warning(f"Entity name is not valid, entity is: {_entity}, skipping.")
                continue
                
            # Initialize attributes and type
            entity_attributes = {}
            final_entity_type = ''
            
            # Check for custom ontology
            custom_ontology = getattr(self.config, 'loaded_custom_ontology', None)
            if custom_ontology and custom_ontology.get('entities'):
                # Try to match entity type if possible
                for entity_def in custom_ontology['entities']:
                    name_match = entity_def.get('name') == entity_name
                    type_match = isinstance(_entity, dict) and entity_def.get('name') == _entity.get('type', '')
                    
                    if name_match or type_match:
                        final_entity_type = entity_def['name']
                        
                        # Populate defined custom attributes if present
                        if 'properties' in entity_def and isinstance(_entity, dict):
                            for prop_def in entity_def['properties']:
                                prop_name = prop_def.get('name')
                                if prop_name in _entity:
                                    entity_attributes[prop_name] = _entity[prop_name]
                        break
            
            # If no type found in ontology, try to get it from the entity itself
            if not final_entity_type:
                final_entity_type = _entity.get('type', '') if isinstance(_entity, dict) and 'type' in _entity else ''
            
            # Create entity object and add to nodes
            entity = Entity(
                entity_name=entity_name, 
                entity_type=final_entity_type, 
                source_id=chunk_key, 
                attributes=entity_attributes
            )
            maybe_nodes[entity_name].append(entity)

        # Process triples (relationships)
        for triple in triples:
            # Handle case where triple might be nested in a list
            if isinstance(triple[0], list): 
                triple = triple[0]
                
            # Validate triple length
            if len(triple) != 3:
                logger.warning(f"Triple length is not 3, triple is: {triple}, len is {len(triple)}, skipping.")
                continue
                
            # Clean entities and relation names
            src_entity = clean_str(triple[0])
            tgt_entity = clean_str(triple[2])
            relation_name = clean_str(triple[1])
            
            # Validate entities and relation are not empty
            if src_entity == '' or tgt_entity == '' or relation_name == '':
                logger.warning(f"Triple is not valid, contains empty entity or relation: {triple}, skipping.")
                continue
                
            # Make sure all elements are strings
            if isinstance(src_entity, str) and isinstance(tgt_entity, str) and isinstance(relation_name, str):
                # Initialize relation attributes
                relation_attributes = {}
                final_relation_name = relation_name
                
                # Check for custom ontology for relations
                custom_ontology = getattr(self.config, 'loaded_custom_ontology', None)
                if custom_ontology and custom_ontology.get('relations'):
                    for relation_def in custom_ontology['relations']:
                        if relation_def.get('name') == relation_name:
                            final_relation_name = relation_def['name']
                            
                            # Populate defined custom attributes if present
                            if 'properties' in relation_def and isinstance(triple, dict):
                                for prop_def in relation_def['properties']:
                                    prop_name = prop_def.get('name')
                                    if prop_name in triple:
                                        relation_attributes[prop_name] = triple[prop_name]
                            break
                
                # Create relationship object and add to edges
                relationship = Relationship(
                    src_id=src_entity,
                    tgt_id=tgt_entity,
                    weight=1.0, 
                    source_id=chunk_key,
                    relation_name=final_relation_name,
                    attributes=relation_attributes
                )
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        # Convert defaultdicts to regular dicts before returning
        return dict(maybe_nodes), dict(maybe_edges)
