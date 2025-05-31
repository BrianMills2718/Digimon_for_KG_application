# Core/AgentTools/entity_tools.py

from typing import List, Tuple, Optional, Any
from Core.AgentSchema.tool_contracts import (
    EntityVDBSearchInputs, 
    EntityVDBSearchOutputs,
    EntityPPRInputs, 
    EntityPPROutputs
    # We will add other entity-related tool input/output models here later,
)

from Core.Schema.EntityRelation import Entity as CoreEntity # For constructing output if needed

# Placeholder for actual GraphRAG core components & context
# The Agent Orchestrator will need to provide access to these,
# e.g., through a shared context object or by direct imports if feasible.
# For now, we'll assume these are accessible or passed in.
# from Core.Index.VectorIndex import VectorIndex # Example
# from Core.Config.EmbConfig import EmbeddingConfig # Example
# from Core.Provider.EmbeddingProvider import EmbeddingProvider # Example

# --- Tool Implementation for: Entity Vector Database Search ---
# tool_id: "Entity.VDBSearch"

import os
from typing import List, Tuple, Optional, Any
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import EntityVDBSearchInputs, EntityVDBSearchOutputs
import os
from typing import List, Tuple, Optional, Any, Dict
from pydantic import BaseModel
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import EntityVDBSearchInputs, EntityVDBSearchOutputs
from Core.Index.FaissIndex import FaissIndex
from llama_index.core.embeddings import BaseEmbedding as LlamaIndexBaseEmbedding
from Core.Index.BaseIndex import BaseIndex

# A simple placeholder for the config that FaissIndex expects.
class MockIndexConfig(BaseModel):
    persist_path: str
    embed_model: LlamaIndexBaseEmbedding
    retrieve_top_k: Optional[int] = 10
    name: Optional[str] = None
    class Config:
        arbitrary_types_allowed = True

#from Core.Provider.EmbeddingProvider import EmbeddingProvider
from Core.Provider.BaseEmb import BaseEmb
# from Core.Storage.NameSpace import NameSpace

async def entity_vdb_search_tool(
    params: EntityVDBSearchInputs,
    graphrag_context: GraphRAGContext
) -> EntityVDBSearchOutputs:
    """
    Performs a vector search for entities using the LlamaIndex-based FaissIndex.
    """
    print(f"Executing tool 'Entity.VDBSearch' with parameters: {params}")

    if not params.query_text:
        raise ValueError("query_text must be provided for Entity.VDBSearch with current FaissIndex.retrieval method.")

    llama_embed_provider = graphrag_context.embedding_provider
    if not llama_embed_provider:
        raise ValueError("LlamaIndex embedding provider not found in GraphRAGContext.")
    if not isinstance(llama_embed_provider, LlamaIndexBaseEmbedding):
        print(f"Warning: embedding_provider in context is type {type(llama_embed_provider)}, expecting LlamaIndexBaseEmbedding.")

    vdb_search_results: List[Tuple[str, float]] = []
    try:
        root_dir = graphrag_context.resolved_configs.get("storage_root_dir", "./results")
        vdb_folder_path = os.path.join(
            root_dir,
            graphrag_context.target_dataset_name,
            "kg_graph",
            params.vdb_reference_id
        )
        print(f"Attempting to use VDB from persist_path: {vdb_folder_path}")
        index_init_config = MockIndexConfig(
            persist_path=vdb_folder_path,
            embed_model=llama_embed_provider,
            retrieve_top_k=params.top_k_results,
            name=params.vdb_reference_id
        )
        vdb_instance = FaissIndex(config=index_init_config)
        if not await vdb_instance.load():
            print(f"Warning: Failed to load VDB from {vdb_folder_path}. It might not exist or be corrupted.")
            return EntityVDBSearchOutputs(similar_entities=[])
        print(f"Successfully loaded VDB '{params.vdb_reference_id}'.")
        print(f"Searching VDB '{params.vdb_reference_id}' for query '{params.query_text}' with top_k={params.top_k_results}")
        llama_index_results: List[Any] = await vdb_instance.retrieval(
            query=params.query_text,
            top_k=params.top_k_results
        )
        for node_with_score in llama_index_results:
            entity_id = node_with_score.node.id_
            score = node_with_score.score
            if entity_id and score is not None:
                vdb_search_results.append((str(entity_id), float(score)))
    except FileNotFoundError as e:
        print(f"VDB artifact not found or invalid at '{vdb_folder_path}': {e}")
    except Exception as e:
        print(f"Error during VDB operation for '{params.vdb_reference_id}': {e}")
        import traceback
        traceback.print_exc()
    return EntityVDBSearchOutputs(similar_entities=vdb_search_results)

# --- Tool Implementation for: Entity Personalized PageRank (PPR) ---
# tool_id: "Entity.PPR"

from Core.Retriever.EntitiyRetriever import EntityRetriever  # Note: typo in filename is intentional
from Config.RetrieverConfig import RetrieverConfig
from Core.Common.Logger import logger
import numpy as np
from typing import List, Tuple, Optional, Any
from Core.AgentSchema.tool_contracts import EntityPPRInputs, EntityPPROutputs
from Core.AgentSchema.context import GraphRAGContext

from Core.Retriever.EntitiyRetriever import EntityRetriever
from Config.RetrieverConfig import RetrieverConfig
from Core.Common.Logger import logger
import numpy as np

async def entity_ppr_tool(
    params: EntityPPRInputs,
    graphrag_context: GraphRAGContext
) -> EntityPPROutputs:
    """
    Computes Personalized PageRank for entities in a graph using EntityRetriever.
    """
    logger.info(f"Executing tool 'Entity.PPR' with parameters: {params}")

    # --- 1. Get necessary components from GraphRAGContext ---
    graph_instance = graphrag_context.graph_instance
    entities_vdb_instance = graphrag_context.entities_vdb_instance

    if not graph_instance:
        logger.error("Entity.PPR: Graph instance not found in GraphRAGContext.")
        raise ValueError("Graph instance not found in GraphRAGContext.")

    # --- 2. Prepare RetrieverConfig ---
    method_retriever_config_dict = graphrag_context.resolved_configs.get("retriever_config_dict", {})
    
    internal_top_k = params.top_k_results if params.top_k_results is not None else 10
    if "top_k" in method_retriever_config_dict: # Use top_k from provided config if greater
        internal_top_k = max(internal_top_k, method_retriever_config_dict.get("top_k", internal_top_k))
    
    # Set defaults carefully, allowing method_retriever_config_dict to override
    # These are typical settings that would come from a method's YAML retriever config.
    final_retriever_config_values = {
        "retriever_type": "unknown", # Placeholder, actual type might be in method_retriever_config_dict
        "llm_config": graphrag_context.resolved_configs.get("main_config_dict", {}).get("llm"),
        "embedding_config": graphrag_context.resolved_configs.get("main_config_dict", {}).get("embedding"),
        "top_k": internal_top_k,
        "use_entity_similarity_for_ppr": method_retriever_config_dict.get("use_entity_similarity_for_ppr", False),
        "node_specificity": method_retriever_config_dict.get("node_specificity", True),
        "top_k_entity_for_ppr": method_retriever_config_dict.get("top_k_entity_for_ppr", 5)
    }
    # Overlay any other values from the passed method_retriever_config_dict
    final_retriever_config_values.update(method_retriever_config_dict)

    try:
        retriever_config = RetrieverConfig(**final_retriever_config_values)
        logger.info(f"Entity.PPR: Using RetrieverConfig: {retriever_config.model_dump(exclude_none=True, exclude_defaults=True)}")
    except Exception as e:
        logger.error(f"Entity.PPR: Failed to create RetrieverConfig. Error: {e}", exc_info=True)
        logger.error(f"Entity.PPR: Provided final_retriever_config_values was: {final_retriever_config_values}")
        raise ValueError(f"Failed to create RetrieverConfig for Entity.PPR: {e}")

    entities_vdb_instance = graphrag_context.entities_vdb_instance
    if retriever_config.use_entity_similarity_for_ppr and not entities_vdb_instance:
        logger.error("Entity.PPR: Entities VDB instance is required for similarity-based PPR but not found in GraphRAGContext.")
        raise ValueError("Entities VDB instance not found in GraphRAGContext (required for similarity-based PPR).")

    # --- 3. Instantiate EntityRetriever ---
    entity_retriever = EntityRetriever(
        config=retriever_config,
        graph=graph_instance,
        entities_vdb=entities_vdb_instance
    )
    # Ensure internal top_k for retriever processing is sufficient
    entity_retriever.config.top_k = internal_top_k 

    # --- 4. Call _find_relevant_entities_by_ppr from EntityRetriever ---
    if not params.seed_entity_ids:
        logger.warning("Entity.PPR: No seed_entity_ids provided. Returning empty results.")
        return EntityPPROutputs(ranked_entities=[])

    placeholder_query_for_vdb = params.seed_entity_ids[0] 
    attempt_linking = bool(entities_vdb_instance) 
    
    seed_entities_arg = params.seed_entity_ids
    if not attempt_linking and not retriever_config.use_entity_similarity_for_ppr:
        seed_entities_arg = [{"entity_name": eid} for eid in params.seed_entity_ids]
        logger.info(f"Entity.PPR: VDB not available for linking & not using VDB for PPR sim. Passing seed entities as list of dicts.")
    
    logger.info(f"Entity.PPR: Calling _find_relevant_entities_by_ppr with {len(params.seed_entity_ids)} seeds. Linking: {attempt_linking}. Args: {seed_entities_arg}")

    try:
        top_k_nodes_data, full_ppr_scores = await entity_retriever._find_relevant_entities_by_ppr(
            query=placeholder_query_for_vdb,
            seed_entities=seed_entities_arg,
            link_entity=attempt_linking
        )
    except Exception as e:
        logger.error(f"Entity.PPR: Error during entity_retriever._find_relevant_entities_by_ppr: {e}", exc_info=True)
        return EntityPPROutputs(ranked_entities=[])

    if top_k_nodes_data is None or full_ppr_scores is None:
        logger.warning("Entity.PPR: PPR execution in EntityRetriever returned None or empty scores. Returning empty results.")
        return EntityPPROutputs(ranked_entities=[])

    # --- 5. Format results into EntityPPROutputs ---
    ranked_entities: List[Tuple[str, float]] = []
    entity_meta_key = graph_instance.entity_metakey

    logger.info(f"Entity.PPR: Processing {len(top_k_nodes_data)} nodes returned by retriever's PPR.")
    for node_data in top_k_nodes_data:
        if node_data is None:
            logger.warning("Entity.PPR: Encountered None in top_k_nodes_data. Skipping.")
            continue
        
        entity_id = node_data.get(entity_meta_key)
        if not entity_id:
            logger.warning(f"Entity.PPR: Node data missing '{entity_meta_key}'. Data: {node_data}. Skipping.")
            continue
        
        try:
            node_idx = await graph_instance.get_node_index(entity_id)
            if node_idx is not None and 0 <= node_idx < len(full_ppr_scores):
                score = full_ppr_scores[node_idx]
                ranked_entities.append((str(entity_id), float(score)))
            else:
                logger.warning(f"Entity.PPR: Node index {node_idx} for entity '{entity_id}' is out of bounds or None for full_ppr_scores (len: {len(full_ppr_scores)}). Skipping.")
        except Exception as e:
            logger.error(f"Entity.PPR: Error processing node '{entity_id}' for PPR score: {e}", exc_info=True)
            
    ranked_entities.sort(key=lambda x: x[1], reverse=True)

    if params.top_k_results is not None:
        ranked_entities = ranked_entities[:params.top_k_results]
            
    logger.info(f"Entity.PPR tool returning {len(ranked_entities)} ranked entities.")
    return EntityPPROutputs(ranked_entities=ranked_entities)
    num_to_add = (params.top_k_results if params.top_k_results is not None else 3) - len(dummy_ppr_results)
    for i in range(num_to_add):
        dummy_ppr_results.append((f"dummy_ppr_entity_{i}", 0.5 - (i * 0.05)))

    # 3. Transform the results into the 'EntityPPROutputs' Pydantic model
    #    The 'dummy_ppr_results' above is already in List[Tuple[str, float]] format.
    
    return EntityPPROutputs(ranked_entities=dummy_ppr_results[:params.top_k_results])


# --- Tool Implementation for: Entity Operator - Agent ---
# tool_id: "Entity.Agent"

from Core.AgentSchema.tool_contracts import (
    EntityAgentInputs, EntityAgentOutputs, ExtractedEntityData,
    EntityRelNodeInputs, EntityRelNodeOutputs,
    EntityLinkInputs, EntityLinkOutputs, LinkedEntityPair,
    RelationshipData
)
from typing import Literal
from Core.AgentSchema.context import GraphRAGContext

async def entity_agent_tool(
    params: EntityAgentInputs,
    graphrag_context: Optional[Any] = None # Placeholder for GraphRAG system context
) -> EntityAgentOutputs:
    """
    Utilizes an LLM to find or extract useful entities from the given context.
    Wraps core GraphRAG logic for LLM-based entity extraction.
    """
    print(f"Executing tool 'Entity.Agent' with parameters: {params}")

    # 1. Extract parameters from 'params: EntityAgentInputs'
    #    - query_text: str
    #    - text_context: Union[str, List[str]]
    #    - existing_entity_ids: Optional[List[str]]
    #    - target_entity_types: Optional[List[str]]
    #    - llm_config_override_patch: Optional[Dict[str, Any]]
    #    - max_entities_to_extract: Optional[int]

    # Placeholder: Prepare prompt for LLM using query_text, text_context, and target_entity_types.
    # Placeholder: Get LLM instance (potentially from graphrag_context, applying llm_config_override_patch).
    # Placeholder: Make LLM call.
    # Placeholder: Parse LLM response into List[ExtractedEntityData].

    print(f"Placeholder: Would use LLM to extract entities from context related to '{params.query_text}'.")
    print(f"Target entity types: {params.target_entity_types}")

    # Dummy results
    dummy_extracted_entities = []
    num_to_extract = params.max_entities_to_extract if params.max_entities_to_extract is not None else 2
    for i in range(num_to_extract):
        entity_type = params.target_entity_types[0] if params.target_entity_types else "Unknown"
        dummy_extracted_entities.append(
            ExtractedEntityData( # This should be the harmonized model from tool_contracts.py
                entity_name=f"llm_extracted_entity_{i+1}", # CoreEntity uses entity_name
                source_id="llm_agent_tool", # Or a more specific source
                entity_type=entity_type,
                description=f"Description for LLM-extracted entity {i+1} of type {entity_type}.",
                attributes={"llm_confidence": 0.85 + (i*0.01)},
                extraction_confidence=0.85 + (i*0.01) # Example additional field
            )
        )

    return EntityAgentOutputs(extracted_entities=dummy_extracted_entities)


# --- Tool Implementation for: Entity Operator - RelNode ---
# tool_id: "Entity.RelNode"

async def entity_rel_node_tool(
    params: EntityRelNodeInputs,
    graphrag_context: Optional[Any] = None
) -> EntityRelNodeOutputs:
    """
    Extracts unique entity IDs from a given list of relationships, based on node roles.
    Wraps core GraphRAG logic.
    """
    print(f"Executing tool 'Entity.RelNode' with parameters: {params}")

    # 1. Extract parameters from 'params: EntityRelNodeInputs'
    #    - relationships: List[RelationshipData]
    #    - node_role: Optional[Literal["source", "target", "both"]]

    # Placeholder: Process params.relationships to extract entity IDs based on params.node_role
    print(f"Placeholder: Would extract entities from {len(params.relationships)} relationships playing role '{params.node_role}'.")

    extracted_ids = set()
    for rel in params.relationships:
        if params.node_role == "source" or params.node_role == "both":
            extracted_ids.add(rel.src_id) # Assuming RelationshipData has src_id
        if params.node_role == "target" or params.node_role == "both":
            extracted_ids.add(rel.tgt_id) # Assuming RelationshipData has tgt_id

    # Dummy results
    return EntityRelNodeOutputs(extracted_entity_ids=list(extracted_ids))


# --- Tool Implementation for: Entity Operator - Link ---
# tool_id: "Entity.Link"

async def entity_link_tool(
    params: EntityLinkInputs,
    graphrag_context: Optional[Any] = None
) -> EntityLinkOutputs:
    """
    Links source entity mentions/objects to canonical entities in a knowledge base or VDB.
    Wraps core GraphRAG logic for entity linking.
    """
    print(f"Executing tool 'Entity.Link' with parameters: {params}")

    # 1. Extract parameters from 'params: EntityLinkInputs'
    #    - source_entities: List[Union[str, ExtractedEntityData]]
    #    - knowledge_base_reference_id: Optional[str]
    #    - similarity_threshold: Optional[float]
    #    - embedding_model_id: Optional[str]

    # Placeholder: For each source_entity, attempt to link it.
    # This would involve embedding, searching the KB/VDB, and applying threshold.
    print(f"Placeholder: Would attempt to link {len(params.source_entities)} entities against KB '{params.knowledge_base_reference_id}'.")

    dummy_linked_results = []
    for i, src_entity in enumerate(params.source_entities):
        mention = src_entity if isinstance(src_entity, str) else getattr(src_entity, 'entity_name', f"unknown_entity_{i}")
        dummy_linked_results.append(
            LinkedEntityPair(
                source_entity_mention=mention,
                linked_entity_id=f"canonical_id_for_{mention}" if (i % 2 == 0) else None,
                linked_entity_description=f"Description of canonical entity for {mention}" if (i % 2 == 0) else None,
                similarity_score=0.85 - (i * 0.05) if (i % 2 == 0) else None,
                link_status="linked" if (i % 2 == 0) else "not_found"
            )
        )

    return EntityLinkOutputs(linked_entities_results=dummy_linked_results)


# --- Tool Implementation for: Entity TF-IDF Ranking ---
# tool_id: "Entity.TFIDF"

from Core.AgentSchema.tool_contracts import EntityTFIDFInputs, EntityTFIDFOutputs

async def entity_tfidf_tool(
    params: EntityTFIDFInputs,
    graphrag_context: Optional[Any] = None
) -> EntityTFIDFOutputs:
    """
    Ranks candidate entities based on TF-IDF scores against a query or context documents.
    Wraps core GraphRAG logic.
    """
    print(f"Executing tool 'Entity.TFIDF' with parameters: {params}")

    # 1. Extract parameters from 'params: EntityTFIDFInputs'
    #    - candidate_entity_ids: List[str]
    #    - query_text: Optional[str]
    #    - context_document_ids: Optional[List[str]]
    #    - corpus_reference_id: str
    #    - top_k_results: Optional[int]

    # Placeholder: Access corpus, build/load TF-IDF matrix, rank entities.
    print(f"Placeholder: Would rank entities {params.candidate_entity_ids} from corpus '{params.corpus_reference_id}' using TF-IDF against query '{params.query_text}'.")

    # Dummy results
    dummy_ranked_entities = []
    num_to_return = params.top_k_results if params.top_k_results is not None else len(params.candidate_entity_ids)
    for i, entity_id in enumerate(params.candidate_entity_ids[:num_to_return]):
        dummy_ranked_entities.append((entity_id, 0.75 - (i * 0.1))) # (entity_id, tfidf_score)

    return EntityTFIDFOutputs(ranked_entities=dummy_ranked_entities)
