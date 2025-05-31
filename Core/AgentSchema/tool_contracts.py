# Core/AgentSchema/tool_contracts.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, Literal, Union
import uuid

# Harmonized imports from Core.Schema
from Core.Schema.EntityRelation import Entity as CoreEntity, Relationship as CoreRelationship
from Core.Schema.ChunkSchema import TextChunk as CoreTextChunk
from Core.Schema.CommunitySchema import LeidenInfo as CoreCommunityInfo

# --- Generic Base Models for Tool Inputs/Outputs (Optional, but can enforce common patterns) ---

class BaseToolParams(BaseModel):
    """Base model for tool-specific parameters, encouraging consistent structure."""
    pass

class BaseToolOutput(BaseModel):
    """Base model for tool-specific outputs, encouraging consistent structure."""
    pass

# --- Tool Contract for: Entity Personalized PageRank (PPR) ---
# Based on conceptual contract: tool_id = "Entity.PPR"

class EntityPPRInputs(BaseToolParams):
    graph_reference_id: str = Field(description="Identifier for the graph artifact to operate on.")
    seed_entity_ids: List[str] = Field(description="List of entity IDs to start PPR from.")
    personalization_weight_alpha: Optional[float] = Field(default=0.15, description="Teleportation probability for PageRank.")
    max_iterations: Optional[int] = Field(default=100, description="Maximum iterations for the PPR algorithm.")
    top_k_results: Optional[int] = Field(default=10, description="Number of top-ranked entities to return.")
    # Add other specific parameters relevant to your PPR implementation

class EntityPPROutputs(BaseToolOutput):
    ranked_entities: List[Tuple[str, float]] = Field(description="List of (entity_id, ppr_score) tuples.")
    # Potentially add metadata about the run, e.g., number of iterations completed.

# --- Tool Contract for: Entity Vector Database Search (VDBSearch) ---
# Based on conceptual contract: tool_id = "Entity.VDBSearch"

class EntityVDBSearchInputs(BaseToolParams):
    vdb_reference_id: str = Field(description="Identifier for the entity vector database.")
    query_text: Optional[str] = Field(default=None, description="Natural language query. Mutually exclusive with query_embedding.")
    query_embedding: Optional[List[float]] = Field(default=None, description="Pre-computed query embedding. Mutually exclusive with query_text.")
    embedding_model_id: Optional[str] = Field(default=None, description="Identifier for embedding model if query_text is used.")
    top_k_results: int = Field(default=5, description="Number of top similar entities to return.")
    # Add other parameters like filtering conditions, etc.

class VDBSearchResultItem(BaseModel):
    node_id: str = Field(description="Internal ID of the node in the VDB (e.g., LlamaIndex TextNode ID).")
    entity_name: str = Field(description="The actual name/identifier of the entity used in the graph.")
    score: float = Field(description="Similarity score from the VDB search.")

class EntityVDBSearchOutputs(BaseToolOutput):
    similar_entities: List[VDBSearchResultItem] = Field(description="List of VDB search result items.")

# --- Tool Contract for: Chunks From Relationships (FromRelationships) ---
# Based on conceptual contract: tool_id = "Chunk.FromRelationships"

class ChunkFromRelationshipsInputs(BaseToolParams):
    target_relationships: List[Union[str, Dict[str, str]]] = Field(description="List of relationship identifiers (simple names or structured queries).")
    document_collection_id: str = Field(description="Identifier for the source document/chunk collection.")
    max_chunks_per_relationship: Optional[int] = Field(default=None, description="Optional limit on chunks per relationship type.")
    top_k_total: Optional[int] = Field(default=10, description="Optional overall limit on the number of chunks returned.")
    # Add other parameters like desired chunk context size, etc.

# --- Harmonized: ChunkData inherits from canonical CoreTextChunk ---
class ChunkData(CoreTextChunk):
    relevance_score: Optional[float] = Field(default=None, description="Score indicating relevance to the query or operation.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata specific to this tool's output for the chunk.")
    # Inherits all fields from CoreTextChunk (tokens, chunk_id, content/text, doc_id, index, title)
    pass

class ChunkFromRelationshipsOutputs(BaseToolOutput):
    relevant_chunks: List[ChunkData] = Field(description="List of structured chunk data.")

# --- Tool Contract for: K-Hop Paths (KHopPaths) ---
# Based on conceptual contract: tool_id = "Subgraph.KHopPaths"

# --- Harmonized: PathSegment represents either an entity or relationship in a path, with reference to core schema concepts ---
class PathSegment(BaseModel):
    item_id: str  # ID of the entity (CoreEntity.entity_name or ID) or relationship (CoreRelationship.relationship_id)
    item_type: Literal["entity", "relationship"]
    label: Optional[str] = None  # e.g., CoreEntity.entity_name or CoreRelationship.type
    # Optionally, could add: item_data: Optional[Union[CoreEntity, CoreRelationship]] = None

# --- Harmonized: PathObject is a structured sequence of PathSegments, referencing core schema IDs ---
class PathObject(BaseModel):
    path_id: str = Field(default_factory=lambda: f"path_{uuid.uuid4().hex[:8]}")
    segments: List[PathSegment]  # List of PathSegment items
    start_node_id: str  # Should correspond to CoreEntity.entity_name or ID
    end_node_id: Optional[str] = None  # Should correspond to CoreEntity.entity_name or ID
    hop_count: int

class SubgraphKHopPathsInputs(BaseToolParams):
    graph_reference_id: str
    start_entity_ids: List[str]
    end_entity_ids: Optional[List[str]] = Field(default=None, description="If None, finds k-hop neighborhoods from start_entity_ids.")
    k_hops: int = Field(default=2, ge=1, description="Maximum number of hops.")
    max_paths_to_return: Optional[int] = Field(default=10)
    # Add other parameters like relationship types to traverse, etc.

class SubgraphKHopPathsOutputs(BaseToolOutput):
    discovered_paths: List[PathObject] = Field(description="List of discovered paths, each represented as a structured object.")

# --- Tool Contract for: Relationship One-Hop Neighbors (OneHopNeighbors) ---
# Based on conceptual contract: tool_id = "Relationship.OneHopNeighbors"

# --- Harmonized: RelationshipData inherits from canonical CoreRelationship ---
class RelationshipData(CoreRelationship):
    relevance_score: Optional[float] = Field(default=None, description="Relevance score if applicable, e.g., from VDB search.")
    # Inherits all fields from CoreRelationship (relationship_id, source_node_id, target_node_id, type, etc.)
    pass

class RelationshipOneHopNeighborsInputs(BaseToolParams):
    entity_ids: List[str]
    graph_reference_id: str = Field(default="kg_graph", description="Reference ID for the graph to use (e.g., 'kg_graph').")
    relationship_types_to_include: Optional[List[str]] = None
    direction: Optional[Literal["outgoing", "incoming", "both"]] = Field(default="both")

class RelationshipOneHopNeighborsOutputs(BaseToolOutput):
    one_hop_relationships: List[RelationshipData]

# --- Tool Contract for: Chunk Aggregator based on Relationships (RelationshipScoreAggregator) ---
# Based on conceptual contract: tool_id = "Chunk.RelationshipScoreAggregator"

class ChunkRelationshipScoreAggregatorInputs(BaseToolParams):
    # Assuming chunks already have entity and relationship mentions within them or linked
    chunk_candidates: List[ChunkData] # Or List[str] of chunk_ids if chunks are fetched separately
    relationship_scores: Dict[str, float] # Key: relationship_id or type, Value: score
    # This operator will need a clear way to know which relationships are in which chunk_candidate
    # This might require chunk_candidates to be objects with pre-processed relationship info,
    # or providing additional mappings.
    top_k_chunks: int

class ChunkRelationshipScoreAggregatorOutputs(BaseToolOutput):
    ranked_aggregated_chunks: List[ChunkData] # Chunks with an added aggregated_score field, or List[Tuple[str, float]]

# TODO:
# - Define Pydantic models for other operators from README.md.
# - Consider if 'parameters' in ToolCall should be a Union of these specific input models
#   for stronger type checking, e.g., parameters: Union[EntityPPRInputs, EntityVDBSearchInputs, ...].
#   This makes the ToolCall model more complex but safer.
# - These input/output models will need to be aligned with the actual Python function
#   signatures of the underlying operator implementations in your Core modules.
# Instructions for LLM IDE: End
# Explanation and Purpose of this Step:
#
# Concrete Data Structures: This file, Core/AgentSchema/tool_contracts.py, starts to give us concrete Pydantic data structures for the inputs and outputs of individual tools (your KG operators).
# Type Safety & Clarity: Using specific Pydantic models (like EntityPPRInputs, EntityPPROutputs) for each tool's parameters and results, instead of just generic dictionaries, provides better type safety, auto-validation, and makes it clearer what each tool expects and returns.
# Informing the ToolCall Model:
# In our main Core/AgentSchema/plan.py file, the ToolCall.parameters field (which is currently Optional[Dict[str, Any]]) could eventually be changed to something like parameters: Union[EntityPPRInputs, EntityVDBSearchInputs, ChunkFromRelationshipsInputs, ...] if we want the LLM agent to produce a plan with these strongly-typed parameter objects for each tool.
# Alternatively, the ToolCall.parameters: Dict[str, Any] can still be used, and the "Agent Orchestrator" would be responsible for validating that dictionary against the corresponding Pydantic input model (e.g., EntityPPRInputs) when it's about to execute the "Entity.PPR" tool.
# Foundation for Orchestrator & LLM Agent:
# Orchestrator: When the orchestrator sees a ToolCall with tool_id: "Entity.PPR", it will know to expect parameters that fit the EntityPPRInputs model and that the tool will produce something fitting EntityPPROutputs.
# LLM Agent: The description fields and parameter names/types in these Pydantic models will form part of the "knowledge base" or prompt context for the LLM agent, helping it understand how to correctly use each tool.

# --- Tool Contract for: Entity TF-IDF Ranking ---
# Based on README.md operator: Entity Operators - TF-IDF "Rank entities based on the TF-IFG matrix"
# Assuming TF-IDF is used to rank a given set of candidate entities against a query or document set.

class EntityTFIDFInputs(BaseToolParams):
    candidate_entity_ids: List[str] = Field(description="List of entity IDs to be ranked.")
    query_text: Optional[str] = Field(default=None, description="Query text to rank entities against.")
    # Alternatively, context could be defined by a set of document/chunk IDs
    context_document_ids: Optional[List[str]] = Field(default=None, description="List of document IDs providing context for TF-IDF calculation.")
    corpus_reference_id: str = Field(description="Identifier for the corpus or document collection where TF-IDF matrix is built or can be derived.")
    top_k_results: Optional[int] = Field(default=10, description="Number of top-ranked entities to return.")

class EntityTFIDFOutputs(BaseToolOutput):
    ranked_entities: List[Tuple[str, float]] = Field(description="List of (entity_id, tfidf_score) tuples.")

# --- Tool Contract for: Relationship Vector Database Search (VDB) ---
# Based on README.md operator: Relationship Operators - VDB "Retrieve relationships by vector-database"

class RelationshipVDBSearchInputs(BaseToolParams):
    vdb_reference_id: str = Field(description="Identifier for the relationship vector database.")
    query_text: Optional[str] = Field(default=None, description="Natural language query for relationships.")
    query_embedding: Optional[List[float]] = Field(default=None, description="Pre-computed query embedding.")
    embedding_model_id: Optional[str] = Field(default=None, description="Identifier for embedding model if query_text is used.")
    top_k_results: int = Field(default=5, description="Number of top similar relationships to return.")
    # Add other parameters like filtering conditions for relationship types, etc.

class RelationshipVDBSearchOutputs(BaseToolOutput):
    # Assuming RelationshipData is already defined in this file (from previous batch)
    #
    similar_relationships: List[Tuple[RelationshipData, float]] = Field(description="List of (RelationshipData, similarity_score) tuples.")

# --- Tool Contract for: Community Detection from Entities ---
# Based on README.md operator: Community Operators - Entity "Detects communities containing specified entities"
# This might involve running a community detection algorithm (like Leiden) and then filtering/identifying communities relevant to seed entities.

class CommunityDetectFromEntitiesInputs(BaseToolParams):
    graph_reference_id: str = Field(description="Identifier for the graph artifact.")
    seed_entity_ids: List[str] = Field(description="List of entity IDs to find relevant communities for.")
    community_algorithm: Optional[str] = Field(default="leiden", description="Algorithm to use for community detection if not already computed, e.g., 'leiden'.")
    # Parameters for the community detection algorithm itself could go here if needed
    # e.g., resolution_parameter: Optional[float] for Leiden
    max_communities_to_return: Optional[int] = Field(default=5)

# --- Harmonized: CommunityData inherits from canonical CoreCommunityInfo (LeidenInfo) ---
class CommunityData(CoreCommunityInfo):
    community_id: str  # This might be redundant if 'title' from LeidenInfo is used as ID, or could be a specific ID assigned during this tool's operation.
    description: Optional[str] = Field(default=None, description="Optional LLM-generated or tool-derived summary of the community.")
    # Inherits all fields from CoreCommunityInfo/LeidenInfo (level, title, edges, nodes, chunk_ids, occurrence, sub_communities)
    pass

class CommunityDetectFromEntitiesOutputs(BaseToolOutput):
    relevant_communities: List[CommunityData] = Field(description="List of communities relevant to the seed entities.")

# TODO for next steps:
# - Continue adding Pydantic Input/Output models for the remaining ~10 operators from README.md.
#   Examples:
#     - Entity.Agent (LLM to find entities)
#     - Relationship.Agent (LLM to find relationships)
#     - Chunk.Occurrence
#     - Subgraph.SteinerTree
#     - Subgraph.AgentPath
#     - Community.Layer
# - Refine these models as we get closer to mapping them to actual Python functions in Core modules.
# - The Agent Orchestrator will use these models to validate parameters for ToolCalls and
#   to understand the structure of data being passed between tools.

# --- Tool Contract for: Entity Operator - RelNode ---
# Based on README.md operator: Entity Operators - RelNode "Extract nodes from given relationships"
# This tool likely takes a list of relationship objects (or their IDs) and extracts unique entities involved in them.

class EntityRelNodeInputs(BaseToolParams):
    # Assuming RelationshipData is already defined in this file from previous batches
    #
    relationships: List[RelationshipData] = Field(description="List of relationship objects from which to extract nodes.")
    # Or, if only IDs are passed and relationships need to be fetched:
    # relationship_ids: List[str]
    # graph_reference_id: Optional[str] # If relationship_ids are passed, graph might be needed to fetch them

    node_role: Optional[Literal["source", "target", "both"]] = Field(default="both", description="Which role(s) the nodes play in the relationships (source, target, or both).")

class EntityRelNodeOutputs(BaseToolOutput):
    # Assuming EntityData Pydantic model would be defined if we want structured entity output
    # For now, returning IDs. Could align with Core/Schema/EntityRelation.py
    extracted_entity_ids: List[str] = Field(description="List of unique entity IDs extracted from the given relationships.")

# --- Tool Contract for: Chunk Operator - Occurrence ---
# Based on README.md operator: Chunk Operators - Occurrence "Rank top-k chunks based on occurrence of both entities in relationships"
# This implies we have entities that form relationships, and we want chunks where these related entities co-occur.

class ChunkOccurrenceInputs(BaseToolParams):
    # Assuming EntityPairInRelationship is a Pydantic model or dict like:
    # {"entity1_id": "id1", "entity2_id": "id2", "relationship_type": "optional_rel_type"}
    # Or perhaps a list of RelationshipData objects
    #
    target_entity_pairs_in_relationship: List[Dict[str, str]] = Field(description="List of entity pairs (and optionally their relationship type) whose co-occurrence in chunks is sought.")
    
    document_collection_id: str = Field(description="Identifier for the source document/chunk collection.")
    # How are chunks initially retrieved or filtered before ranking?
    # candidate_chunk_ids: Optional[List[str]] = Field(default=None, description="Optional list of chunk IDs to rank. If None, might search all chunks in the collection.")
    
    top_k_chunks: int = Field(default=5, description="Number of top-ranked chunks to return.")
    # Add parameters for ranking algorithm if any (e.g., weighting schemes)

class ChunkOccurrenceOutputs(BaseToolOutput):
    # Assuming ChunkData is already defined in this file
    #
    ranked_occurrence_chunks: List[ChunkData] = Field(description="List of ranked chunk data, potentially with co-occurrence scores or explanations.")

# --- Tool Contract for: Subgraph Operator - SteinerTree ---
# Based on README.md operator: Subgraph Operators - Steiner "Compute Steiner tree based on given entities and relationships"

class SubgraphSteinerTreeInputs(BaseToolParams):
    graph_reference_id: str = Field(description="Identifier for the graph artifact.")
    terminal_node_ids: List[str] = Field(description="List of entity/node IDs that must be included in the Steiner tree.")
    # Optional: Weight attribute for edges if the algorithm considers edge weights
    edge_weight_attribute: Optional[str] = Field(default=None, description="Name of the edge attribute to use for weights (if any).")
    # Other algorithm-specific parameters

class SubgraphSteinerTreeOutputs(BaseToolOutput):
    # Output could be a list of edges forming the tree, or a reference to a new subgraph artifact.
    # For now, let's assume a list of edges. Each edge could be a tuple or a structured object.
    # Align with Core/Schema/GraphSchema.py if possible.
    steiner_tree_edges: List[Dict[str, Any]] = Field(description="List of edges (e.g., {'source': 'id1', 'target': 'id2', 'weight': 0.5}) forming the Steiner tree.")
    # Or:
    # steiner_tree_subgraph_reference_id: str = Field(description="Identifier for a new graph artifact representing the Steiner tree.")

# TODO for next steps:
# - Continue adding Pydantic Input/Output models for the remaining operators from README.md.
#   Remaining examples:
#     - Entity.Agent (Utilizes LLM to find the useful entities)
#     - Entity.Link (Return top-1 similar entity for each given entity)
#     - Relationship.Agent (Utilizes LLM to find useful relationships)
#     - Relationship.Aggregator (Compute relationship scores from entity PPR matrix)
#     - Subgraph.AgentPath (Identify relevant k-hop paths using LLM)
#     - Community.Layer (Returns all communities below a required layer)
# - As we define these, we'll get a clearer picture of common parameter patterns and output structures,
#   which will help in finalizing the ToolCall model in plan.py and designing the Agent Orchestrator.

# --- Tool Contract for: Entity Operator - Agent ---
# Based on README.md operator: Entity Operators - Agent "Utilizes LLM to find the useful entities"
# This tool uses an LLM to extract or identify relevant entities from a given context (e.g., query, text chunks).

class EntityAgentInputs(BaseToolParams):
    query_text: str = Field(description="The user query or task description to guide entity extraction.")
    text_context: Union[str, List[str]] = Field(description="The text content (or list of text chunks) from which to extract entities.")
    # Potentially, a list of existing entity IDs to avoid re-extracting or to provide context
    existing_entity_ids: Optional[List[str]] = Field(default=None)
    # Ontology context: what types of entities to look for?
    target_entity_types: Optional[List[str]] = Field(default=None, description="Specific entity types the LLM should focus on extracting (e.g., ['person', 'organization']).")
    llm_config_override_patch: Optional[Dict[str, Any]] = Field(default=None, description="Optional patch to apply to the global LLM configuration for this specific tool call.")
    max_entities_to_extract: Optional[int] = Field(default=10)
    # Could include a specific prompt template if the LLM call is highly specialized
    # prompt_template_id: Optional[str] = None

# --- Harmonized: ExtractedEntityData inherits from canonical CoreEntity ---
class ExtractedEntityData(CoreEntity):
    extraction_confidence: Optional[float] = Field(default=None, description="LLM's confidence in this extraction.")
    # Inherits all fields from CoreEntity (entity_name, source_id, entity_type, description, attributes)
    pass

class EntityAgentOutputs(BaseToolOutput):
    extracted_entities: List[ExtractedEntityData] = Field(description="List of entities identified or extracted by the LLM.")

# --- Tool Contract for: Relationship Operator - Agent ---
# Based on README.md operator: Relationship Operators - Agent "Utilizes LLM to find the useful relationships"

class RelationshipAgentInputs(BaseToolParams):
    query_text: str = Field(description="The user query or task description to guide relationship extraction.")
    text_context: Union[str, List[str]] = Field(description="The text content (or list of text chunks) from which to extract relationships.")
    # Context of known entities is crucial for finding relationships between them
    context_entities: List[ExtractedEntityData] # Or List[str] of entity_ids if full data isn't needed for the prompt
    target_relationship_types: Optional[List[str]] = Field(default=None, description="Specific relationship types the LLM should focus on (e.g., ['works_for', 'located_in']).")
    llm_config_override_patch: Optional[Dict[str, Any]] = Field(default=None, description="Optional patch for LLM configuration.")
    max_relationships_to_extract: Optional[int] = Field(default=10)

class RelationshipAgentOutputs(BaseToolOutput):
    # Assuming RelationshipData is already defined in this file
    #
    extracted_relationships: List[RelationshipData] = Field(description="List of relationships identified or extracted by the LLM.")

# --- Tool Contract for: Subgraph Operator - AgentPath ---
# Based on README.md operator: Subgraph Operators - AgentPath "Identify the most relevant k-hop paths to a given question, by using LLM to filter out the irrelevant paths"
# This implies a two-stage process: 1. Generate candidate paths (e.g., using KHopPaths tool). 2. LLM filters/ranks these paths.
# This tool might just represent the LLM filtering part.

class SubgraphAgentPathInputs(BaseToolParams):
    user_question: str = Field(description="The original question to determine path relevance.")
    # Assuming PathObject is already defined in this file
    #
    candidate_paths: List[PathObject] = Field(description="List of candidate paths to be filtered/ranked by the LLM.")
    llm_config_override_patch: Optional[Dict[str, Any]] = Field(default=None, description="Optional patch for LLM configuration.")
    max_paths_to_return: Optional[int] = Field(default=5)
    # Criteria for relevance could be an input too, or part of the prompt
    # relevance_criteria_prompt: Optional[str] = None

class SubgraphAgentPathOutputs(BaseToolOutput):
    relevant_paths: List[PathObject] = Field(description="List of paths deemed most relevant by the LLM, possibly with relevance scores/explanations.")
    # Or: ranked_paths: List[Tuple[PathObject, float]]

# --- Tool Contract for: Entity Operator - Link ---
# Based on README.md operator: Entity Operators - Link "Return top-1 similar entity for each given entity"
# This sounds like an entity linking or canonicalization step, perhaps finding the closest match in a knowledge base or VDB.

class EntityLinkInputs(BaseToolParams):
    source_entities: List[Union[str, ExtractedEntityData]] = Field(description="List of entity mentions (strings) or preliminary entity objects to be linked.")
    # Target for linking:
    knowledge_base_reference_id: Optional[str] = Field(default=None, description="Identifier for a target knowledge base or entity VDB to link against.")
    # Or, if linking within a pre-defined set:
    # candidate_target_entity_ids: Optional[List[str]] = None
    similarity_threshold: Optional[float] = Field(default=None, description="Optional threshold for a link to be considered valid.")
    # May involve an embedding model if similarity is embedding-based
    embedding_model_id: Optional[str] = Field(default=None, description="Identifier for embedding model if needed.")

class LinkedEntityPair(BaseModel):
    source_entity_mention: str # Or the original ExtractedEntityData
    linked_entity_id: Optional[str] = None
    linked_entity_description: Optional[str] = None
    similarity_score: Optional[float] = None
    link_status: Literal["linked", "ambiguous", "not_found"]

class EntityLinkOutputs(BaseToolOutput):
    linked_entities_results: List[LinkedEntityPair] = Field(description="Results of the entity linking process for each source entity.")

# TODO for next steps:
# - Define Pydantic Input/Output models for the few remaining operators:
#     - Relationship.Aggregator
#     - Community.Layer
# - This batch focused on "Agent" tools; their implementation in the Orchestrator will involve an LLM call.
# - The `ExtractedEntityData` and `RelationshipData` used here should be harmonized with your main schema
#   definitions in Core/Schema/EntityRelation.py and Core/Schema/CommunitySchema.py
#   (e.g., by importing and using them, or ensuring field compatibility).

# --- Tool Contract for: Relationship Operator - Score Aggregator ---
# Based on README.md operator: Relationship Operators - Aggregator "Compute relationship scores from entity PPR matrix, return top-k"
# This tool likely takes entity scores (e.g., from PPR) and uses them to score associated relationships.

class RelationshipScoreAggregatorInputs(BaseToolParams):
    # Assuming EntityPPROutputs or similar (containing entity scores) is available
    # from a previous step or can be referenced.
    # For now, let's assume a dictionary of entity_id to score.
    entity_scores: Dict[str, float] = Field(description="Dictionary of entity IDs to their scores (e.g., from PPR).")
    graph_reference_id: str = Field(description="Identifier for the graph artifact to fetch relationships.")
    # How to determine which relationships are associated with the scored entities?
    # Option 1: Fetch all relationships for scored entities.
    # Option 2: Provide a list of candidate relationship_ids or RelationshipData objects.
    # For now, assuming it fetches relationships of scored entities.
    top_k_relationships: Optional[int] = Field(default=10, description="Number of top-scored relationships to return.")
    aggregation_method: Optional[Literal["sum", "average", "max"]] = Field(default="sum", description="Method to aggregate entity scores onto relationships.")

class RelationshipScoreAggregatorOutputs(BaseToolOutput):
    # Assuming RelationshipData is already defined in this file
    #
    # We'll add a score to it.
    scored_relationships: List[Tuple[RelationshipData, float]] = Field(description="List of (RelationshipData, aggregated_score) tuples.")

# --- Tool Contract for: Community Operator - Get Layer ---
# Based on README.md operator: Community Operators - Layer "Returns all communities below a required layer"
# This assumes a hierarchical community structure has been previously computed and stored.

class CommunityGetLayerInputs(BaseToolParams):
    community_hierarchy_reference_id: str = Field(description="Identifier for the stored community hierarchy artifact (e.g., from LeidenCommunity storage).")
    max_layer_depth: int = Field(description="The maximum layer depth to retrieve communities from (e.g., 0 for top-level, 1 for next level down).")
    # Optional: filter by specific parent community IDs if needed
    # parent_community_ids: Optional[List[str]] = None

class CommunityGetLayerOutputs(BaseToolOutput):
    # Assuming CommunityData is already defined in this file
    #
    communities_in_layers: List[CommunityData] = Field(description="List of communities found at or below the specified layer depth.")

# TODO for next steps (updated):
# 1. Schema Harmonization: Critically review all Pydantic models in this file (tool_contracts.py)
#    and in Core/AgentSchema/plan.py. Ensure that where placeholder types like Dict[str, Any] or
#    simple type hints like List[str] for IDs are used, they are updated to reference or align with
#    the more detailed Pydantic models from your Core/Schema/ directory
#    (e.g., Core.Schema.EntityRelation.Entity, 
#     Core.Schema.ChunkSchema.TextChunk, 
#     Core.Schema.CommunitySchema.CommunityReport, etc.).
#    This is a key refactoring step for type consistency and data integrity.
# 2. Mapping to Existing Code: Begin the process of mapping each `tool_id` and its
#    Pydantic input/output contract to actual, callable Python functions/methods in your Core modules.
#    This will likely involve some refactoring or writing new wrapper functions.
# 3. Design and Implement the "Agent Orchestrator".
# 4. Develop LLM Agent Prompts and integrate with a framework like PydanticAI/LiteLLM.
