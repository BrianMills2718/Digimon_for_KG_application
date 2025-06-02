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

# --- Tool Contract for: Entity One-Hop Neighbors (OneHopNeighbors) ---
# Based on conceptual contract: tool_id = "Entity.OneHopNeighbors"

class EntityOneHopInput(BaseModel):
    """Input for finding one-hop neighbor entities."""
    entity_ids: List[str] = Field(
        description="List of entity IDs to find neighbors for"
    )
    graph_id: str = Field(
        description="The ID of the graph to search in"
    )
    include_edge_attributes: Optional[bool] = Field(
        default=False,
        description="Whether to include edge attributes in the results"
    )
    neighbor_limit_per_entity: Optional[int] = Field(
        default=None,
        description="Maximum number of neighbors to return per entity. If None, returns all neighbors."
    )

class EntityOneHopOutput(BaseModel):
    """Output containing one-hop neighbor entities."""
    neighbors: Dict[str, List[Dict[str, Any]]] = Field(
        description="Dictionary mapping each input entity ID to its list of neighbor entities with their attributes"
    )
    total_neighbors_found: int = Field(
        description="Total number of unique neighbor entities found"
    )
    message: str = Field(description="Status message or error description")

# --- Tool Contract for: Community Detection from Entities ---
# Based on conceptual contract: tool_id = "Community.Entity"

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

# --- Tool Contract for: Relationship VDB Build ---
# Build a vector database index for relationships

class RelationshipVDBBuildInputs(BaseToolParams):
    graph_reference_id: str = Field(description="ID of the graph containing relationships to index.")
    vdb_collection_name: str = Field(description="Name for the VDB collection to create.")
    embedding_fields: List[str] = Field(default=["type", "description"], description="Fields from relationships to embed and index.")
    include_metadata: bool = Field(default=True, description="Whether to include relationship metadata in the index.")
    force_rebuild: bool = Field(default=False, description="Force rebuild even if index exists.")

class RelationshipVDBBuildOutputs(BaseToolOutput):
    vdb_reference_id: str = Field(description="ID of the created VDB for future reference.")
    num_relationships_indexed: int = Field(description="Number of relationships indexed.")
    status: str = Field(description="Status message about the build process.")

# --- Tool Contract for: Relationship VDB Search ---
# Search for similar relationships using vector similarity

class RelationshipVDBSearchInputs(BaseToolParams):
    vdb_reference_id: str = Field(description="ID of the VDB to search.")
    query_text: Optional[str] = Field(default=None, description="Text query to search for similar relationships.")
    query_embedding: Optional[List[float]] = Field(default=None, description="Pre-computed embedding vector for search.")
    top_k: int = Field(default=10, description="Number of most similar relationships to return.")
    score_threshold: Optional[float] = Field(default=None, description="Minimum similarity score threshold.")

class RelationshipVDBSearchOutputs(BaseToolOutput):
    similar_relationships: List[Tuple[str, str, float]] = Field(
        description="List of (relationship_id, relationship_description, similarity_score) tuples."
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional search metadata.")

# --- Tool Contract for: Entity VDB Build ---
# Build a vector database index for entities

class EntityVDBBuildInputs(BaseToolParams):
    graph_reference_id: str = Field(description="ID of the graph containing entities to index.")
    vdb_collection_name: str = Field(description="Name for the VDB collection to create.")
    entity_types: Optional[List[str]] = Field(default=None, description="Specific entity types to include. If None, includes all entities.")
    include_metadata: bool = Field(default=True, description="Whether to include entity metadata in the index.")
    force_rebuild: bool = Field(default=False, description="Force rebuild even if index exists.")

class EntityVDBBuildOutputs(BaseToolOutput):
    vdb_reference_id: str = Field(description="ID of the created VDB for future reference.")
    num_entities_indexed: int = Field(description="Number of entities indexed.")
    status: str = Field(description="Status message about the build process.")

# --- Tool Contract for: Graph Visualizer ---
# Purpose: Take a graph_id as input and provide a representation of the graph suitable for visualization

class GraphVisualizerInput(BaseToolParams):
    graph_id: str = Field(description="ID of the graph to visualize (the artifact name)")
    output_format: Optional[str] = Field(default="JSON_NODES_EDGES", description="Output format: 'GML', 'JSON_NODES_EDGES'")

class GraphVisualizerOutput(BaseToolOutput):
    graph_representation: str = Field(description="The graph data in the specified format")
    format_used: str = Field(description="The actual format returned")
    message: Optional[str] = Field(default=None, description="Error message or additional info")

# --- Tool Contract for: Graph Analyzer ---
# Purpose: Take a graph_id as input and provide various metrics and statistics about the graph

class GraphAnalyzerInput(BaseModel):
    graph_id: str = Field(description="The ID of the graph to analyze")
    metrics_to_calculate: Optional[List[str]] = Field(
        default=None,
        description="Specific metrics to calculate. If None, calculates all available metrics. Options: 'basic_stats', 'centrality', 'clustering', 'connectivity', 'components', 'paths'"
    )
    top_k_nodes: Optional[int] = Field(
        default=10,
        description="For centrality metrics, return only the top K nodes by each centrality measure"
    )
    calculate_expensive_metrics: bool = Field(
        default=False,
        description="Whether to calculate computationally expensive metrics like betweenness centrality and diameter for large graphs"
    )

class GraphAnalyzerOutput(BaseToolOutput):
    graph_id: str = Field(description="The ID of the analyzed graph")
    basic_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Basic graph statistics: node_count, edge_count, density, is_directed, is_connected"
    )
    centrality_metrics: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Centrality measures for nodes: degree, closeness, betweenness, eigenvector, pagerank"
    )
    clustering_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Clustering metrics: average_clustering, transitivity, triangles"
    )
    connectivity_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Connectivity metrics: is_connected, number_connected_components, largest_component_size"
    )
    component_details: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Details about connected components"
    )
    path_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Path-related metrics: diameter, average_shortest_path_length, radius"
    )
    message: str = Field(default="", description="Status message or warnings about the analysis")

# Entity RelNode Tool Contracts
class EntityRelNodeInput(BaseModel):
    """Input for extracting entities connected by specific relationships."""
    relationship_ids: List[str] = Field(
        description="List of relationship IDs to extract entities from"
    )
    graph_id: str = Field(
        description="The ID of the graph to search in"
    )
    entity_role_filter: Optional[str] = Field(
        default=None,
        description="Filter for entity role in relationship. Options: 'source', 'target', 'both'"
    )
    entity_type_filter: Optional[List[str]] = Field(
        default=None,
        description="Filter entities by type (e.g., ['Person', 'Organization'])"
    )

class EntityRelNodeOutput(BaseModel):
    """Output containing entities extracted from relationships."""
    entities: List[Dict[str, Any]] = Field(
        description="List of entities with their attributes and relationship connections"
    )
    entity_count: int = Field(
        description="Total number of unique entities found"
    )
    relationship_entity_map: Dict[str, List[str]] = Field(
        description="Map of relationship_id to list of entity_ids involved"
    )
    message: str = Field(description="Status message or error description")

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
