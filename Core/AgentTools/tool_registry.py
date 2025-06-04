# Core/AgentTools/tool_registry.py

from typing import Dict, List, Callable, Type, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import inspect
from pydantic import BaseModel
from Core.Common.Logger import logger


class ToolCategory(Enum):
    """Tool categorization for execution optimization"""
    READ_ONLY = "read_only"      # Can be parallelized
    WRITE = "write"               # Must be sequential
    BUILD = "build"               # Heavy operations, sequential
    TRANSFORM = "transform"       # Data transformation
    ANALYZE = "analyze"          # Analysis operations


class ToolCapability(Enum):
    """Tool capabilities for planning"""
    ENTITY_DISCOVERY = "entity_discovery"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"
    GRAPH_CONSTRUCTION = "graph_construction"
    TEXT_RETRIEVAL = "text_retrieval"
    VECTOR_SEARCH = "vector_search"
    DATA_PREPARATION = "data_preparation"
    VISUALIZATION = "visualization"
    ANALYSIS = "analysis"


@dataclass
class ToolMetadata:
    """Metadata for registered tools"""
    tool_id: str
    name: str
    description: str
    category: ToolCategory
    capabilities: Set[ToolCapability]
    input_model: Type[BaseModel]
    output_model: Optional[Type[BaseModel]] = None
    version: str = "1.0.0"
    author: str = "DIGIMON"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    performance_hint: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_parallelizable(self) -> bool:
        """Check if tool can be run in parallel"""
        return self.category in [ToolCategory.READ_ONLY, ToolCategory.ANALYZE]
    
    @property
    def requires_context(self) -> bool:
        """Check if tool requires GraphRAG context"""
        return self.tool_id not in ["corpus.PrepareFromDirectory"] and not self.tool_id.startswith("graph.Build")


@dataclass
class ToolRegistration:
    """Complete tool registration information"""
    metadata: ToolMetadata
    function: Callable
    validator: Optional[Callable] = None
    pre_processor: Optional[Callable] = None
    post_processor: Optional[Callable] = None
    

class DynamicToolRegistry:
    """
    Enhanced tool registry with dynamic registration, categorization,
    and discovery capabilities.
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolRegistration] = {}
        self._capability_index: Dict[ToolCapability, Set[str]] = {cap: set() for cap in ToolCapability}
        self._category_index: Dict[ToolCategory, Set[str]] = {cat: set() for cat in ToolCategory}
        self._initialize_default_tools()
        
    def _initialize_default_tools(self):
        """Initialize with default DIGIMON tools"""
        # Import tool functions
        from Core.AgentTools.entity_tools import entity_vdb_search_tool, entity_ppr_tool
        from Core.AgentTools.entity_vdb_tools import entity_vdb_build_tool
        from Core.AgentTools.entity_onehop_tools import entity_onehop_neighbors_tool
        from Core.AgentTools.entity_relnode_tools import entity_relnode_extract_tool
        from Core.AgentTools.relationship_tools import (
            relationship_one_hop_neighbors_tool, 
            relationship_vdb_build_tool, 
            relationship_vdb_search_tool
        )
        from Core.AgentTools.chunk_tools import chunk_from_relationships_tool, chunk_get_text_for_entities_tool
        from Core.AgentTools.graph_construction_tools import (
            build_er_graph, build_rk_graph, build_tree_graph, 
            build_tree_graph_balanced, build_passage_graph
        )
        from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
        from Core.AgentTools.graph_visualization_tools import visualize_graph
        from Core.AgentTools.graph_analysis_tools import analyze_graph
        
        # Import social media analysis tools
        try:
            from Core.AgentTools.social_media_dataset_tools import (
                ingest_covid_conspiracy_dataset, DatasetIngestionInput
            )
            from Core.AgentTools.automated_interrogative_planner import (
                generate_interrogative_analysis_plans, AutoInterrogativePlanInput
            )
        except ImportError:
            logger.warning("Social media analysis tools not found, skipping registration")
        
        # Import input models
        from Core.AgentSchema.tool_contracts import (
            EntityVDBSearchInputs, EntityVDBBuildInputs, EntityPPRInputs,
            EntityOneHopInput, EntityRelNodeInput, RelationshipOneHopNeighborsInputs,
            RelationshipVDBBuildInputs, RelationshipVDBSearchInputs,
            ChunkFromRelationshipsInputs, ChunkGetTextForEntitiesInput,
            GraphVisualizerInput, GraphAnalyzerInput
        )
        from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs
        from Core.AgentSchema.graph_construction_tool_contracts import (
            BuildERGraphInputs, BuildRKGraphInputs, BuildTreeGraphInputs,
            BuildTreeGraphBalancedInputs, BuildPassageGraphInputs
        )
        
        # Register entity tools
        self.register_tool(
            tool_id="Entity.VDBSearch",
            function=entity_vdb_search_tool,
            metadata=ToolMetadata(
                tool_id="Entity.VDBSearch",
                name="Entity Vector Database Search",
                description="Search for similar entities using vector embeddings",
                category=ToolCategory.READ_ONLY,
                capabilities={ToolCapability.ENTITY_DISCOVERY, ToolCapability.VECTOR_SEARCH},
                input_model=EntityVDBSearchInputs,
                tags=["entity", "search", "embeddings"],
                performance_hint="Fast for <1000 entities, use batching for larger sets"
            )
        )
        
        self.register_tool(
            tool_id="Entity.VDB.Build",
            function=entity_vdb_build_tool,
            metadata=ToolMetadata(
                tool_id="Entity.VDB.Build",
                name="Entity Vector Database Builder",
                description="Build vector database index for entities",
                category=ToolCategory.WRITE,
                capabilities={ToolCapability.ENTITY_DISCOVERY, ToolCapability.DATA_PREPARATION},
                input_model=EntityVDBBuildInputs,
                tags=["entity", "build", "index"],
                performance_hint="CPU intensive, benefits from GPU acceleration"
            )
        )
        
        self.register_tool(
            tool_id="Entity.PPR",
            function=entity_ppr_tool,
            metadata=ToolMetadata(
                tool_id="Entity.PPR",
                name="Entity Personalized PageRank",
                description="Rank entities using Personalized PageRank algorithm",
                category=ToolCategory.READ_ONLY,
                capabilities={ToolCapability.ENTITY_DISCOVERY, ToolCapability.ANALYSIS},
                input_model=EntityPPRInputs,
                tags=["entity", "ranking", "pagerank"],
                performance_hint="Computation scales with graph size"
            )
        )
        
        self.register_tool(
            tool_id="Entity.Onehop",
            function=entity_onehop_neighbors_tool,
            metadata=ToolMetadata(
                tool_id="Entity.Onehop",
                name="Entity One-Hop Neighbors",
                description="Find one-hop neighbor entities in the graph",
                category=ToolCategory.READ_ONLY,
                capabilities={ToolCapability.ENTITY_DISCOVERY, ToolCapability.RELATIONSHIP_ANALYSIS},
                input_model=EntityOneHopInput,
                tags=["entity", "neighbors", "graph"],
                performance_hint="Fast for sparse graphs"
            )
        )
        
        self.register_tool(
            tool_id="Entity.RelNode",
            function=entity_relnode_extract_tool,
            metadata=ToolMetadata(
                tool_id="Entity.RelNode",
                name="Entity Relationship Node Extraction",
                description="Extract entities and their relationships from graph",
                category=ToolCategory.READ_ONLY,
                capabilities={ToolCapability.ENTITY_DISCOVERY, ToolCapability.RELATIONSHIP_ANALYSIS},
                input_model=EntityRelNodeInput,
                tags=["entity", "relationship", "extraction"]
            )
        )
        
        # Register relationship tools
        self.register_tool(
            tool_id="Relationship.OneHopNeighbors",
            function=relationship_one_hop_neighbors_tool,
            metadata=ToolMetadata(
                tool_id="Relationship.OneHopNeighbors",
                name="Relationship One-Hop Neighbors",
                description="Find relationships within one hop of entities",
                category=ToolCategory.READ_ONLY,
                capabilities={ToolCapability.RELATIONSHIP_ANALYSIS},
                input_model=RelationshipOneHopNeighborsInputs,
                tags=["relationship", "neighbors", "graph"]
            )
        )
        
        self.register_tool(
            tool_id="Relationship.VDB.Build",
            function=relationship_vdb_build_tool,
            metadata=ToolMetadata(
                tool_id="Relationship.VDB.Build",
                name="Relationship Vector Database Builder",
                description="Build vector database for relationships",
                category=ToolCategory.WRITE,
                capabilities={ToolCapability.RELATIONSHIP_ANALYSIS, ToolCapability.DATA_PREPARATION},
                input_model=RelationshipVDBBuildInputs,
                tags=["relationship", "build", "index"]
            )
        )
        
        self.register_tool(
            tool_id="Relationship.VDB.Search",
            function=relationship_vdb_search_tool,
            metadata=ToolMetadata(
                tool_id="Relationship.VDB.Search",
                name="Relationship Vector Database Search",
                description="Search for similar relationships using embeddings",
                category=ToolCategory.READ_ONLY,
                capabilities={ToolCapability.RELATIONSHIP_ANALYSIS, ToolCapability.VECTOR_SEARCH},
                input_model=RelationshipVDBSearchInputs,
                tags=["relationship", "search", "embeddings"]
            )
        )
        
        # Register chunk tools
        self.register_tool(
            tool_id="Chunk.FromRelationships",
            function=chunk_from_relationships_tool,
            metadata=ToolMetadata(
                tool_id="Chunk.FromRelationships",
                name="Chunk Retrieval from Relationships",
                description="Retrieve text chunks associated with relationships",
                category=ToolCategory.READ_ONLY,
                capabilities={ToolCapability.TEXT_RETRIEVAL},
                input_model=ChunkFromRelationshipsInputs,
                tags=["chunk", "text", "retrieval"]
            )
        )
        
        self.register_tool(
            tool_id="Chunk.GetTextForEntities",
            function=chunk_get_text_for_entities_tool,
            metadata=ToolMetadata(
                tool_id="Chunk.GetTextForEntities",
                name="Chunk Text for Entities",
                description="Get text chunks containing specified entities",
                category=ToolCategory.READ_ONLY,
                capabilities={ToolCapability.TEXT_RETRIEVAL},
                input_model=ChunkGetTextForEntitiesInput,
                tags=["chunk", "text", "entity"]
            )
        )
        
        # Register graph construction tools
        graph_build_tools = [
            ("graph.BuildERGraph", build_er_graph, BuildERGraphInputs, "Entity-Relationship Graph"),
            ("graph.BuildRKGraph", build_rk_graph, BuildRKGraphInputs, "Relation-Knowledge Graph"),
            ("graph.BuildTreeGraph", build_tree_graph, BuildTreeGraphInputs, "Hierarchical Tree Graph"),
            ("graph.BuildTreeGraphBalanced", build_tree_graph_balanced, BuildTreeGraphBalancedInputs, "Balanced Tree Graph"),
            ("graph.BuildPassageGraph", build_passage_graph, BuildPassageGraphInputs, "Passage-based Graph")
        ]
        
        for tool_id, func, input_model, desc in graph_build_tools:
            self.register_tool(
                tool_id=tool_id,
                function=func,
                metadata=ToolMetadata(
                    tool_id=tool_id,
                    name=f"Build {desc}",
                    description=f"Construct a {desc} from corpus",
                    category=ToolCategory.BUILD,
                    capabilities={ToolCapability.GRAPH_CONSTRUCTION},
                    input_model=input_model,
                    tags=["graph", "construction", "build"],
                    performance_hint="Heavy operation, may take minutes for large corpora"
                )
            )
        
        # Register corpus tools
        self.register_tool(
            tool_id="corpus.PrepareFromDirectory",
            function=prepare_corpus_from_directory,
            metadata=ToolMetadata(
                tool_id="corpus.PrepareFromDirectory",
                name="Prepare Corpus from Directory",
                description="Process text files in directory into corpus format",
                category=ToolCategory.WRITE,
                capabilities={ToolCapability.DATA_PREPARATION},
                input_model=PrepareCorpusInputs,
                tags=["corpus", "preparation", "text"],
                dependencies=[]
            )
        )
        
        # Register visualization and analysis tools
        self.register_tool(
            tool_id="graph.Visualize",
            function=visualize_graph,
            metadata=ToolMetadata(
                tool_id="graph.Visualize",
                name="Graph Visualizer",
                description="Generate visual representation of graph",
                category=ToolCategory.READ_ONLY,
                capabilities={ToolCapability.VISUALIZATION},
                input_model=GraphVisualizerInput,
                tags=["graph", "visualization", "display"]
            )
        )
        
        self.register_tool(
            tool_id="graph.Analyze",
            function=analyze_graph,
            metadata=ToolMetadata(
                tool_id="graph.Analyze",
                name="Graph Analyzer",
                description="Analyze graph structure and statistics",
                category=ToolCategory.ANALYZE,
                capabilities={ToolCapability.ANALYSIS},
                input_model=GraphAnalyzerInput,
                tags=["graph", "analysis", "statistics"]
            )
        )
        
        # Register social media analysis tools if available
        try:
            self.register_tool(
                tool_id="social.IngestCOVIDDataset",
                function=ingest_covid_conspiracy_dataset,
                metadata=ToolMetadata(
                    tool_id="social.IngestCOVIDDataset",
                    name="COVID Conspiracy Dataset Ingestion",
                    description="Ingest COVID-19 conspiracy theory tweets from Hugging Face",
                    category=ToolCategory.WRITE,
                    capabilities={ToolCapability.DATA_PREPARATION},
                    input_model=DatasetIngestionInput,
                    tags=["social", "dataset", "covid", "conspiracy", "twitter"],
                    performance_hint="Downloads dataset from internet"
                )
            )
            
            self.register_tool(
                tool_id="social.AutoInterrogativePlanner",
                function=generate_interrogative_analysis_plans,
                metadata=ToolMetadata(
                    tool_id="social.AutoInterrogativePlanner",
                    name="Automated Interrogative Analysis Planner",
                    description="Generate diverse analysis scenarios with interrogative views",
                    category=ToolCategory.ANALYZE,
                    capabilities={ToolCapability.ANALYSIS},
                    input_model=AutoInterrogativePlanInput,
                    tags=["social", "planning", "analysis", "interrogative"],
                    performance_hint="LLM-powered planning"
                )
            )
        except Exception as e:
            logger.warning(f"Could not register social media tools: {e}")
        
        logger.info(f"DynamicToolRegistry: Initialized with {len(self._tools)} default tools")
    
    def register_tool(
        self,
        tool_id: str,
        function: Callable,
        metadata: ToolMetadata,
        validator: Optional[Callable] = None,
        pre_processor: Optional[Callable] = None,
        post_processor: Optional[Callable] = None
    ) -> None:
        """Register a new tool or update existing one"""
        registration = ToolRegistration(
            metadata=metadata,
            function=function,
            validator=validator,
            pre_processor=pre_processor,
            post_processor=post_processor
        )
        
        # Update registry
        self._tools[tool_id] = registration
        
        # Update indices
        for capability in metadata.capabilities:
            self._capability_index[capability].add(tool_id)
        self._category_index[metadata.category].add(tool_id)
        
        logger.info(f"Registered tool: {tool_id} ({metadata.category.value}, {len(metadata.capabilities)} capabilities)")
    
    def get_tool(self, tool_id: str) -> Optional[ToolRegistration]:
        """Get tool by ID"""
        return self._tools.get(tool_id)
    
    def get_tool_function(self, tool_id: str) -> Optional[Tuple[Callable, Type[BaseModel]]]:
        """Get tool function and input model (backward compatible)"""
        tool = self.get_tool(tool_id)
        if tool:
            return (tool.function, tool.metadata.input_model)
        return None
    
    def discover_tools(
        self,
        capabilities: Optional[Set[ToolCapability]] = None,
        category: Optional[ToolCategory] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """Discover tools based on criteria"""
        tool_ids = set(self._tools.keys())
        
        # Filter by capabilities
        if capabilities:
            capability_tools = set()
            for cap in capabilities:
                capability_tools.update(self._capability_index.get(cap, set()))
            tool_ids &= capability_tools
        
        # Filter by category
        if category:
            tool_ids &= self._category_index.get(category, set())
        
        # Filter by tags
        if tags:
            tool_ids = {
                tid for tid in tool_ids
                if any(tag in self._tools[tid].metadata.tags for tag in tags)
            }
        
        return sorted(list(tool_ids))
    
    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """Get all tools in a category"""
        return sorted(list(self._category_index.get(category, set())))
    
    def get_parallelizable_tools(self, tool_ids: List[str]) -> Tuple[List[str], List[str]]:
        """Separate tools into parallelizable and sequential groups"""
        parallel = []
        sequential = []
        
        for tool_id in tool_ids:
            tool = self.get_tool(tool_id)
            if tool and tool.metadata.is_parallelizable:
                parallel.append(tool_id)
            else:
                sequential.append(tool_id)
        
        return parallel, sequential
    
    def validate_tool_chain(self, tool_ids: List[str]) -> List[str]:
        """Validate tool execution order and return issues"""
        issues = []
        
        # Check dependencies
        for i, tool_id in enumerate(tool_ids):
            tool = self.get_tool(tool_id)
            if not tool:
                issues.append(f"Tool {tool_id} not found")
                continue
                
            # Check if dependencies are satisfied
            for dep in tool.metadata.dependencies:
                if dep not in tool_ids[:i]:
                    issues.append(f"Tool {tool_id} depends on {dep} which hasn't been executed")
        
        return issues
    
    def get_tool_metadata(self, tool_id: str) -> Optional[ToolMetadata]:
        """Get tool metadata"""
        tool = self.get_tool(tool_id)
        return tool.metadata if tool else None
    
    def list_all_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with metadata"""
        tools = []
        for tool_id, registration in self._tools.items():
            tools.append({
                "tool_id": tool_id,
                "name": registration.metadata.name,
                "description": registration.metadata.description,
                "category": registration.metadata.category.value,
                "capabilities": [cap.value for cap in registration.metadata.capabilities],
                "tags": registration.metadata.tags,
                "parallelizable": registration.metadata.is_parallelizable
            })
        return sorted(tools, key=lambda x: x["tool_id"])
    
    def __len__(self) -> int:
        """Number of registered tools"""
        return len(self._tools)
    
    def __contains__(self, tool_id: str) -> bool:
        """Check if tool is registered"""
        return tool_id in self._tools


# Global registry instance
global_tool_registry = DynamicToolRegistry()