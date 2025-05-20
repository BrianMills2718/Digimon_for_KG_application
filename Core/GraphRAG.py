from typing import Union, Any, Optional
from pyfiglet import Figlet
from Core.Chunk.DocChunk import DocChunk
from Core.Common.Logger import logger
import tiktoken
from pydantic import BaseModel, model_validator, ConfigDict # Removed Field from here as we're not using it for the problematic attrs
from Core.Common.ContextMixin import ContextMixin
from Core.Schema.RetrieverContext import RetrieverContext
from Core.Common.TimeStatistic import TimeStatistic
from Core.Graph import get_graph
from Core.Index import get_index, get_index_config
from Core.Query import get_query
from Core.Storage.NameSpace import Workspace
from Core.Community.ClusterFactory import get_community
from Core.Storage.PickleBlobStorage import PickleBlobStorage
from colorama import Fore, Style, init
import os

init(autoreset=True)


class GraphRAG(ContextMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Pydantic-managed fields (mostly from ContextMixin now)
    # Other complex objects will be initialized as instance attributes in initialize_components

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Non-Pydantic instance attributes are initialized in the 'initialize_components' validator
        # This avoids declaring them as Pydantic Fields if they cause conflicts.

    @model_validator(mode="before")
    @classmethod
    def welcome_message_validator(cls, values):
        f = Figlet(font='big')
        logo = f.renderText('DIGIMON')
        print(f"{Fore.GREEN}{'#' * 100}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{logo}{Style.RESET_ALL}")
        text = [
            "Welcome to DIGIMON: Deep Analysis of Graph-Based RAG Systems.",
            "",
            "Unlock advanced insights with our comprehensive tool for evaluating and optimizing RAG models.",
            "",
            "You can freely combine any graph-based RAG algorithms you desire. We hope this will be helpful to you!"
        ]
        def print_box(text_lines, border_color=Fore.BLUE, text_color=Fore.CYAN):
            max_length = max(len(line) for line in text_lines)
            border = f"{border_color}╔{'═' * (max_length + 2)}╗{Style.RESET_ALL}"
            print(border)
            for line in text_lines:
                print(
                    f"{border_color}║{Style.RESET_ALL} {text_color}{line.ljust(max_length)} {border_color}║{Style.RESET_ALL}")
            border = f"{border_color}╚{'═' * (max_length + 2)}╝{Style.RESET_ALL}"
            print(border)
        print_box(text)
        print(f"{Fore.GREEN}{'#' * 100}{Style.RESET_ALL}")
        return values

    @model_validator(mode="after")
    def initialize_components(self) -> 'GraphRAG':
        # Initialize standard Python attributes here.
        self.ENCODER: Any = tiktoken.encoding_for_model(self.config.token_model)
        self.workspace: Workspace = Workspace(self.config.working_dir, self.config.index_name)
        self.graph: Any = get_graph(self.config, llm=self.llm, encoder=self.ENCODER)
        self.doc_chunk: DocChunk = DocChunk(self.config.chunk, self.ENCODER, self.workspace.make_for("chunk_storage"))
        self.time_manager: TimeStatistic = TimeStatistic()
        self.retriever_context: RetrieverContext = RetrieverContext()
        
        self.entities_vdb_namespace: Optional[Any] = None
        self.relations_vdb_namespace: Optional[Any] = None
        self.subgraphs_vdb_namespace: Optional[Any] = None
        self.community_namespace: Optional[Any] = None
        self.e2r_namespace: Optional[Any] = None
        self.r2c_namespace: Optional[Any] = None

        self.entities_vdb: Optional[Any] = None
        self.relations_vdb: Optional[Any] = None
        self.subgraphs_vdb: Optional[Any] = None
        self.community: Optional[Any] = None
        self.entities_to_relationships: Optional[PickleBlobStorage] = None
        self.relationships_to_chunks: Optional[PickleBlobStorage] = None
        
        self.retriever_context_internal_config: dict = {}
        self.querier_internal: Optional[Any] = None
        self.artifacts_loaded_internal: bool = False
        
        self._init_storage_namespace()
        self._register_vdbs()
        self._register_community()
        self._register_e2r_r2c_matrix()
        self._update_retriever_context_config_internal()
        return self

    def _init_storage_namespace(self):
        self.graph.namespace = self.workspace.make_for("graph_storage")
        if self.config.use_entities_vdb:
            self.entities_vdb_namespace = self.workspace.make_for("entities_vdb")
        if self.config.use_relations_vdb:
            self.relations_vdb_namespace = self.workspace.make_for("relations_vdb")
        if self.config.use_subgraphs_vdb:
            self.subgraphs_vdb_namespace = self.workspace.make_for("subgraphs_vdb")
        if self.config.graph.use_community:
            self.community_namespace = self.workspace.make_for("community_storage")
        if self.config.use_entity_link_chunk and self.config.graph.graph_type != "tree_graph":
            self.e2r_namespace = self.workspace.make_for("map_e2r")
            self.r2c_namespace = self.workspace.make_for("map_r2c")

    def _register_vdbs(self):
        if self.config.use_entities_vdb:
            self.entities_vdb = get_index(
                get_index_config(self.config, persist_path=self.entities_vdb_namespace.get_save_path()))
        if self.config.use_relations_vdb:
            self.relations_vdb = get_index(
                get_index_config(self.config, persist_path=self.relations_vdb_namespace.get_save_path()))
        if self.config.use_subgraphs_vdb:
            self.subgraphs_vdb = get_index(
                get_index_config(self.config, persist_path=self.subgraphs_vdb_namespace.get_save_path()))

    def _register_community(self):
        if self.config.graph.use_community:
            self.community = get_community(self.config.graph.graph_cluster_algorithm,
                                           enforce_sub_communities=self.config.graph.enforce_sub_communities, 
                                           llm=self.llm, namespace=self.community_namespace)

    def _register_e2r_r2c_matrix(self):
        if self.config.graph.graph_type == "tree_graph":
            logger.warning("Tree graph is not supported for entity-link-chunk mapping. Skipping entity-link-chunk mapping.")
            if hasattr(self.config, "use_entity_link_chunk"):
                 self.config.use_entity_link_chunk = False
            return
        if self.config.use_entity_link_chunk:
            # Initialize PickleBlobStorage instances and assign them
            self.entities_to_relationships = PickleBlobStorage(namespace=self.e2r_namespace)
            self.relationships_to_chunks = PickleBlobStorage(namespace=self.r2c_namespace)


    def _update_retriever_context_config_internal(self):
        self.retriever_context_internal_config = {
            "config": True, "graph": True, "doc_chunk": True, "llm": True,
            "entities_vdb": self.config.use_entities_vdb,
            "relations_vdb": self.config.use_relations_vdb,
            "subgraphs_vdb": self.config.use_subgraphs_vdb,
            "community": self.config.graph.use_community,
            "relationships_to_chunks": self.config.use_entity_link_chunk and self.config.graph.graph_type != "tree_graph",
            "entities_to_relationships": self.config.use_entity_link_chunk and self.config.graph.graph_type != "tree_graph",
            "query_config": True,
        }

    async def _build_retriever_context(self):
        logger.info("Building retriever context for the current execution")
        try:
            for context_name, use_context in self.retriever_context_internal_config.items():
                if use_context:
                    config_value = None
                    if context_name == "config":
                        config_value = self.config.retriever
                    elif context_name == "query_config":
                        config_value = self.config.query
                    elif hasattr(self, context_name):
                        config_value = getattr(self, context_name)
                    if config_value is not None:
                        self.retriever_context.register_context(context_name, config_value)
                    else:
                        logger.warning(f"Retriever context component '{context_name}' configured to be used but not found on GraphRAG instance.")
            
            if self.retriever_context: 
                 self.querier_internal = get_query(self.config.retriever.query_type, self.config.query, self.retriever_context)
            else:
                logger.error("Retriever context is empty. Querier cannot be initialized.")
                self.querier_internal = None

        except Exception as e:
            logger.error(f"Failed to build retriever context: {e}", exc_info=True)
            self.querier_internal = None
            raise

    async def build_e2r_r2c_maps(self, force=False):
        if not self.config.use_entity_link_chunk or self.config.graph.graph_type == "tree_graph":
            logger.info("Skipping E2R/R2C map building as it's not configured or not applicable for tree graph.")
            return
        
        # Ensure these attributes are initialized before use
        if self.entities_to_relationships is None or self.relationships_to_chunks is None:
            logger.error("E2R/R2C maps are not initialized. Call _register_e2r_r2c_matrix first.")
            self._register_e2r_r2c_matrix() # Attempt to initialize them

        logger.info("Starting build two maps: 1️⃣ entity <-> relationship; 2️⃣ relationship <-> chunks ")
        loaded_e2r = await self.entities_to_relationships.load(force) # type: ignore
        if not loaded_e2r:
            await self.entities_to_relationships.set(await self.graph.get_entities_to_relationships_map(False)) # type: ignore
            await self.entities_to_relationships.persist() # type: ignore
        
        loaded_r2c = await self.relationships_to_chunks.load(force) # type: ignore
        if not loaded_r2c:
            await self.relationships_to_chunks.set(await self.graph.get_relationships_to_chunks_map(self.doc_chunk)) # type: ignore
            await self.relationships_to_chunks.persist() # type: ignore
        logger.info("✅ Finished building the two maps ")

    def _update_costs_info(self, stage_str: str):
        if self.llm and hasattr(self.llm, 'get_last_stage_cost'): 
            last_cost = self.llm.get_last_stage_cost() # type: ignore
            logger.info(f"{stage_str} stage cost: Total prompt token: {last_cost.total_prompt_tokens}, Total completion token: {last_cost.total_completion_tokens}, Total cost: {last_cost.total_cost}")
        else:
            logger.warning(f"LLM or cost tracking not fully initialized for '{stage_str}' stage.")
        last_stage_time = self.time_manager.stop_last_stage() # type: ignore
        logger.info(f"{stage_str} time(s): {last_stage_time:.2f}")

    async def build_and_persist_artifacts(self, docs: Union[str, list[Any]]):
        logger.info(f"--- Starting Artifact Build Process for {self.config.exp_name} ---")
        self.time_manager.start_stage() # type: ignore
        await self.doc_chunk.build_chunks(docs, force=self.config.graph.force) # type: ignore
        self._update_costs_info("Chunking")
        
        self.time_manager.start_stage() # type: ignore
        await self.graph.build_graph(await self.doc_chunk.get_chunks(), self.config.graph.force) # type: ignore
        self._update_costs_info("Build Graph")
        
        self.time_manager.start_stage() # type: ignore
        if self.config.use_entities_vdb:
            node_metadata = await self.graph.node_metadata() # type: ignore
            if not node_metadata:
                logger.warning("No node metadata found. Skipping entity indexing.")
            else:
                logger.info("Forcing rebuild of entities VDB for testing metadata propagation.")
                await self.entities_vdb.build_index(await self.graph.nodes_data(), node_metadata, True) # force=True for testing

        if self.config.enable_graph_augmentation: 
            await self.graph.augment_graph_by_similarity_search(self.entities_vdb) # type: ignore

        if self.config.use_entity_link_chunk:
            await self.build_e2r_r2c_maps(force=self.config.graph.force)

        if self.config.use_relations_vdb:
            edge_metadata = await self.graph.edge_metadata() # type: ignore
            if not edge_metadata:
                logger.warning("No edge metadata found. Skipping relation indexing.")
            else:
                await self.relations_vdb.build_index(await self.graph.edges_data(), edge_metadata, force=self.config.graph.force) # type: ignore

        if self.config.use_subgraphs_vdb:
            subgraph_metadata = await self.graph.subgraph_metadata() # type: ignore
            if not subgraph_metadata:
                logger.warning("No subgraph metadata found. Skipping subgraph indexing.")
            else:
                await self.subgraphs_vdb.build_index(await self.graph.subgraphs_data(), subgraph_metadata, force=self.config.graph.force) # type: ignore

        if self.config.graph.use_community:
            await self.community.cluster(largest_cc=await self.graph.stable_largest_cc(), # type: ignore
                                         max_cluster_size=self.config.graph.max_graph_cluster_size,
                                         random_seed=self.config.graph.graph_cluster_seed, force=self.config.graph.force)
            await self.community.generate_community_report(self.graph, force=self.config.graph.force) # type: ignore
        self._update_costs_info("Index Building & Community")
        
        await self._build_retriever_context()
        logger.info(f"--- Artifact Build Process for {self.config.exp_name} Completed ---")

    async def setup_for_querying(self):
        if self.artifacts_loaded_internal:
            logger.info("Artifacts already loaded for querying.")
            return True

        logger.info(f"--- Starting Artifact Loading Process for {self.config.exp_name} ---")
        
        if not await self.doc_chunk._load_chunk(force=False): # type: ignore
            logger.error("Failed to load chunk data. Ensure 'build' mode was run successfully.")
            return False
        logger.info("Chunks loaded successfully.")

        if not await self.graph.load_persisted_graph(force=False): # type: ignore
            logger.error("Failed to load graph data. Ensure 'build' mode was run successfully.")
            return False
        logger.info("Graph loaded successfully.")

        if self.config.use_entities_vdb:
            if not await self.entities_vdb._load_index(): # type: ignore
                logger.error("Failed to load entities VDB.")
                return False
            logger.info("Entities VDB loaded. Index object: " + str(self.entities_vdb._index)) # type: ignore


        if self.config.use_relations_vdb:
            if not await self.relations_vdb._load_index(): # type: ignore
                logger.error("Failed to load relations VDB.")
                return False
            logger.info("Relations VDB loaded successfully.")

        if self.config.use_subgraphs_vdb:
            if not await self.subgraphs_vdb._load_index(): # type: ignore
                logger.error("Failed to load subgraphs VDB.")
                return False
            logger.info("Subgraphs VDB loaded successfully.")

        if self.config.graph.use_community:
            if not await self.community._load_cluster_map(force=False): # type: ignore
                logger.error("Failed to load community node map.")
            else:
                logger.info("Community node map loaded successfully.")
            
            if not await self.community._load_community_report(self.graph, force=False): # type: ignore
                logger.error("Failed to load community reports.")
            else:
                logger.info("Community reports loaded successfully.")

        if self.config.use_entity_link_chunk and self.config.graph.graph_type != "tree_graph":
            # Ensure these are initialized if not already
            if self.entities_to_relationships is None or self.relationships_to_chunks is None:
                self._register_e2r_r2c_matrix()

            if not await self.entities_to_relationships.load(force=False): # type: ignore
                logger.error("Failed to load entities_to_relationships map.")
            else:
                logger.info("Entities_to_relationships map loaded successfully.")
            
            if not await self.relationships_to_chunks.load(force=False): # type: ignore
                logger.error("Failed to load relationships_to_chunks map.")
            else:
                logger.info("Relationships_to_chunks map loaded successfully.")
        
        try:
            await self._build_retriever_context()
            if self.querier_internal is None: 
                 logger.error("Querier failed to initialize after loading artifacts.")
                 return False
        except Exception as e:
            logger.error(f"Error building retriever context after loading artifacts: {e}", exc_info=True)
            return False
        
        self.artifacts_loaded_internal = True
        logger.info(f"--- Artifact Loading Process for {self.config.exp_name} Completed ---")
        return True

    async def query(self, query_text: str):
        if not self.artifacts_loaded_internal:
            logger.info("Artifacts not loaded for querying, attempting to load now...")
            if not await self.setup_for_querying():
                return "Error: Failed to load necessary artifacts for querying. Please run 'build' mode first."
        
        if not self.querier_internal:
            logger.error("Query engine (querier_internal) is not initialized. Cannot proceed with query.")
            return "Error: Query engine not available."
            
        logger.info(f"Processing query: '{query_text}'")
        response = await self.querier_internal.query(query_text) 
        return response
