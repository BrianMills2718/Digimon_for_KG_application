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
import json
from pathlib import Path

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

    async def build_and_persist_artifacts(self, docs_path: Union[str, list[Any]]):
        logger.info(f"--- Starting Artifact Build Process for {self.config.exp_name} ---")
        self.time_manager.start_stage() # type: ignore
        await self.doc_chunk.build_chunks(docs_path, force=self.config.graph.force) # type: ignore
        
        # Check if any chunks were generated
        graph_chunks = await self.doc_chunk.get_chunks()
        if not graph_chunks:
            logger.error(f"No chunks were generated from document path: {docs_path}. Halting build process.")
            # Return an error dictionary that the API can understand
            return {"error": f"Failed to build artifacts: No processable documents or chunks found in {docs_path}."}
        logger.info(f"Successfully generated {len(graph_chunks)} chunks.")
        
        self._update_costs_info("Chunking")
        
        self.time_manager.start_stage() # type: ignore
        await self.graph.build_graph(graph_chunks, self.config.graph.force) # type: ignore
        self._update_costs_info("Build Graph")
        
        self.time_manager.start_stage() # type: ignore
        if self.config.use_entities_vdb:
            node_metadata = await self.graph.node_metadata() # type: ignore
            nodes_data_for_index = await self.graph.nodes_data()
            if not nodes_data_for_index:
                logger.warning("No nodes data found to build entities VDB. Skipping entity indexing.")
            elif not node_metadata:
                logger.warning("No node metadata found. Skipping entity indexing.")
            else:
                logger.info(f"Found {len(nodes_data_for_index)} nodes to index for entities VDB.")
                logger.info("Forcing rebuild of entities VDB for testing metadata propagation.")
                await self.entities_vdb.build_index(nodes_data_for_index, node_metadata, True) # force=True for testing

        if self.config.enable_graph_augmentation: 
            await self.graph.augment_graph_by_similarity_search(self.entities_vdb) # type: ignore

        if self.config.use_entity_link_chunk:
            await self.build_e2r_r2c_maps(force=self.config.graph.force)

        if self.config.use_relations_vdb:
            edge_metadata = await self.graph.edge_metadata() # type: ignore
            edges_data_for_index = await self.graph.edges_data()
            if not edges_data_for_index:
                logger.warning("No edges data found to build relations VDB. Skipping relation indexing.")
            elif not edge_metadata:
                logger.warning("No edge metadata found. Skipping relation indexing.")
            else:
                logger.info(f"Found {len(edges_data_for_index)} edges to index for relations VDB.")
                await self.relations_vdb.build_index(edges_data_for_index, edge_metadata, force=self.config.graph.force) # type: ignore

        if self.config.use_subgraphs_vdb:
            subgraph_metadata = await self.graph.subgraph_metadata() # type: ignore
            subgraphs_data_for_index = await self.graph.subgraphs_data()
            if not subgraphs_data_for_index:
                logger.warning("No subgraphs data found to build subgraphs VDB. Skipping subgraph indexing.")
            elif not subgraph_metadata:
                logger.warning("No subgraph metadata found. Skipping subgraph indexing.")
            else:
                logger.info(f"Found {len(subgraphs_data_for_index)} subgraphs to index for subgraphs VDB.")
                await self.subgraphs_vdb.build_index(subgraphs_data_for_index, subgraph_metadata, force=self.config.graph.force) # type: ignore

        if self.config.graph.use_community:
            largest_cc_data = await self.graph.stable_largest_cc()
            if not largest_cc_data or (hasattr(largest_cc_data, 'number_of_nodes') and largest_cc_data.number_of_nodes() == 0):
                logger.warning("Largest connected component is empty or invalid. Skipping community detection.")
            else:
                logger.info("Proceeding with community detection.")
                await self.community.cluster(largest_cc=largest_cc_data, 
                                           max_cluster_size=self.config.graph.max_graph_cluster_size,
                                           random_seed=self.config.graph.graph_cluster_seed, force=self.config.graph.force)
                await self.community.generate_community_report(self.graph, force=self.config.graph.force) # type: ignore
        self._update_costs_info("Index Building & Community")
        
        await self._build_retriever_context()
        logger.info(f"--- Artifact Build Process for {self.config.exp_name} Completed ---")
        return {"message": f"Artifacts built successfully for {self.config.exp_name} using data from {str(docs_path)}."}

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

    async def evaluate_model(self):
        """
        Runs the evaluation process for the configured dataset and method.
        Loads questions, runs queries, (optionally scores them), saves results, and returns a summary.
        """
        logger.info(f"--- Starting Evaluation Process for {self.config.exp_name} on dataset {self.config.dataset_name} ---")

        if not self.artifacts_loaded_internal:
            logger.info("Artifacts not loaded for evaluation, attempting to load now...")
            if not await self.setup_for_querying():
                error_message = "Error: Failed to load necessary artifacts for evaluation. Please run 'build' mode first."
                logger.error(error_message)
                return {"error": error_message, "metrics": {}}

        if not self.querier_internal:
            error_message = "Error: Query engine (querier_internal) is not initialized. Cannot proceed with evaluation."
            logger.error(error_message)
            return {"error": error_message, "metrics": {}}

        # Determine the path to the Question.json file
        # Assumes 'Data' directory is at the same level as 'Core', 'Option', etc.,
        # and api.py (which calls this) is run from the project root '~/digimon/'
        project_root = Path(os.getcwd()) 
        eval_questions_path = project_root / "Data" / self.config.dataset_name / "Question.json"

        if not eval_questions_path.exists():
            message = f"Evaluation questions file not found at {eval_questions_path}. Looked for 'Data/{self.config.dataset_name}/Question.json' relative to project root."
            logger.error(message)
            return {"error": message, "metrics": {}}

        evaluation_questions_data = []
        try:
            with open(eval_questions_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        question_obj = json.loads(line)
                        if not isinstance(question_obj, dict):
                            logger.warning(f"Line {line_number} in {eval_questions_path} is not a valid JSON object (dictionary). Skipping: {line}")
                            continue
                        evaluation_questions_data.append(question_obj)
                    except json.JSONDecodeError as e_line:
                        logger.warning(f"Skipping invalid JSON line {line_number} in {eval_questions_path}: {line}. Error: {e_line}")

            if not evaluation_questions_data:
                raise ValueError(f"No valid JSON objects found in {eval_questions_path} or the file is empty.")
            logger.info(f"Successfully loaded {len(evaluation_questions_data)} questions from {eval_questions_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {eval_questions_path}: {e}", exc_info=True)
            return {"error": f"Invalid JSON in evaluation questions file: {str(e)}", "metrics": {}}
        except Exception as e:
            logger.error(f"Error loading or parsing evaluation questions from {eval_questions_path}: {e}", exc_info=True)
            return {"error": f"Failed to load evaluation questions: {str(e)}", "metrics": {}}

        if not evaluation_questions_data:
            return {"message": "No evaluation questions loaded or file was empty.", "metrics": {}}

        generated_results = []
        metrics_summary = {
            "total_questions_in_file": len(evaluation_questions_data),
            "questions_processed": 0,
            "questions_failed_to_answer": 0,
            # TODO: Add specific metrics like accuracy, precision, recall, F1, semantic similarity scores etc.
        }

        # Refined evaluation output directory structure
        output_dir = Path("results") / self.config.dataset_name / "test" / "Evaluation_Outputs" / self.config.exp_name
        logger.info(f"Constructed evaluation output directory: {str(output_dir)}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for item in evaluation_questions_data:
            q_id = item.get("id", f"q_{metrics_summary['questions_processed']}")
            question_text = item.get("question")
            ground_truth_answer = item.get("expected_answer") or item.get("ground_truth_answer") 

            if not question_text:
                logger.warning(f"Skipping question with id '{q_id}' due to missing question text.")
                continue

            logger.info(f"Evaluating Q_ID: {q_id} - Question: {question_text[:60]}...")
            try:
                raw_answer = await self.query(question_text)

                generated_answer_text = ""
                if hasattr(raw_answer, 'answer') and isinstance(raw_answer.answer, str):
                    generated_answer_text = raw_answer.answer
                elif isinstance(raw_answer, str):
                    generated_answer_text = raw_answer
                else: 
                    generated_answer_text = str(raw_answer) 
                    logger.warning(f"Query for Q_ID {q_id} returned an unexpected type: {type(raw_answer)}. Converted to string.")

                generated_results.append({
                    "id": q_id,
                    "question": question_text,
                    "generated_answer": generated_answer_text,
                    "ground_truth": ground_truth_answer,
                    # TODO: Add your actual scoring logic here
                })
                metrics_summary["questions_processed"] += 1
            except Exception as e:
                logger.error(f"Error querying for Q_ID {q_id}: {e}", exc_info=True)
                metrics_summary["questions_failed_to_answer"] += 1
                generated_results.append({
                    "id": q_id,
                    "question": question_text,
                    "generated_answer": f"ERROR: {str(e)}",
                    "ground_truth": ground_truth_answer,
                })

        eval_results_filename = f"{self.config.dataset_name}_query_outputs_for_eval.json"
        eval_results_file_path = output_dir / eval_results_filename
        try:
            with open(eval_results_file_path, 'w', encoding='utf-8') as f:
                json.dump(generated_results, f, indent=2)
            logger.info(f"Detailed evaluation results saved to: {eval_results_file_path}")
        except Exception as e:
            logger.error(f"Failed to save detailed evaluation results: {e}", exc_info=True)

        metrics_filename = f"{self.config.dataset_name}_evaluation_metrics.json"
        metrics_file_path = output_dir / metrics_filename
        try:
            with open(metrics_file_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_summary, f, indent=2)
            logger.info(f"Evaluation metrics summary saved to: {metrics_file_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation metrics summary: {e}", exc_info=True)

        logger.info(f"--- Evaluation Process for {self.config.exp_name} Completed ---")
        return {
            "message": "Evaluation completed. Results and metrics saved to files.",
            "metrics": metrics_summary,
            "results_file_path": str(eval_results_file_path),
            "metrics_file_path": str(metrics_file_path)
        }
