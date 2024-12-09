import asyncio
from collections import defaultdict, Counter
from typing import Union, Any
from Core.Index.TFIDFStore import TFIDFIndex
from Core.Prompt.QueryPrompt import KGP_QUERY_PROMPT

from lazy_object_proxy.utils import await_
from scipy.sparse import csr_matrix, csr_array
from pyfiglet import Figlet
from Core.Chunk.DocChunk import DocChunk
import numpy as np
from Core.Common.Constants import GRAPH_FIELD_SEP, ANSI_COLOR
from Core.Common.Logger import logger
import tiktoken
from Core.Common.Utils import (mdhash_id, prase_json_from_response, clean_str, truncate_list_by_token_size, \
                               split_string_by_multi_markers, min_max_normalize,    list_to_quoted_csv_string,)
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from Core.Common.ContextMixin import ContextMixin
from Core.Common.TimeStatistic import TimeStatistic
from Core.Graph.GraphFactory import get_graph
from Core.Index import get_index
from Core.Index.IndexConfigFactory import get_index_config
from Core.Prompt import GraphPrompt, QueryPrompt
from Core.Storage.NameSpace import Workspace
from Core.Community.ClusterFactory import get_community
from Core.Storage.PickleBlobStorage import PickleBlobStorage
from colorama import Fore, Style, init
import json
from Core.Retriever.MixRetriever import MixRetriever

init(autoreset=True)  # Initialize colorama and reset color after each print


class GraphRAG(ContextMixin, BaseModel):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    working_dir: str = Field(default="", exclude=True)
    # The following two matrices are utilized for mapping entities to their corresponding chunks through the specified link-path:
    # Entity Matrix: Represents the entities in the dataset.
    # Chunk Matrix: Represents the chunks associated with the entities.
    # These matrices facilitate the entity -> relationship -> chunk linkage, which is integral to the HippoRAG and FastGraphRAG models.

    @field_validator("working_dir", mode="before")
    @classmethod
    def check_working_dir(cls, value: str):
        if value == "":
            logger.error("Working directory cannot be empty")
        return value

    @model_validator(mode="before")
    def welcome_message(cls, values):
        f = Figlet(font='big')  #
        # Generate the large ASCII art text
        logo = f.renderText('DIGIMON')
        print(f"{Fore.GREEN}{'#' * 100}{Style.RESET_ALL}")
        # Print the logo with color
        print(f"{Fore.MAGENTA}{logo}{Style.RESET_ALL}")
        text = [
            "Welcome to DIGIMON: Deep Analysis of Graph-Based RAG Systems.",
            "",
            "Unlock advanced insights with our comprehensive tool for evaluating and optimizing RAG models.",
            "",
            "You can freely combine any graph-based RAG algorithms you desire. We hope this will be helpful to you!"
        ]

        # Function to print a boxed message
        def print_box(text_lines, border_color=Fore.BLUE, text_color=Fore.CYAN):
            max_length = max(len(line) for line in text_lines)
            border = f"{border_color}╔{'═' * (max_length + 2)}╗{Style.RESET_ALL}"
            print(border)
            for line in text_lines:
                print(
                    f"{border_color}║{Style.RESET_ALL} {text_color}{line.ljust(max_length)} {border_color}║{Style.RESET_ALL}")
            border = f"{border_color}╚{'═' * (max_length + 2)}╝{Style.RESET_ALL}"
            print(border)

        # Print the boxed welcome message
        print_box(text)

        # Add a decorative line for separation
        print(f"{Fore.GREEN}{'#' * 100}{Style.RESET_ALL}")
        return values

    @model_validator(mode="after")
    def _update_context(cls, data):
        cls.config = data.context.config
        cls.ENCODER = tiktoken.encoding_for_model(cls.config.token_model)
        cls.workspace = Workspace(data.working_dir, cls.config.exp_name)  # register workspace
        cls.graph = get_graph(data.config, llm=data.llm, encoder=cls.ENCODER)  # register graph
        cls.doc_chunk = DocChunk(data.config.chunk_method, cls.ENCODER, data.workspace.make_for("chunk_storage"))
        cls.time_manager = TimeStatistic()
        cls.retriever = MixRetriever()
        data = cls._init_storage_namespace(data)
        data = cls._register_vdbs(data)
        data = cls._register_community(data)
        data = cls._register_e2r_r2c_matrix(data)
        data = cls._register_retriever_context(data)
        return data

    @classmethod
    def _init_storage_namespace(cls, data):
        data.graph.namespace = data.workspace.make_for("graph_storage")
        if data.config.use_entities_vdb:
            data.entities_vdb_namespace = data.workspace.make_for("entities_vdb")
        if data.config.use_relations_vdb:
            data.relations_vdb_namespace = data.workspace.make_for("relations_vdb")
        if data.config.use_community:
            data.community_namespace = data.workspace.make_for("community_storage")
        if data.config.use_entity_link_chunk:
            data.e2r_namespace = data.workspace.make_for("map_e2r")
            data.r2c_namespace = data.workspace.make_for("map_r2c")

   
        return data

    @classmethod
    def _register_vdbs(cls, data):
        # If vector database is needed, register them into the class
        if data.config.use_entities_vdb:
            cls.entities_vdb = get_index(
                get_index_config(data.config, persist_path=data.entities_vdb_namespace.get_save_path()))
        if data.config.use_relations_vdb:
            cls.relations_vdb = get_index(
                get_index_config(data.config, persist_path=data.relations_vdb_namespace.get_save_path()))
        return data

    @classmethod
    def _register_community(cls, data):
        if data.config.use_community:
            cls.community = get_community(data.config.graph_cluster_algorithm,
                                          enforce_sub_communities=data.config.enforce_sub_communities, llm=data.llm,namespace = data.community_namespace
                                         )

        return data

    @classmethod
    def _register_e2r_r2c_matrix(cls, data):
        # The following two matrices are utilized for mapping entities to their corresponding chunks through the specified link-path:
        # Entity Matrix: Represents the entities in the dataset.
        # Chunk Matrix: Represents the chunks associated with the entities.
        # These matrices facilitate the entity -> relationship -> chunk linkage, which is integral to the HippoRAG and FastGraphRAG models.
        if data.config.use_entity_link_chunk:
            cls.entities_to_relationships = PickleBlobStorage(
                namespace=data.e2r_namespace, config=None
            )
            cls.relationships_to_chunks = PickleBlobStorage(
                namespace=data.r2c_namespace, config=None
            )
        return data

    @classmethod
    def _register_retriever_context(cls, data):
        """
        Register the retriever context based on the configuration provided in `data`.

        Args:
            data: An object containing the configuration.

        Returns:
            The input `data` object.
        """
        cls._retriever_context = {
            "graph": True,
            "doc_chunk": True,
            "entities_vdb": data.config.use_entities_vdb,
            "relations_vdb": data.config.use_relations_vdb,
            "community": data.config.use_community,
            "relationships_to_chunks": data.config.use_entity_link_chunk,
            "entities_to_relationships": data.config.use_entity_link_chunk,
        }
        return data

    async def _build_retriever_context(self):
        """
        Build the retriever context for subsequent retriever calls.

        This method registers the necessary contexts with the retriever based on the
        configuration set in `_retriever_context`.
        """
        logger.info("Building retriever context for the current execution")
        try:
            self.retriever.register_context("graph", self.graph)
            for context_name, use_context in self._retriever_context.items():
                if use_context:
                    self.retriever.register_context(context_name, getattr(self, context_name))
        except Exception as e:
            logger.error(f"Failed to build retriever context: {e}")
            raise


    async def build_e2r_r2c_maps(self, force = False):
        # await self._build_ppr_context()
        logger.info("Starting build two maps: 1️⃣ entity <-> relationship; 2️⃣ relationship <-> chunks ")
        if not await self._entities_to_relationships.load(force):
            await self._entities_to_relationships.set(await self.graph.get_entities_to_relationships_map(False))
            await self._entities_to_relationships.persist()
        if not await self._relationships_to_chunks.load(force):
            await self._relationships_to_chunks.set(await self.graph.get_relationships_to_chunks_map(self.doc_chunk))
            await self._relationships_to_chunks.persist()
        logger.info("✅ Finished building the two maps ")


    def _update_costs_info(self, stage_str:str):
        last_cost = self.llm.get_last_stage_cost()
        logger.info(f"{stage_str} stage cost: Total prompt token: {last_cost.total_prompt_tokens}, Total completeion token: {last_cost.total_completion_tokens}, Total cost: {last_cost.total_cost}")
        last_stage_time = self.time_manager.stop_last_stage()
        logger.info(f"{stage_str} time(s): {last_stage_time:.2f}")

        
    async def insert(self, docs: Union[str, list[Any]]):

        """
        The main function that orchestrates the first step in the Graph RAG pipeline.
        This function is responsible for executing the various stages of the Graph RAG process,
        including chunking, graph construction, index building, and graph augmentation (optional).

        Configuration of the Graph RAG method is based on the parameters provided in the config file.
        For detailed information on the configuration and usage, please refer to the README.md.

        Args:
            docs (Union[str, list[[Any]]): A list of documents to be processed and inserted into the Graph RAG pipeline.
        """

        ####################################################################################################
        # 1. Chunking Stage
        ####################################################################################################
        self.time_manager.start_stage()
        
        logger.info("Starting chunk the given documents")
        await self.doc_chunk.build_chunks(docs, True)
        logger.info("✅ Finished the chunking stage")

        self._update_costs_info("Chunking")
        
        ####################################################################################################
        # 2. Building Graph Stage
        ####################################################################################################
        logger.info("Starting build graph for the given documents")
        await self.graph.build_graph(await self.doc_chunk.get_chunks(), True)
        logger.info("✅ Finished the graph building stage")
        
        self._update_costs_info("Build Graph")
        
        import pdb
        pdb.set_trace()
        ####################################################################################################
        # 3. Index building Stage 
        # Data-driven content should be pre-built offline to ensure efficient online query performance.
        ####################################################################################################

        # NOTE: ** Ensure the graph is successfully loaded before proceeding to load the index from storage, as it represents a one-to-one mapping. **
        if self.config.use_entities_vdb:
            node_metadata = await self.graph.node_metadata()
            if not node_metadata:
                logger.warning("No node metadata found. Skipping entity indexing.")
            logger.info("Starting insert entities of the given graph into vector database")
            # await self.entities_vdb.build_index(await self.graph.nodes(), entity_metadata, force=False)
            await self.entities_vdb.build_index(await self.graph.nodes_data(), node_metadata, False)
            logger.info("✅ Finished starting insert entities of the given graph into vector database")

        # Graph Augmentation Stage  (Optional) 
        # For HippoRAG and MedicalRAG, similarities between entities are utilized to create additional edges.
        # These edges represent similarity types and are leveraged in subsequent processes.

        # if self.config.enable_graph_augmentation:
        #     logger.info("Starting augment the existing graph with similariy edges")

        #     await self.graph.augment_graph_by_similrity_search()
        #     logger.info("✅ Finished augment the existing graph with similariy edges")

        if self.config.use_entity_link_chunk:
            await self.build_e2r_r2c_maps(False)

        if self.config.use_relations_vdb:
            edge_metadata = await self.graph.edge_metadata()
            if not edge_metadata:
                logger.warning("No edge metadata found. Skipping relation indexing.")
                return
            logger.info("Starting insert relations of the given graph into vector database")
            await self.relations_vdb.build_index(await self.graph.edges_data(), edge_metadata, force=False)
            logger.info("✅ Finished starting insert relations of the given graph into vector database")

          
        if self.config.use_community:
            logger.info("Starting build community of the given graph")
            logger.start("Clustering nodes")
            await self.community.cluster(largest_cc=await self.graph.stable_largest_cc(),
                                         max_cluster_size=self.config.max_graph_cluster_size,
                                         random_seed=self.config.graph_cluster_seed)

            await self.community.generate_community_report(self.graph, False)
            logger.info("✅ [Community Report]  Finished")
         

         
    async def retrieve(self, question):
        
        self.k: int = 30
        self.k_nei: int = 3
        graph_nodes = list(await self.graph.get_nodes())
        # corpus = dict({id: (await self.graph.get_node(id))['chunk'] for id in list(self.gra.graph.nodes)})
        corpus = dict({id: (await self.graph.get_node(id))['description'] for id in graph_nodes})
        candidates_idx = list(id for id in graph_nodes)
        import pdb

        seed = question
        contexts = []
        
        idxs = await self.tf_idf(seed, candidates_idx, corpus, k = self.k // self.k_nei)

        cur_contexts = [corpus[_] for _ in idxs]
        next_reasons = [seed + '\n' + (await self.llm.aask(KGP_QUERY_PROMPT.format(question=question, context=context))) for context in cur_contexts]

        logger.info("next_reasons: {next_reasons}".format(next_reasons=next_reasons))

        visited = []

        for idx, next_reason in zip(idxs, next_reasons):
            nei_candidates_idx = list(await self.graph.get_neighbors(idx))
            import pdb
            nei_candidates_idx = [_ for _ in nei_candidates_idx if _ not in visited]
            if (nei_candidates_idx == []):
                continue

            next_contexts = await self.tf_idf(next_reason, nei_candidates_idx, corpus, k = self.k_nei)
            contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
            visited.append(idx)
            visited.extend([_ for _ in next_contexts])
        import pdb
     
        return contexts

    async def tf_idf(self, seed, candidates_idx, corpus, k):

        index = TFIDFIndex()

        index._build_index_from_list([corpus[_] for _ in candidates_idx])
        idxs = index.query(query_str = seed, top_k = k)

        return [candidates_idx[_] for _ in idxs]
    async def query(self, query):
        """
            Executes the query by extracting the relevant content, and then generating a response.
            Args:
                query: The query to be processed.
            Returns:
        """
        await self._build_retriever_context()
        import pdb
        pdb.set_trace()
        
        print("Sdsdsdsd")

        ####################################################################################################
        # 2. Generation Stage
        ####################################################################################################




   

    
    
   
      
        

   



   

  
    async def _map_global_communities(
            self,
            query: str,
            communities_data
        ):
            
            community_groups = []
            while len(communities_data):
                this_group = truncate_list_by_token_size(
                    communities_data,
                    key=lambda x: x["report_string"],
                    max_token_size=self.config.global_max_token_for_community_report,
                )
                community_groups.append(this_group)
                communities_data = communities_data[len(this_group) :]

            async def _process(community_truncated_datas: list[Any]) -> dict:
                communities_section_list = [["id", "content", "rating", "importance"]]
                for i, c in enumerate(community_truncated_datas):
                    communities_section_list.append(
                        [
                            i,
                            c["report_string"],
                            c["report_json"].get("rating", 0),
                            c['community_info']['occurrence'],
                        ]
                    )
                community_context = list_to_quoted_csv_string(communities_section_list)
                sys_prompt_temp = QueryPrompt.GLOBAL_MAP_RAG_POINTS
                sys_prompt = sys_prompt_temp.format(context_data=community_context)
       
                response = await self.llm.aask(
                    query,
                    system_msgs = [sys_prompt]
                )
      
       
                data = prase_json_from_response(response)
                return data.get("points", [])
     
            logger.info(f"Grouping to {len(community_groups)} groups for global search")
            responses = await asyncio.gather(*[_process(c) for c in community_groups])
  
            return responses

    async def global_query(
            self,
            query
        ) -> str:
            community_schema = self.community.community_schema
            community_schema = {
                k: v for k, v in community_schema.items() if v.level <= self.config.level
            }
            if not len(community_schema):
                return QueryPrompt.FAIL_RESPONSE

            sorted_community_schemas = sorted(
                community_schema.items(),
                key=lambda x: x[1].occurrence,
                reverse=True,
            )
     
            sorted_community_schemas = sorted_community_schemas[
                : self.config.global_max_consider_community
            ]
            community_datas = await self.community.community_reports.get_by_ids( ###
                [k[0] for k in sorted_community_schemas]
            )
      
            community_datas = [c for c in community_datas if c is not None]
            community_datas = [
                c
                for c in community_datas
                if c["report_json"].get("rating", 0) >= self.config.global_min_community_rating
            ]
            community_datas = sorted(
                community_datas,
                key=lambda x: (x['community_info']['occurrence'], x["report_json"].get("rating", 0)),
                reverse=True,
            )
            logger.info(f"Revtrieved {len(community_datas)} communities")
            map_communities_points = await self._map_global_communities(
                query, community_datas
            )
            final_support_points = []
            for i, mc in enumerate(map_communities_points):
                for point in mc:
                    if "description" not in point:
                        continue
                    final_support_points.append(
                        {
                            "analyst": i,
                            "answer": point["description"],
                            "score": point.get("score", 1),
                        }
                    )
            final_support_points = [p for p in final_support_points if p["score"] > 0]
            if not len(final_support_points):
                return QueryPrompt.FAIL_RESPONSE
            final_support_points = sorted(
                final_support_points, key=lambda x: x["score"], reverse=True
            )
            final_support_points = truncate_list_by_token_size(
                final_support_points,
                key=lambda x: x["answer"],
                max_token_size=self.config.global_max_token_for_community_report,
            )
            points_context = []
            for dp in final_support_points:
                points_context.append(
                    f"""----Analyst {dp['analyst']}----
        Importance Score: {dp['score']}
        {dp['answer']}
        """
                )
            points_context = "\n".join(points_context)
            if self.config.only_need_context:
                return points_context
            sys_prompt_temp = QueryPrompt.GLOBAL_REDUCE_RAG_RESPONSE
            response = await self.llm.aask(
                query,
                system_msgs= [sys_prompt_temp.format(
                    report_data=points_context, response_type=self.query_config.response_type
                )],
            )                
            return response
