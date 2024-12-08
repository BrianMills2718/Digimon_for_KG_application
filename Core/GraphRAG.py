import asyncio
from collections import defaultdict, Counter
from typing import Union, Any

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
from Core.Graph.GraphFactory import get_graph
from Core.Index import get_index
from Core.Index.IndexConfigFactory import get_index_config
from Core.Prompt import GraphPrompt, QueryPrompt
from Core.Storage.NameSpace import Workspace
from Core.Community.ClusterFactory import get_community
from Core.Storage.PickleBlobStorage import PickleBlobStorage
from colorama import Fore, Style, init
import json

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
        if data.config.use_entity_link_chunk:
            cls._entities_to_relationships = PickleBlobStorage(
                namespace=data.e2r_namespace, config=None
            )
            cls._relationships_to_chunks = PickleBlobStorage(
                namespace=data.r2c_namespace, config=None
            )
        return data

    @classmethod
    def _register_retriever_context(cls, data):
        # register the retriever context
        cls._retriever = None
        cls._retriever_context = defaultdict
        cls._retriever_context.update({"graph": True})
        cls._retriever_context.update({"chunks": True})
        cls._retriever_context.update({"entities_vdb": data.config.use_entities_vdb})
        cls._retriever_context.update({"relations_vdb": data.config.use_relations_vdb})
        cls._retriever_context.update({"community": data.config.use_community})
        cls._retriever_context.update({"community": data.config.use_entity_link_chunk})
        return data

    async def _extract_query_entities(self, query):
        entities = []
        try:
            ner_messages = GraphPrompt.NER.format(user_input=query)

            response_content = await self.llm.aask(ner_messages)
            entities = prase_json_from_response(response_content)

            if 'named_entities' not in entities:
                entities = []
            else:
                entities = entities['named_entities']

            entities = [clean_str(p) for p in entities]
        except Exception as e:
            logger.error('Error in Retrieval NER: {}'.format(e))

        return entities

    async def _extract_query_keywords(self, query, mode = "low"):
        kw_prompt = QueryPrompt.KEYWORDS_EXTRACTION.format(query=query)
        result = await self.llm.aask(kw_prompt)

        keywords_data = prase_json_from_response(result)
        if mode == "low":
            keywords = keywords_data.get("low_level_keywords", [])
            keywords = ", ".join(keywords)
        elif mode == "high":
            keywords = keywords_data.get("high_level_keywords", [])
            keywords = ", ".join(keywords)
        elif mode == "hybrid":
           low_level = keywords_data.get("low_level_keywords", [])
           high_level = keywords_data.get("high_level_keywords", [])
           keywords = [low_level, high_level]

        return keywords


    async def _build_retriever_context(self, query):
        """Build the retriever context for the given query."""
        logger.info("Building retriever context for the given query: {query}".format(query=query))

        self._retriever.register_context("graph", self.graph)
        self._retriever.register_context("query", query)
        for context_name, use_context in self._retriever_context.items():
            if use_context:
                self._retriever.register_context(context_name, getattr(self, context_name))

        if self.config.use_query_entity:
            try:
                query_entities = await self._extract_query_entities(query)
                self._retriever.register_context("query_entity", query_entities)
            except Exception as e:
                logger.error(f"Failed to extract query entities: {e}")
                raise

    async def _build_ppr_context(self):
        """
        Build the context for the Personalized PageRank (PPR) query.

        This function constructs two mappings:
            1. chunk_to_edge: Maps chunks (document sources) to edge indices.
            2. edge_to_entity: Maps edges to the entities (nodes) they connect.

        The function iterates over all edges in the graph, retrieves relevant metadata for each edge,
        and updates the mappings. These mappings are essential for executing PPR queries efficiently.
        """
        self.chunk_to_edge = defaultdict(int)
        self.edge_to_entity = defaultdict(int)
        self.entity_to_edge = defaultdict(int)
        self.id_to_entity = defaultdict(int)

        nodes = list(await self.graph.nodes())
        edges = list(await self.graph.edges())

        async def _build_edge_chunk_mapping(edge) -> None:
            """
            Build mappings for the edges of a given graph.

            Args:
                edge (Tuple[str, str]): A tuple representing the edge (node1, node2).
                edges (list): List of all edges in the graph.
                nodes (list): List of all nodes in the graph.
                docs_to_facts (Dict[Tuple[int, int], int]): Mapping of document indices to fact indices.
                facts_to_phrases (Dict[Tuple[int, int], int]): Mapping of fact indices to phrase indices.
            """
            try:
                # Fetch edge data asynchronously
                edge_data = await self.graph.get_edge(edge[0], edge[1])
                source_ids = edge_data['source_id'].split(GRAPH_FIELD_SEP)
                for source_id in source_ids:
                    # Map document to edge
                    source_idx = await self.doc_chunk.get_index_by_key(source_id)
                    edge_idx = edges.index((edge[0], edge[1]))
                    self.chunk_to_edge[(source_idx, edge_idx)] = 1

                # Map fact to phrases for both nodes in the edge
                node_idx_1 = nodes.index(edge[0])
                node_idx_2 = nodes.index(edge[1])

                self.edge_to_entity[(edge_idx, node_idx_1)] = 1
                self.edge_to_entity[(edge_idx, node_idx_2)] = 1
                self.entity_to_edge[(node_idx_1, edge_idx)] = 1
                self.entity_to_edge[(node_idx_2, edge_idx)] = 1
            except ValueError as ve:
                # Handle specific errors, such as when edge or node is not found
                logger.error(f"ValueError in edge {edge}: {ve}")
            except KeyError as ke:
                # Handle missing data in chunk_key_to_idx
                logger.error(f"KeyError in edge {edge}: {ke}")
            except Exception as e:
                # Handle general exceptions gracefully
                logger.error(f"Unexpected error processing edge {edge}: {e}")

        # Process all nodes asynchronously
        await asyncio.gather(*[_build_edge_chunk_mapping(edge) for edge in edges])

        for node in nodes:
            self.id_to_entity[nodes.index(node)] = node

        self.chunk_to_edge_mat = csr_array(([int(v) for v in self.chunk_to_edge.values()], (
            [int(e[0]) for e in self.chunk_to_edge.keys()], [int(e[1]) for e in self.chunk_to_edge.keys()])),
                                           shape=(await self.doc_chunk.size, len(edges)))

        self.edge_to_entity_mat = csr_array(([int(v) for v in self.edge_to_entity.values()], (
            [e[0] for e in self.edge_to_entity.keys()], [e[1] for e in self.edge_to_entity.keys()])),
                                            shape=(len(edges), len(nodes)))

        self.chunk_to_entity_mat = self.chunk_to_edge_mat.dot(self.edge_to_entity_mat)
      
        self.chunk_to_entity_mat[self.chunk_to_entity_mat.nonzero()] = 1
        self.entity_doc_count = self.chunk_to_entity_mat.sum(0).T

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
        logger.info("Starting chunk the given documents")
        await self.doc_chunk.build_chunks(docs)
        logger.info("✅ Finished the chunking stage")

        ####################################################################################################
        # 2. Building Graph Stage
        ####################################################################################################
        logger.info("Starting build graph for the given documents")
        await self.graph.build_graph(await self.doc_chunk.get_chunks(), False)

        ####################################################################################################
        # 3. Index building Stage 
        # Data-driven content should be pre-built offline to ensure efficient online query performance.
        ####################################################################################################

        # NOTE: ** Ensure the graph is successfully loaded before proceeding to load the index from storage, as it represents a one-to-one mapping. **
        if self.config.use_entities_vdb:
            logger.info("Starting insert entities of the given graph into vector database")
            # await self.entities_vdb.build_index(await self.graph.nodes(), entity_metadata, force=False)
            await self.entities_vdb.build_index(await self.graph.nodes_data(), await self.graph.node_metadata(), False)
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
            logger.info("Starting insert relations of the given graph into vector database")
      
           
      
            await self.relations_vdb.build_index(await self.graph.edges_data(), await self.graph.edge_metadata(), force=False)
            logger.info("✅ Finished starting insert relations of the given graph into vector database")

        if self.config.use_community:
            logger.info("Starting build community of the given graph")
            logger.start("Clustering nodes")
            await self.community.cluster(largest_cc=await self.graph.stable_largest_cc(),
                                         max_cluster_size=self.config.max_graph_cluster_size,
                                         random_seed=self.config.graph_cluster_seed)

            await self.community.generate_community_report(self.graph, False)
            logger.info("✅ [Community Report]  Finished")
         

         

    async def query(self, query):
        """
            Executes the query by extracting the relevant content, and then generating a response.
            Args:
                query: The query to be processed.
            Returns:
            """
        ####################################################################################################
        # 1. Building query relevant content (subgraph) Stage
        ####################################################################################################
        # await self._build_retriever_context(query)
        # await self._build_retriever_operator()
        # await self.global_query(query)
        keywords = await self._extract_query_keywords(query, "high")
        relationships = await self._find_relevant_relations_vdb(keywords)
        entities = await self._find_relevant_entities_vdb(query)

        
        entities_by_relations = await self._find_relevant_entities_by_relationships(relationships)
        await self._find_relevant_chunks_from_relationships(entities_by_relations)
        query_entities = await self._extract_query_entities(query)
        link_entities = await self._link_entities(query_entities)
        entities = await self._find_relevant_entities_vdb(query)
        ppr_entites, ppr_node_matrix = await self._find_relevant_entities_by_ppr(query, query_entities)

        await self._find_relevant_community_from_entities(ppr_entites, self.community.community_reports)

        # await self._find_relevant_entities_from_keywords(query)
        await self._find_relevant_relationships_by_ppr(query, query_entities, 5, ppr_node_matrix)
        await self._find_relevant_chunks_by_ppr(entities)

        await self.graph.get_induced_subgraph(entities, relationships)
        # entities = await self._find_relevant_entities_vdb(query)
        # await self._find_relevant_chunks_from_entities(entities)
  
        relationships = await self._find_relevant_relations_vdb(query)
        print(relationships)

        ####################################################################################################
        # 2. Generation Stage
        ####################################################################################################

    async def _find_relevant_entities_by_relationships(self, edge_datas):
        # ✅
        entity_names = set()
        for e in edge_datas:
            entity_names.add(e["src_id"])
            entity_names.add(e["tgt_id"])

        node_datas = await asyncio.gather(
            *[self.graph.get_node(entity_name) for entity_name in entity_names]
        )

        node_degrees = await asyncio.gather(
            *[self.graph.node_degree(entity_name) for entity_name in entity_names]
        )
        node_datas = [
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(entity_names, node_datas, node_degrees)
        ]

        node_datas = truncate_list_by_token_size(
            node_datas,
            key=lambda x: x["description"],
            max_token_size = self.config.max_token_for_local_context,
        )

        return node_datas
    
    async def _run_personalized_pagerank(self, query, query_entities):
        # ✅
        assert self.config.use_entities_vdb
        # Run Personalized PageRank
        reset_prob_matrix = np.zeros(self.graph.node_num)

        if self.config.use_entity_similarity_for_ppr:
            # Here, we re-implement the key idea of the FastGraphRAG, you can refer to the source code for more details:
            # https://github.com/circlemind-ai/fast-graphrag/tree/main

            # Use entity similarity to compute the reset probability matrix
            reset_prob_matrix += await self.entities_vdb.retrieval_nodes_with_score_matrix(query_entities, top_k=1, graph = self.graph)
            # Run Personalized PageRank on the linked entities      
            reset_prob_matrix += await self.entities_vdb.retrieval_nodes_with_score_matrix(query, top_k=self.config.top_k_entity_for_ppr, graph = self.graph)     
        else:
            # Set the weight of the retrieved documents based on the number of documents they appear in
            # Please refer to the HippoRAG code for more details: https://github.com/OSU-NLP-Group/HippoRAG/tree/main
            if not hasattr(self, "entity_chunk_count"):
                    # Register the entity-chunk count matrix into the class when you first use it.
                    e2r = await self._entities_to_relationships.get()
                    r2c = await self._relationships_to_chunks.get()
                    c2e= e2r.dot(r2c).T
                    c2e[c2e.nonzero()] = 1
                    self.entity_chunk_count = c2e.sum(0).T
            for entity in query_entities:
                entity_idx = await self.graph.get_node_index(entity["entity_name"])
                if self.config.node_specificity:
                    if self.entity_chunk_count[entity_idx] == 0:
                        weight = 1
                    else:
                        weight = 1 / float(self.entity_chunk_count[entity_idx])
                    reset_prob_matrix[entity_idx] = weight
                else:
                    reset_prob_matrix[entity_idx] = 1.0
        #TODO: as a method in our NetworkXGraph class or directly use the networkx graph
        # Transform the graph to igraph format 
        return await self.graph.personalized_pagerank([reset_prob_matrix])
        
    
    # def get_colbert_max_score(self, query):
    async def _find_relevant_entities_by_ppr(self, query, seed_entities: list[dict], top_k=5):
        # ✅
        if len(seed_entities) == 0:
            return None
        # Create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
        ppr_node_matrix = await self._run_personalized_pagerank(query, seed_entities)
        topk_indices = np.argsort(ppr_node_matrix)[-top_k:]
        nodes = await self.graph.get_node_by_indices(topk_indices)
 
        return nodes, ppr_node_matrix

    async def _find_relevant_relationships_by_ppr(self, query, seed_entities: list[dict], top_k=5, node_ppr_matrix=None):
        # ✅
        entity_to_edge_mat = await self._entities_to_relationships.get()
        if node_ppr_matrix is None:
        # Create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
            node_ppr_matrix = await self._run_personalized_pagerank(query, seed_entities)
        edge_prob_matrix = entity_to_edge_mat.T.dot(node_ppr_matrix)
        topk_indices = np.argsort(edge_prob_matrix)[-top_k:]
        edges =  await self.graph.get_edge_by_indices(topk_indices)
        return await self._construct_relationship_context(edges)
    
    async def _find_relevant_chunks_by_ppr(self, query, seed_entities: list[dict], top_k=5, node_ppr_matrix=None):
        # ✅
        entity_to_edge_mat = await self._entities_to_relationships.get()
        relationship_to_chunk_mat = await self._relationships_to_chunks.get()
        if node_ppr_matrix is None:
        # Create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
            node_ppr_matrix = await self._run_personalized_pagerank(query, seed_entities)
        edge_prob = entity_to_edge_mat.T.dot(node_ppr_matrix)
        ppr_chunk_prob = relationship_to_chunk_mat.T.dot(edge_prob)
        ppr_chunk_prob = min_max_normalize(ppr_chunk_prob)
         # Return top k documents
        sorted_doc_ids = np.argsort(ppr_chunk_prob, kind='mergesort')[::-1]
        sorted_scores = ppr_chunk_prob[sorted_doc_ids]
        soreted_docs = await self.doc_chunk.get_data_by_indices(sorted_doc_ids[:top_k])
        return soreted_docs, sorted_scores[:top_k]
    
    async def _link_entities(self, query_entities):

        entities = []
        for query_entity in query_entities: 
            node_datas = await self.entities_vdb.retrieval_nodes(query_entity, top_k=1, graph = self.graph)
            # For entity link, we only consider the top-ranked entity
            entities.append(node_datas[0]) 
 
        return entities

    async def _find_relevant_relations_by_entity_agent(self, query: str, entity: str, pre_relations_name=None,
                                                       pre_head=None, width=3):
        """
        Use agent to select the top-K relations based on the input query and entities
        Args:
            query: str, the query to be processed.
            entity: str, the entity seed
            pre_relations_name: list, the relation name that has already exists
            pre_head: bool, indicator that shows whether te pre head relations exist or not
            width: int, the search width of agent
        Returns:
            results: list[str], top-k relation candidates list
        """
        # ✅
        try:
            from Core.Common.Constants import GRAPH_FIELD_SEP
            from collections import defaultdict
            from Core.Prompt.TogPrompt import extract_relation_prompt

            # get relations from graph
            edges = await self.graph._graph.get_node_edges(source_node_id=entity)
            relations_name_super_edge = await asyncio.gather(
                *[self.graph._graph.get_edge_relation_name(edge[0], edge[1]) for edge in edges]
            )
            relations_name = list(map(lambda x: x.split(GRAPH_FIELD_SEP), relations_name_super_edge))  # [[], [], []]

            relations_dict = defaultdict(list)
            for index, edge in enumerate(edges):
                src, tar = edge[0], edge[1]
                for rel in relations_name[index]:
                    relations_dict[(src, rel)].append(tar)

            tail_relations = []
            head_relations = []
            for index, rels in enumerate(relations_name):
                if edges[index][0] == entity:
                    head_relations.extend(rels)  # head
                else:
                    tail_relations.extend(rels)  # tail

            if pre_relations_name:
                if pre_head:
                    tail_relations = list(set(tail_relations) - set(pre_relations_name))
                else:
                    head_relations = list(set(head_relations) - set(pre_relations_name))

            head_relations = list(set(head_relations))
            tail_relations = list(set(tail_relations))
            total_relations = head_relations + tail_relations
            total_relations.sort()  # make sure the order in prompt is always equal

            # agent
            prompt = extract_relation_prompt % (
            width, width) + query + '\nTopic Entity: ' + entity + '\nRelations: ' + '; '.join(
                total_relations) + ';' + "\nA: "
            result = await self.llm.aask(msg=[
                {"role": "user",
                 "content": prompt}
            ])

            # clean
            import re
            pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
            relations = []
            for match in re.finditer(pattern, result):
                relation = match.group("relation").strip()
                if ';' in relation:
                    continue
                score = match.group("score")
                if not relation or not score:
                    return False, "output uncompleted.."
                try:
                    score = float(score)
                except ValueError:
                    return False, "Invalid score"
                if relation in head_relations:
                    relations.append({"entity": entity, "relation": relation, "score": score, "head": True})
                else:
                    relations.append({"entity": entity, "relation": relation, "score": score, "head": False})

            if len(relations) == 0:
                flag = False
                logger.info("No relations found by entity: {} and query: {}".format(entity, query))
            else:
                flag = True

            # return
            if flag:
                return relations, relations_dict
            else:
                return [], relations_dict
        except Exception as e:
            logger.exception(f"Failed to find relevant relations by entity agent: {e}")

    async def _find_relevant_entities_by_relation_agent(self, query: str, current_entity_relations_list: list[dict],
                                                        relations_dict: defaultdict[list], width=3):
        """
        Use agent to select the top-K relations based on the input query and entities
        Args:
            query: str, the query to be processed.
            current_entity_relations_list: list,  whose element is {"entity": entity_name, "relation": relation, "score": score, "head": bool}
            relations_dict: defaultdict[list], key is (src, rel), value is tar
        Returns:
            flag: bool,  indicator that shows whether to reason or not
            relations, heads
            cluster_chain_of_entities: list[list], reasoning paths
            candidates: list[str], entity candidates
            relations: list[str], related relation
            heads: list[bool]
        """
        # ✅
        try:
            from Core.Prompt.TogPrompt import  score_entity_candidates_prompt
            total_candidates = []
            total_scores = []
            total_relations = []
            total_topic_entities = []
            total_head = []

            for index, entity in enumerate(current_entity_relations_list):
                candidate_list = relations_dict[(entity["entity"], entity["relation"])]

                # score these candidate entities
                if len(candidate_list) == 1:
                    scores = [entity["score"]]
                elif len(candidate_list) == 0:
                    scores = [0.0]
                else:
                    # agent
                    prompt = score_entity_candidates_prompt.format(query, entity["relation"]) + '; '.join(
                        candidate_list) + ';' + '\nScore: '
                    result = await self.llm.aask(msg=[
                        {"role": "user",
                         "content": prompt}
                    ])

                    # clean
                    import re
                    scores = re.findall(r'\d+\.\d+', result)
                    scores = [float(number) for number in scores]
                    if len(scores) != len(candidate_list):
                        logger.info("All entities are created with equal scores.")
                        scores = [1 / len(candidate_list)] * len(candidate_list)

                # update
                if len(candidate_list) == 0:
                    candidate_list.append("[FINISH]")
                candidates_relation = [entity['relation']] * len(candidate_list)
                topic_entities = [entity['entity']] * len(candidate_list)
                head_num = [entity['head']] * len(candidate_list)
                total_candidates.extend(candidate_list)
                total_scores.extend(scores)
                total_relations.extend(candidates_relation)
                total_topic_entities.extend(topic_entities)
                total_head.extend(head_num)

            # entity_prune
            zipped = list(zip(total_relations, total_candidates, total_topic_entities, total_head, total_scores))
            sorted_zipped = sorted(zipped, key=lambda x: x[4], reverse=True)
            sorted_relations = list(map(lambda x: x[0], sorted_zipped))
            sorted_candidates = list(map(lambda x: x[1], sorted_zipped))
            sorted_topic_entities = list(map(lambda x: x[2], sorted_zipped))
            sorted_head = list(map(lambda x: x[3], sorted_zipped))
            sorted_scores = list(map(lambda x: x[4], sorted_zipped))

            # prune according to width
            relations = sorted_relations[:width]
            candidates = sorted_candidates[:width]
            topics = sorted_topic_entities[:width]
            heads = sorted_head[:width]
            scores = sorted_scores[:width]

            # merge and output
            merged_list = list(zip(relations, candidates, topics, heads, scores))
            filtered_list = [(rel, ent, top, hea, score) for rel, ent, top, hea, score in merged_list if score != 0]
            if len(filtered_list) == 0:
                return False, [], [], [], []
            relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))
            cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
            return True, cluster_chain_of_entities, candidates, relations, heads
        except Exception as e:
            logger.exception(f"Failed to find relevant entities by relation agent: {e}")
      
      
        

    async def _find_relevant_entities_vdb(self, query, top_k=5):
        # ✅
        try:
            assert self.config.use_entities_vdb
            node_datas = await self.entities_vdb.retrieval_nodes(query, top_k, self.graph)             
            if not len(node_datas):
                return None
            if not all([n is not None for n in node_datas]):
                logger.warning("Some nodes are missing, maybe the storage is damaged")
            node_degrees = await asyncio.gather(
                *[self.graph.node_degree(node["entity_name"]) for node in node_datas]
            )
            node_datas = [
                {**n, "entity_name": n["entity_name"], "rank": d}
                for n, d in zip( node_datas, node_degrees)
                if n is not None
            ]
  
            return node_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant entities_vdb: {e}")


    async def _find_relevant_tree_nodes_vdb(self, query, top_k=5):
        # ✅
        try:
            assert self.config.use_entities_vdb
            node_datas = await self.entities_vdb.retrieval(query, top_k)
            import pdb
            pdb.set_trace()             
            if not len(node_datas):
                return None
            if not all([n is not None for n in node_datas]):
                logger.warning("Some nodes are missing, maybe the storage is damaged")
            node_degrees = await asyncio.gather(
                *[self.graph.node_degree(node["entity_name"]) for node in node_datas]
            )
            node_datas = [
                {**n, "entity_name": n["entity_name"], "rank": d}
                for n, d in zip( node_datas, node_degrees)
                if n is not None
            ]
  
            return node_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant entities_vdb: {e}")
    async def _find_relevant_relations_vdb(self, query, top_k=5):
        # ✅
        try:
            if query is None: return None
            assert self.config.use_relations_vdb
            edge_datas = await self.relations_vdb.retrieval_edges(query, top_k=top_k, graph = self.graph)
        
            if not len(edge_datas):
                return None
     
            edge_datas = await self._construct_relationship_context(edge_datas)
            return edge_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant relationships: {e}")

    async def _find_relevant_chunks_from_entities_relationships(self, node_datas: list[dict]):
        # ✅
        if len(node_datas) == 0:
            return None
        text_units = [
            split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
            for dp in node_datas
        ]
        edges = await asyncio.gather(
            *[self.graph.get_node_edges(dp["entity_name"]) for dp in node_datas]
        )
        all_one_hop_nodes = set()
        for this_edges in edges:
            if not this_edges:
                continue
            all_one_hop_nodes.update([e[1] for e in this_edges])
        all_one_hop_nodes = list(all_one_hop_nodes)
        all_one_hop_nodes_data = await asyncio.gather(
            *[self.graph.get_node(e) for e in all_one_hop_nodes]
        )
        all_one_hop_text_units_lookup = {
            k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
            for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
            if v is not None
        }
        all_text_units_lookup = {}
        for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
            for c_id in this_text_units:
                if c_id in all_text_units_lookup:
                    continue
                relation_counts = 0
                for e in this_edges:
                    if (
                            e[1] in all_one_hop_text_units_lookup
                            and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1
                all_text_units_lookup[c_id] = {
                    "data": await self.doc_chunk.get_data_by_key(c_id),
                    "order": index,
                    "relation_counts": relation_counts,
                }
        if any([v is None for v in all_text_units_lookup.values()]):
            logger.warning("Text chunks are missing, maybe the storage is damaged")
        all_text_units = [
            {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
        ]   
        # for node_data in node_datas:
        all_text_units = sorted(
            all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
        )
        all_text_units = truncate_list_by_token_size(
            all_text_units,
            key=lambda x: x["data"],
            max_token_size=self.config.local_max_token_for_text_unit,
        )
        all_text_units = [t["data"] for t in all_text_units]

        return all_text_units


    async def _find_relevant_chunks_from_relationships(self, edge_datas: list[dict]):
        text_units = [
            split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
            for dp in edge_datas
        ]

        all_text_units_lookup = {}

        for index, unit_list in enumerate(text_units):
            for c_id in unit_list:
                if c_id not in all_text_units_lookup:
                    all_text_units_lookup[c_id] = {
                        "data": await self.doc_chunk.get_data_by_key(c_id),
                        "order": index,
                    }

        if any([v is None for v in all_text_units_lookup.values()]):
            logger.warning("Text chunks are missing, maybe the storage is damaged")
        all_text_units = [
            {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
        ]
        all_text_units = sorted(all_text_units, key=lambda x: x["order"])
        all_text_units = truncate_list_by_token_size(
            all_text_units,
            key=lambda x: x["data"],
            max_token_size = self.config.max_token_for_text_unit,
        )
        all_text_units = [t["data"] for t in all_text_units]

        return all_text_units
    
    async def _find_relevant_relationships_from_entities(self, node_datas: list[dict]):
        # ✅
        all_related_edges = await asyncio.gather(
            *[self.graph.get_node_edges(node["entity_name"]) for node in node_datas]
        )
        all_edges = set()
        for this_edges in all_related_edges:
            all_edges.update([tuple(sorted(e)) for e in this_edges])
        all_edges = list(all_edges)
        all_edges_pack = await asyncio.gather(
            *[self.graph.get_edge(e[0], e[1]) for e in all_edges]
        )
        all_edges_degree = await asyncio.gather(
            *[self.graph.edge_degree(e[0], e[1]) for e in all_edges]
        )
        all_edges_data = [
            {"src_tgt": k, "rank": d, **v}
            for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
            if v is not None
        ]
        all_edges_data = sorted(
            all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        all_edges_data = truncate_list_by_token_size(
            all_edges_data,
            key=lambda x: x["description"],
            max_token_size=self.config.max_token_for_local_context,
        )
        return all_edges_data




    async def _construct_relationship_context(self, edge_datas: list[dict]):

        if not all([n is not None for n in edge_datas]):
            logger.warning("Some edges are missing, maybe the storage is damaged")
        edge_degree = await asyncio.gather(
            *[self.graph.edge_degree(r["src_id"], r["tgt_id"]) for r in edge_datas]
        )
        edge_datas = [
            {"src_id": v["src_id"], "tgt_id": v["tgt_id"], "rank": d, **v}
            for v, d in zip( edge_datas, edge_degree)
            if v is not None
        ]
        edge_datas = sorted(
            edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        edge_datas = truncate_list_by_token_size(
            edge_datas,
            key=lambda x: x["description"],
            max_token_size=self.config.max_token_for_global_context,
        )
        return edge_datas
    
    async def _find_relevant_community_from_entities(self, node_datas: list[dict], community_reports):
        # ✅
        related_communities = []
        for node_d in node_datas:
            if "clusters" not in node_d:
                continue
            related_communities.extend(json.loads(node_d["clusters"]))
        related_community_dup_keys = [
            str(dp["cluster"])
            for dp in related_communities
            if dp["level"] <= self.config.level
        ]
        import pdb
        pdb.set_trace()
        related_community_keys_counts = dict(Counter(related_community_dup_keys))
        _related_community_datas = await asyncio.gather(
            *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
        )
        related_community_datas = {
            k: v
            for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
            if v is not None
        }
        related_community_keys = sorted(
            related_community_keys_counts.keys(),
            key=lambda k: (
                related_community_keys_counts[k],
                related_community_datas[k]["report_json"].get("rating", -1),
            ),
            reverse=True,
        )
        sorted_community_datas = [
            related_community_datas[k] for k in related_community_keys
        ]

        use_community_reports = truncate_list_by_token_size(
            sorted_community_datas,
            key=lambda x: x["report_string"],
            max_token_size= self.config.local_max_token_for_community_report,
        )
        if self.config.local_community_single_one:
            use_community_reports = use_community_reports[:1]

        return use_community_reports
    

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
            import pdb
            pdb.set_trace()
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
