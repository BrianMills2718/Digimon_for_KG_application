#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from dataclasses import field
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from pydantic import BaseModel
from Config import *
from Core.Common.Constants import CONFIG_ROOT, GRAPHRAG_ROOT
from Core.Utils.YamlModel import YamlModel
import json
from pathlib import Path
# from Core.Common.Logger import logger  # Moved to inside parse to avoid circular import

class WorkingParams(BaseModel):
    """Working parameters"""

    working_dir: str = ""
    exp_name: str = ""
    data_root: str = ""
    dataset_name: str = ""


import os
import yaml
from pathlib import Path
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field
from Core.Utils.YamlModel import YamlModel
from Config.LLMConfig import LLMConfig
from Config.EmbConfig import EmbeddingConfig
from Config.GraphConfig import GraphConfig
from Config.ChunkConfig import ChunkConfig
from Config.QueryConfig import QueryConfig
from Config.RetrieverConfig import RetrieverConfig

class StorageConfig(BaseModel):
    root_dir: str = "./results"
    artifact_dir: str = "artifacts"

class Config(WorkingParams, YamlModel):
    """Configurations for our project"""

    # Key Parameters
    llm: LLMConfig
    exp_name: str = "default"
    # RAG Embedding
    embedding: EmbeddingConfig = EmbeddingConfig()

    # Basic Config
    use_entities_vdb: bool = True
    use_relations_vdb: bool = True  # Only set True for LightRAG
    use_subgraphs_vdb: bool = False  # Only set True for Medical-GraphRAG
    vdb_type: str = "vector"  # vector/colbert
    token_model: str = "gpt-3.5-turbo"
    
    
    # Chunking
    chunk: ChunkConfig = ChunkConfig()
    # chunk_token_size: int = 1200
    # chunk_overlap_token_size: int = 100
    # chunk_method: str = "chunking_by_token_size"
   
    llm_model_max_token_size: int = 32768
  
    use_entity_link_chunk: bool = True  # Only set True for HippoRAG and FastGraphRAG
    
    # Graph Config
    graph: GraphConfig = Field(default_factory=GraphConfig)
    
    # Retrieval Parameters
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)

    # Storage Config
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # ColBert Option
    use_colbert: bool = True
    colbert_checkpoint_path: str = "/home/yingli/HippoRAG/exp/colbertv2.0"
    index_name: str = "nbits_2"
    similarity_max: float = 1.0
    # Graph Augmentation

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_yaml_file(cls, path: str):
        """Load config from YAML file"""
        path = Path(path).resolve()
        with open(path, 'r') as f:
            options = yaml.safe_load(f)
        return cls(**options)

    @classmethod
    def parse(cls, options_yaml_path: Path, dataset_name: str = "DefaultDataset", exp_name: str = "DefaultExp") -> 'Config':
        from Core.Common.Logger import logger  # Import here to avoid circular import
        if not options_yaml_path.exists():
            logger.error(f"Configuration file not found: {options_yaml_path}")
            raise FileNotFoundError(f"Configuration file not found: {options_yaml_path}")

        import yaml
        with open(options_yaml_path, 'r') as f:
            options = yaml.safe_load(f)

        llm_config = LLMConfig(**options.get("llm_config", {}))
        emb_config = EmbeddingConfig(**options.get("emb_config", {}))
        chunk_config = ChunkConfig(**options.get("chunk_config", {}))
        graph_config = GraphConfig(**options.get("graph_config", {}))
        retriever_config = RetrieverConfig(**options.get("retriever_config", {}))
        query_config = QueryQueryConfig(**options.get("query_config", {}))
        storage_config = StorageConfig(**options.get("storage_config", {}))
        query_config = QueryConfig(**options.get("query_config", {}))

        # --- Load Custom Ontology ---
        if getattr(graph_config, 'custom_ontology_path', None):
            ontology_file = Path(graph_config.custom_ontology_path)
            if ontology_file.exists() and ontology_file.is_file():
                try:
                    with open(ontology_file, 'r') as f_ont:
                        graph_config.loaded_custom_ontology = json.load(f_ont)
                    logger.info(f"Successfully loaded custom ontology from: {ontology_file}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from custom ontology file {ontology_file}: {e}")
                    graph_config.loaded_custom_ontology = None
                except Exception as e:
                    logger.error(f"Failed to load custom ontology file {ontology_file}: {e}")
                    graph_config.loaded_custom_ontology = None
            else:
                logger.warning(f"Custom ontology file not found at: {ontology_file}. Proceeding without custom ontology.")
                graph_config.loaded_custom_ontology = None
        else:
            logger.info("No custom ontology path specified. Proceeding without custom ontology.")
            graph_config.loaded_custom_ontology = None
        # --- End Load Custom Ontology ---

        return cls(
            dataset_name=dataset_name,
            exp_name=exp_name,
            llm=llm_config,
            embedding=emb_config,
            chunk=chunk_config,
            graph=graph_config,
            retriever=retriever_config,
            raw_yaml_path=str(options_yaml_path)
        )
    enable_graph_augmentation: bool = True

    # Query Config 
    query: QueryConfig = QueryConfig()
  
    @classmethod
    def from_yaml_config(cls, path: str):
        """user config llm
        example:
        llm_config = {"api_type": "xxx", "api_key": "xxx", "model": "xxx"}
        gpt4 = Option.from_llm_config(llm_config)
        A = Role(name="A", profile="Democratic candidate", goal="Win the election", actions=[a1], watch=[a2], config=gpt4)
        """
        opt = parse(path)
        return Config(**opt)

    @classmethod
    def parse(cls, _path, dataset_name, exp_name: Optional[str] = None):
        """Parse config from yaml file"""
        opt = [parse(_path)]

        default_config_paths: List[Path] = [
            GRAPHRAG_ROOT / "Option/Config2.yaml",
            CONFIG_ROOT / "Config2.yaml",
        ]
        opt += [Config.read_yaml(path) for path in default_config_paths]
    
        final = merge_dict(opt)
        final["dataset_name"] = dataset_name
        final["working_dir"] = os.path.join(final["working_dir"], dataset_name)
        if exp_name is not None:
            final["exp_name"] = exp_name
        elif "exp_name" not in final:
            final["exp_name"] = cls.exp_name
        # else: use exp_name from YAML if present
        return Config(**final)
    
    @classmethod
    def default(cls):
        """Generate a default config, trying multiple sources for robustness."""
        import yaml
        import os
        from Config.LLMConfig import LLMConfig, LLMType
        from Config.EmbConfig import EmbeddingConfig, EmbeddingType
        from Config.GraphConfig import GraphConfig
        from Config.ChunkConfig import ChunkConfig
        from Config.RetrieverConfig import RetrieverConfig
        from Config.QueryConfig import QueryConfig
        
        final_config_data = {}
        default_yaml_path_primary = "Option/Config2.example.yaml" # Original default target
        default_yaml_path_secondary = "Option/Config2.yaml"     # Fallback to the main config

        loaded_from_file = False
        try:
            if os.path.exists(default_yaml_path_primary):
                with open(default_yaml_path_primary, 'r') as f:
                    final_config_data = yaml.safe_load(f)
                    if final_config_data and 'llm' in final_config_data and 'embedding' in final_config_data:
                        loaded_from_file = True
                        print(f"INFO [Config.default]: Loaded default config from {default_yaml_path_primary}")
            if not loaded_from_file and os.path.exists(default_yaml_path_secondary):
                print(f"WARNING [Config.default]: {default_yaml_path_primary} not found or incomplete. Attempting to load from {default_yaml_path_secondary}.")
                with open(default_yaml_path_secondary, 'r') as f:
                    final_config_data = yaml.safe_load(f)
                    if final_config_data and 'llm' in final_config_data and 'embedding' in final_config_data:
                        loaded_from_file = True
                        print(f"INFO [Config.default]: Loaded default config from {default_yaml_path_secondary}")
        except Exception as e:
            print(f"ERROR [Config.default]: Error loading default config from YAML files ({default_yaml_path_primary}, {default_yaml_path_secondary}): {e}. Proceeding with programmatic defaults for missing sections.")
            if not isinstance(final_config_data, dict):
                final_config_data = {}
        # Ensure all required fields have at least minimal valid defaults if not loaded.
        if 'llm' not in final_config_data or not isinstance(final_config_data.get('llm'), dict):
            print("WARNING [Config.default]: LLM config not found or invalid in default YAMLs/data, creating minimal programmatic default LLMConfig.")
            final_config_data['llm'] = LLMConfig(api_type=LLMType.OPENAI, model="gpt-3.5-turbo", api_key="YOUR_API_KEY_OR_PLACEHOLDER").model_dump()
        if 'embedding' not in final_config_data or not isinstance(final_config_data.get('embedding'), dict):
            print("WARNING [Config.default]: Embedding config not found or invalid in default YAMLs/data, creating minimal programmatic default EmbeddingConfig.")
            final_config_data['embedding'] = EmbeddingConfig(api_type=EmbeddingType.OPENAI, model="text-embedding-ada-002", api_key="YOUR_API_KEY_OR_PLACEHOLDER").model_dump()
        if 'graph' not in final_config_data or not isinstance(final_config_data.get('graph'), dict):
            print("WARNING [Config.default]: Graph config not found in default YAMLs/data, creating default GraphConfig.")
            final_config_data['graph'] = GraphConfig().model_dump()
        if 'chunk' not in final_config_data or not isinstance(final_config_data.get('chunk'), dict):
            print("WARNING [Config.default]: Chunk config not found in default YAMLs/data, creating default ChunkConfig.")
            final_config_data['chunk'] = ChunkConfig().model_dump()
        if 'retriever' not in final_config_data or not isinstance(final_config_data.get('retriever'), dict):
            print("WARNING [Config.default]: Retriever config not found or invalid in default YAMLs/data, creating default RetrieverConfig.")
            final_config_data['retriever'] = RetrieverConfig().model_dump()
        if 'query_config' not in final_config_data or not isinstance(final_config_data.get('query_config'), dict):
            print("WARNING [Config.default]: Query config (query_config) not found or invalid in default YAMLs/data, creating default QueryConfig.")
            final_config_data['query_config'] = QueryConfig().model_dump()
        try:
            instance = cls(**final_config_data)
            print("INFO [Config.default]: Successfully created default Config instance.")
            return instance
        except Exception as e:
            print(f"ERROR [Config.default]: CRITICAL: Failed to instantiate Config with final_config_data. Error: {e}")
            print(f"ERROR [Config.default]: Final config data used: {final_config_data}")
            raise

    @property
    def extra(self):
        return self._extra

    @extra.setter
    def extra(self, value: dict):
        self._extra = value

    def get_openai_llm(self) -> Optional[LLMConfig]:
        """Get OpenAI LLMConfig by name. If no OpenAI, raise Exception"""
        if self.llm.api_type == LLMType.OPENAI:
            return self.llm
        return None
def parse(opt_path):
    
        with open(opt_path, mode='r') as f:
            opt = YamlModel.read_yaml(opt_path)
        # export CUDA_VISIBLE_DEVICES
        # gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        return opt
def merge_dict(dicts: Iterable[Dict]) -> Dict:
    """Merge multiple dicts into one, with the latter dict overwriting the former"""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


default_config = Config.default() # which is used in other files, only for LLM, embedding and save file. 