#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, model_validator

from Core.Common.EmbConfig import EmbeddingConfig
from Core.Common.LLMConfig import LLMConfig, LLMType
from Core.Common.Constants import CONFIG_ROOT, GRAPHRAG_ROOT
from Core.Utils.YamlModel import YamlModel


class CLIParams(BaseModel):
    """CLI parameters"""

    project_path: str = ""
    project_name: str = ""
    inc: bool = False
    reqa_file: str = ""
    max_auto_summarize_code: int = 0
    git_reinit: bool = False

    @model_validator(mode="after")
    def check_project_path(self):
        """Check project_path and project_name"""
        if self.project_path:
            self.inc = True
            self.project_name = self.project_name or Path(self.project_path).name
        return self


class Config(CLIParams, YamlModel):
    """Configurations for our project"""

    # Key Parameters
    llm: LLMConfig
    exp_name: str = "debug"
    # RAG Embedding
    embedding: EmbeddingConfig = EmbeddingConfig()

    # Basic Config
    use_entities_vdb: bool = True
    use_relations_vdb: bool = True  # Only set True for LightRAG
    vdb_type: str = "vector"  # vector/colbert
    # Chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    token_model: str = "gpt-3.5-turbo"
    chunk_method: str = "chunking_by_token_size"
    use_entity_link_chunk: bool = False  # Only set True for HippoRAG and FastGraphRAG

    # enable LightRAG
    enable_keywords: bool = True

    # Building graph
    graph_type: str = "rkg_graph"
    max_gleaning: int = 1
    enable_entity_description: bool = True
    enable_entity_type: bool = True
    enable_edge_description: bool = True
    enable_edge_name: bool = True
    prior_prob: float = 0.8
    # Graph clustering
    use_community: bool = True
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    summary_max_tokens: int = 500


    # Tree Config
    reduction_dimension: Optional[int] = 5
    summarization_length: Optional[int] = 100
    num_layers: Optional[int] = 10
    top_k: Optional[int] = 5
    start_layer: Optional[int] = 5
    graph_cluster_params: Optional[dict] = None
    selection_mode: Optional[str] = "top_k"

    # Commuity report
    enforce_sub_communities: bool = False

    # Misc Parameters
    repair_llm_output: bool = False
    prompt_schema: Literal["json", "markdown", "raw"] = "json"

    # Retrieval Parameters
    enable_local: bool = False
    enable_naive_rag: bool = False
    node_specificity: bool = True
    damping: float = 0.1
    # ColBert Option
    use_colbert: bool = True
    colbert_checkpoint_path: str = "./Tools/Index/colbertv2.0"
    index_name: str = "nbits_2"
    similarity_max: float = 1.0
    # Graph Augmentation
    enable_graph_augmentation: bool = True

    @classmethod
    def from_home(cls, path):
        pathname = CONFIG_ROOT / path
        if not pathname.exists():
            return None
        return Config.from_yaml_file(pathname)

    @classmethod
    def default(cls):
        """Load default config
        - Priority: env < default_config_paths
        - Inside default_config_paths, the latter one overwrites the former one
        """
        default_config_paths: List[Path] = [
            GRAPHRAG_ROOT / "Option/Config2.yaml",
            CONFIG_ROOT / "Config2.yaml",
        ]

        dicts = [dict(os.environ)]
        dicts += [Config.read_yaml(path) for path in default_config_paths]

        final = merge_dict(dicts)
        return Config(**final)

    @classmethod
    def from_llm_config(cls, llm_config: dict):
        """user config llm
        example:
        llm_config = {"api_type": "xxx", "api_key": "xxx", "model": "xxx"}
        gpt4 = Option.from_llm_config(llm_config)
        A = Role(name="A", profile="Democratic candidate", goal="Win the election", actions=[a1], watch=[a2], config=gpt4)
        """

        llm_config = LLMConfig.model_validate(llm_config)
        dicts = [dict(os.environ)]
        dicts += [{"llm": llm_config}]
        final = merge_dict(dicts)
        return Config(**final)


def merge_dict(dicts: Iterable[Dict]) -> Dict:
    """Merge multiple dicts into one, with the latter dict overwriting the former"""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


config = Config.default()


