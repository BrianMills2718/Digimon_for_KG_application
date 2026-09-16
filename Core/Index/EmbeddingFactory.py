"""Embedding provider factory for DIGIMON."""

from __future__ import annotations

from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from Config.EmbConfig import EmbeddingType
from Config.LLMConfig import LLMType
from Core.Common.BaseFactory import GenericFactory
from Option.Config2 import Config


class RAGEmbeddingFactory(GenericFactory):
    """Create the configured embedding provider without eager optional imports."""

    def __init__(self):
        creators = {
            EmbeddingType.OPENAI: self._create_openai,
            EmbeddingType.OLLAMA: self._create_ollama,
            EmbeddingType.HF: self._create_hf,
        }
        super().__init__(creators)

    def get_rag_embedding(
        self, key: EmbeddingType = None, config: Config = None
    ) -> BaseEmbedding:
        return super().get_instance(
            key or self._resolve_embedding_type(config), config=config
        )

    @staticmethod
    def _resolve_embedding_type(config) -> EmbeddingType | LLMType:
        if config.embedding.api_type:
            return config.embedding.api_type
        raise TypeError("To use RAG, please configure an embedding provider.")

    def _create_openai(self, config) -> OpenAIEmbedding:
        params = {}
        api_key = config.embedding.api_key or config.llm.api_key
        api_base = config.embedding.base_url or config.llm.base_url

        # Omit empty credentials so the OpenAI/LlamaIndex SDK can use the
        # standard OPENAI_API_KEY environment variable on clean checkouts.
        if api_key:
            params["api_key"] = api_key
        if api_base:
            params["api_base"] = api_base

        self._try_set_model_and_batch_size(params, config)
        return OpenAIEmbedding(**params)

    def _create_ollama(self, config) -> BaseEmbedding:
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
        except ImportError as exc:
            raise ImportError(
                "Ollama embeddings require the optional package "
                "'llama-index-embeddings-ollama'."
            ) from exc

        params = {}
        if config.embedding.base_url:
            params["base_url"] = config.embedding.base_url
        self._try_set_model_and_batch_size(params, config)
        return OllamaEmbedding(**params)

    def _create_hf(self, config) -> BaseEmbedding:
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError as exc:
            raise ImportError(
                "Hugging Face embeddings require the optional package "
                "'llama-index-embeddings-huggingface'."
            ) from exc

        params = {
            "model_name": config.embedding.model,
            "device": "cuda",
            "target_devices": ["cuda:7"],
            "embed_batch_size": 128,
        }
        if config.embedding.cache_folder:
            params["cache_folder"] = config.embedding.cache_folder
        return HuggingFaceEmbedding(**params)

    @staticmethod
    def _try_set_model_and_batch_size(params: dict, config):
        if config.embedding.model:
            params["model_name"] = config.embedding.model
        if config.embedding.embed_batch_size:
            params["embed_batch_size"] = config.embedding.embed_batch_size
        if config.embedding.dimensions:
            params["dimensions"] = config.embedding.dimensions

    def _raise_for_key(self, key: Any):
        raise ValueError(
            f"The embedding type is currently not supported: `{type(key)}`, {key}"
        )


get_rag_embedding = RAGEmbeddingFactory().get_rag_embedding
