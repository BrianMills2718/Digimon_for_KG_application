#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
from typing import Optional

from pydantic import field_validator

from Core.Common.Constants import CONFIG_ROOT, LLM_API_TIMEOUT, GRAPHRAG_ROOT
from Core.Utils.YamlModel import YamlModel


class LLMType(Enum):
    OPENAI = "openai"
    FIREWORKS = "fireworks"
    OPEN_LLM = "open_llm"
    OLLAMA = "ollama"
    OLLAMA_GENERATE = "ollama.generate"
    OLLAMA_EMBEDDINGS = "ollama.embeddings"
    OLLAMA_EMBED = "ollama.embed"
    OPENROUTER = "openrouter"
    BEDROCK = "bedrock"
    ARK = "ark"
    LITELLM = "litellm"

    def __missing__(self, key):
        return self.OPENAI


class LLMConfig(YamlModel):
    """Configuration for an LLM provider."""

    api_key: str = ""
    api_type: LLMType = LLMType.OPENAI
    base_url: Optional[str] = None
    api_version: Optional[str] = None

    model: Optional[str] = None
    pricing_plan: Optional[str] = None

    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None
    endpoint: Optional[str] = None

    app_id: Optional[str] = None
    api_secret: Optional[str] = None
    domain: Optional[str] = None

    max_token: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: Optional[str] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: Optional[int] = None
    n: Optional[int] = None
    stream: bool = False
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    timeout: int = 600
    context_length: Optional[int] = None

    region_name: Optional[str] = None
    proxy: Optional[str] = None
    max_concurrent: int = 20
    calc_usage: bool = True
    use_system_prompt: bool = True

    @field_validator("api_key", mode="before")
    @classmethod
    def check_llm_key(cls, value):
        """Example placeholders mean "use provider environment credentials"."""
        if value is None:
            return ""
        text = str(value).strip()
        normalized = text.upper()
        if not text or normalized.startswith("YOUR_API_KEY") or normalized in {
            "CHANGEME",
            "REPLACE_ME",
            "PLACEHOLDER",
        }:
            return ""
        return text

    @field_validator("timeout")
    @classmethod
    def check_timeout(cls, value):
        return value or LLM_API_TIMEOUT
