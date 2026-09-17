from enum import Enum
from typing import Optional

from pydantic import field_validator

from Core.Utils.YamlModel import YamlModel


class EmbeddingType(Enum):
    OPENAI = "openai"
    HF = "hf"
    OLLAMA = "ollama"


def _looks_like_api_key_placeholder(text: str) -> bool:
    normalized = text.strip().upper()
    if not normalized:
        return True
    if normalized in {"CHANGEME", "REPLACE_ME", "PLACEHOLDER"}:
        return True
    if normalized.startswith("YOUR_API_KEY"):
        return True
    return normalized.startswith("YOUR_") and "API_KEY" in normalized


class EmbeddingConfig(YamlModel):
    """Embedding provider configuration."""

    api_type: Optional[EmbeddingType] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None

    model: Optional[str] = None
    cache_folder: Optional[str] = None
    embed_batch_size: Optional[int] = None
    dimensions: Optional[int] = None

    @field_validator("api_type", mode="before")
    @classmethod
    def check_api_type(cls, value):
        if value == "":
            return None
        return value

    @field_validator("api_key", mode="before")
    @classmethod
    def check_api_key(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return None if _looks_like_api_key_placeholder(text) else text
