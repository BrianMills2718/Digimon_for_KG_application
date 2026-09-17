from Config.EmbConfig import EmbeddingConfig, EmbeddingType
from Config.LLMConfig import LLMConfig, LLMType


def test_llm_placeholder_api_key_is_treated_as_unset():
    config = LLMConfig(
        api_type=LLMType.OPENAI,
        api_key="YOUR_API_KEY",
        model="gpt-4o-mini",
    )

    assert config.api_key == ""


def test_programmatic_llm_placeholder_is_treated_as_unset():
    config = LLMConfig(
        api_type=LLMType.OPENAI,
        api_key="YOUR_API_KEY_OR_PLACEHOLDER",
        model="gpt-4o-mini",
    )

    assert config.api_key == ""


def test_checked_in_openai_placeholder_is_treated_as_unset():
    config = LLMConfig(
        api_type=LLMType.OPENAI,
        api_key="YOUR_OPENAI_API_KEY_HERE",
        model="gpt-4o-mini",
    )

    assert config.api_key == ""


def test_embedding_placeholder_api_key_is_treated_as_unset():
    config = EmbeddingConfig(
        api_type=EmbeddingType.OPENAI,
        api_key="YOUR_API_KEY",
        model="text-embedding-3-small",
    )

    assert config.api_key is None


def test_checked_in_embedding_placeholder_is_treated_as_unset():
    config = EmbeddingConfig(
        api_type=EmbeddingType.OPENAI,
        api_key="YOUR_OPENAI_API_KEY_HERE",
        model="text-embedding-3-small",
    )

    assert config.api_key is None
