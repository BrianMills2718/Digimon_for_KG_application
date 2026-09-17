import pytest

import Core.Provider.LLMClientAdapter as adapter_module


def test_agentic_adapter_fails_at_construction_when_llm_client_missing(monkeypatch):
    def missing_module(name):
        assert name == "llm_client"
        raise ModuleNotFoundError("No module named 'llm_client'")

    monkeypatch.setattr(adapter_module.importlib, "import_module", missing_module)

    with pytest.raises(ModuleNotFoundError):
        adapter_module.LLMClientAdapter("test/model")


def test_agentic_adapter_requires_acall_llm_symbol(monkeypatch):
    class ModuleWithoutCall:
        pass

    monkeypatch.setattr(
        adapter_module.importlib,
        "import_module",
        lambda name: ModuleWithoutCall(),
    )

    with pytest.raises(ImportError, match="acall_llm"):
        adapter_module.LLMClientAdapter("test/model")
