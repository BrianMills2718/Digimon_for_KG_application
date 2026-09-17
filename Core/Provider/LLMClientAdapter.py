"""LLMClientAdapter — wraps llm_client.acall_llm behind the BaseLLM interface.

Operators call ctx.llm.aask() unchanged. This adapter routes calls through
llm_client which handles retry, fallback, cost tracking, and model routing.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any, Dict, Optional

from Config.LLMConfig import LLMConfig, LLMType
from Core.Common.Logger import logger
from Core.Provider.BaseLLM import BaseLLM


def _load_llm_client_acall():
    """Resolve llm_client at adapter construction/call time.

    The MCP bootstrap catches ``ImportError`` around agentic-adapter creation and
    falls back to the default DIGIMON LLM. Importing the optional dependency only
    on the first request made that fallback ineffective, so resolve it eagerly.
    """
    module = importlib.import_module("llm_client")
    acall_llm = getattr(module, "acall_llm", None)
    if acall_llm is None:
        raise ImportError("llm_client is installed but does not expose acall_llm")
    return acall_llm


class LLMClientAdapter(BaseLLM):
    """Adapter that makes ``llm_client`` compatible with BaseLLM operators."""

    def __init__(self, model: str, **llm_client_kwargs: Any):
        # Fail here, not on the first user query, so MCP initialization can use
        # its existing ImportError fallback to the default LLM provider.
        self._acall_llm = _load_llm_client_acall()

        self.config = LLMConfig(
            api_type=LLMType.LITELLM,
            model=model,
            api_key="managed-by-llm-client",
        )
        self.model = model
        self._kwargs = dict(llm_client_kwargs)
        self.semaphore = asyncio.Semaphore(
            self._kwargs.pop("max_concurrency", 5)
        )
        self.cost_manager = None
        self.use_system_prompt = True
        self.system_prompt = "You are a helpful assistant."
        self.pricing_plan = model
        self.aclient = None

        logger.info(f"LLMClientAdapter initialized for model: {model}")

    def _call_kwargs(
        self,
        timeout: int,
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        kwargs = dict(self._kwargs)
        if timeout is not None:
            kwargs["timeout"] = timeout
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return kwargs

    async def _achat_completion(
        self,
        messages: list[dict],
        timeout: int = 60,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call llm_client and return an OpenAI-compatible response dict."""
        result = await self._acall_llm(
            self.model,
            messages,
            **self._call_kwargs(timeout, max_tokens),
        )

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": result.finish_reason or "stop",
                }
            ],
            "usage": result.usage,
            "model": result.model,
        }

    async def acompletion_text(
        self,
        messages: list[dict],
        stream: bool = False,
        timeout: int = 60,
        max_tokens: Optional[int] = None,
        format: str = "text",
    ) -> str:
        """Return a string response via llm_client. Called by BaseLLM.aask()."""
        if stream:
            raise NotImplementedError("Use non-streaming for operator calls")

        result = await self._acall_llm(
            self.model,
            messages,
            **self._call_kwargs(timeout, max_tokens),
        )
        return result.content

    async def acompletion(
        self,
        messages: list[dict],
        timeout: int = 60,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await self._achat_completion(
            messages, timeout=timeout, max_tokens=max_tokens, **kwargs
        )

    async def _achat_completion_stream(
        self,
        messages: list[dict],
        timeout: int = 60,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("Use non-streaming for operator calls")
