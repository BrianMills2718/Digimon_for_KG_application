# START: /home/brian/digimon/Core/Provider/LiteLLMProvider.py
import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Type, AsyncGenerator

from pydantic import BaseModel
import litellm
litellm.drop_params = True  # Drop unsupported params for O-series and other models
import instructor
from openai.types.chat import ChatCompletionChunk
from openai.types import CompletionUsage

from Core.Provider.BaseLLM import BaseLLM
from Config.LLMConfig import LLMConfig, LLMType # LLMType is imported here
from Core.Common.Logger import logger, log_llm_stream
from Core.Common.CostManager import CostManager
from Core.Common.Constants import USE_CONFIG_TIMEOUT
from Core.Utils.TokenCounter import count_input_tokens, count_output_tokens
from Core.Common.RetryUtils import retry_llm_call
from Core.Common.CacheManager import cached, get_cache_manager

# This is the import for the decorator functionality
from Core.Provider.LLMProviderRegister import register_provider # Ensure this is clean
from Config.LLMConfig import LLMType


@register_provider(LLMType.LITELLM)
class LiteLLMProvider(BaseLLM):
    def __init__(self, config: LLMConfig):
        self.config: LLMConfig = config
        self.model: str = config.model # This should be a LiteLLM model string, e.g., "openai/gpt-4o"
        self.api_key: Optional[str] = config.api_key if config.api_key and config.api_key != "sk-" and "YOUR_API_KEY" not in config.api_key else None
        self.base_url: Optional[str] = config.base_url if config.base_url else None
        self.temperature: float = getattr(config, "temperature", 0.0)  # Added for compatibility
        self.max_token: int = getattr(config, "max_token", 2000)      # Added for compatibility
        self.max_tokens = self.max_token  # Alias for compatibility with agent_brain
        self.cost_manager: Optional[CostManager] = CostManager() if self.config.calc_usage else None #
        self.pricing_plan = self.config.pricing_plan or self.model # For cost calculation
        # Add semaphore for async request throttling (missing from BaseLLM implementation)
        self.semaphore = asyncio.Semaphore(config.concurrent_requests if hasattr(config, 'concurrent_requests') else 5)

        # For instructor, create clients once
        self._instructor_client_sync = instructor.from_litellm(litellm.completion)
        self._instructor_client_async = instructor.from_litellm(litellm.acompletion)

        logger.info(f"LiteLLMProvider initialized for model: {self.model}")
        if not self.api_key:
            logger.info(f"No specific API key provided for {self.model} in LLMConfig; LiteLLM will rely on environment variables.")
        # Note: BaseLLM might also set self.aclient, but for LiteLLM it's not a persistent client object.
        # self.aclient is more for openai SDK style. We don't need it for LiteLLM.

    def _prepare_litellm_kwargs(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Dict[str, Any]:
        """Prepares common arguments for LiteLLM calls."""
        # Calculate dynamic max_tokens if not explicitly provided
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is None:
            # Use get_max_completion_tokens to dynamically calculate available tokens
            from Core.Utils.TokenCounter import get_max_completion_tokens
            max_tokens = get_max_completion_tokens(messages, self.model, self.config.max_token)
        
        params = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": max_tokens,
            "api_key": self.api_key, # Pass if available, LiteLLM handles None and uses env vars
            "base_url": self.base_url, # Pass if available
            "timeout": self.get_timeout(kwargs.get("timeout", USE_CONFIG_TIMEOUT)),
        }
        # Add other relevant params from self.config if LiteLLM supports them directly
        if self.config.top_p != 1.0: # Default for top_p is often 1.0
            params["top_p"] = self.config.top_p
        if self.config.stop:
            params["stop"] = self.config.stop
        if self.config.presence_penalty != 0.0:
             params["presence_penalty"] = self.config.presence_penalty
        if self.config.frequency_penalty != 0.0:
             params["frequency_penalty"] = self.config.frequency_penalty

        # Filter out None values for optional LiteLLM params like api_key, base_url
        return {k: v for k, v in params.items() if v is not None}

    @retry_llm_call(max_attempts=3)
    async def _achat_completion(self, messages: list[dict], timeout: Optional[int] = None, max_tokens: Optional[int] = None, **kwargs) -> litellm.ModelResponse:
        """
        Core asynchronous completion call using LiteLLM.
        Returns the raw LiteLLM ModelResponse object.
        """
        litellm_kwargs = self._prepare_litellm_kwargs(
            messages, 
            stream=False, 
            timeout=timeout, 
            max_tokens=max_tokens,
            **kwargs
        )
        logger.debug(f"LiteLLMProvider: Calling litellm.acompletion with kwargs: { {k:v for k,v in litellm_kwargs.items() if k != 'messages'} }")
        
        original_gemini_key_env = None
        original_google_key_env = None
        is_gemini_model = "gemini/" in self.model

        if is_gemini_model:
            original_gemini_key_env = os.environ.get('GEMINI_API_KEY')
            original_google_key_env = os.environ.get('GOOGLE_API_KEY')
            if self.api_key:
                os.environ['GEMINI_API_KEY'] = self.api_key
                os.environ['GOOGLE_API_KEY'] = self.api_key
                logger.info(f"Temporarily set os.environ['GEMINI_API_KEY'] and os.environ['GOOGLE_API_KEY'] for Gemini model: {self.model}")

        try:
            response: litellm.ModelResponse = await litellm.acompletion(**litellm_kwargs)
        finally:
            if is_gemini_model and self.api_key:
                if original_gemini_key_env:
                    os.environ['GEMINI_API_KEY'] = original_gemini_key_env
                elif 'GEMINI_API_KEY' in os.environ:
                    del os.environ['GEMINI_API_KEY']
                if original_google_key_env:
                    os.environ['GOOGLE_API_KEY'] = original_google_key_env
                elif 'GOOGLE_API_KEY' in os.environ:
                    del os.environ['GOOGLE_API_KEY']
                logger.info("Restored original GEMINI_API_KEY/GOOGLE_API_KEY environment variables if they existed.")

        if response.usage:
            self._update_costs(response.usage) #
        elif self.config.calc_usage: # if usage not in response, try to calculate it
            # For some models, usage might not be directly available in response
            # We need the response text to calculate output tokens
            response_text = response.choices[0].message.content if response.choices and response.choices[0].message else ""
            calculated_usage = CompletionUsage( #
                prompt_tokens=count_input_tokens(messages, self.pricing_plan), #
                completion_tokens=count_output_tokens(response_text, self.pricing_plan), #
                total_tokens=0 # This will be sum in _update_costs
            )
            # Manually set total_tokens for _update_costs if it expects it
            calculated_usage.total_tokens = calculated_usage.prompt_tokens + calculated_usage.completion_tokens
            self._update_costs(calculated_usage) #
        return response

    @retry_llm_call(max_attempts=3)
    async def _achat_completion_stream(self, messages: list[dict], timeout: Optional[int] = None, max_tokens: Optional[int] = None, **kwargs) -> AsyncGenerator[str, None]:
        """
        Core asynchronous streaming completion call using LiteLLM.
        Yields text chunks.
        """
        litellm_kwargs = self._prepare_litellm_kwargs(
            messages, 
            stream=True, 
            timeout=timeout, 
            max_tokens=max_tokens,
            **kwargs
        )
        logger.debug(f"LiteLLMProvider: Calling litellm.acompletion (stream=True) with kwargs: { {k:v for k,v in litellm_kwargs.items() if k != 'messages'} }")
        
        original_gemini_key_env = None
        original_google_key_env = None
        is_gemini_model = "gemini/" in self.model

        if is_gemini_model:
            original_gemini_key_env = os.environ.get('GEMINI_API_KEY')
            original_google_key_env = os.environ.get('GOOGLE_API_KEY')
            if self.api_key:
                os.environ['GEMINI_API_KEY'] = self.api_key
                os.environ['GOOGLE_API_KEY'] = self.api_key
                logger.info(f"Temporarily set os.environ['GEMINI_API_KEY'] and os.environ['GOOGLE_API_KEY'] for Gemini streaming: {self.model}")

        collected_content = []
        try:
            response_stream = await litellm.acompletion(**litellm_kwargs)
            async for chunk in response_stream:
                text_chunk = chunk.choices[0].delta.content or ""
                log_llm_stream(text_chunk) #
                collected_content.append(text_chunk)
                yield text_chunk
            log_llm_stream("\n")
        finally:
            if is_gemini_model and self.api_key:
                if original_gemini_key_env:
                    os.environ['GEMINI_API_KEY'] = original_gemini_key_env
                elif 'GEMINI_API_KEY' in os.environ:
                    del os.environ['GEMINI_API_KEY']
                if original_google_key_env:
                    os.environ['GOOGLE_API_KEY'] = original_google_key_env
                elif 'GOOGLE_API_KEY' in os.environ:
                    del os.environ['GOOGLE_API_KEY']
                logger.info("Restored original GEMINI_API_KEY/GOOGLE_API_KEY environment variables after streaming for Gemini.")

        if self.config.calc_usage:
            full_response_text = "".join(collected_content)
            # LiteLLM might provide usage data in the last chunk or after stream for some models.
            # For now, we'll calculate manually after collecting the stream.
            # Proper usage tracking with streams might require a custom callback handler in LiteLLM.
            # If the underlying response is available (e.g., response._raw_response), we could get usage.
            calculated_usage = CompletionUsage( #
                prompt_tokens=count_input_tokens(messages, self.pricing_plan), #
                completion_tokens=count_output_tokens(full_response_text, self.pricing_plan), #
                total_tokens=0
            )
            calculated_usage.total_tokens = calculated_usage.prompt_tokens + calculated_usage.completion_tokens
            self._update_costs(calculated_usage) #
    
    # This method is part of the BaseLLM interface.
    # It uses _achat_completion implemented above.
    async def acompletion(self, messages: list[dict], timeout: Optional[int] = None, max_tokens: Optional[int] = None, **kwargs) -> litellm.ModelResponse: # Type hint matches _achat_completion
        return await self._achat_completion(messages, timeout=timeout, max_tokens=max_tokens, **kwargs)

    # get_choice_text is inherited from BaseLLM and should work if _achat_completion returns a LiteLLM ModelResponse,
    # as ModelResponse.choices[0].message.content is compatible with OpenAI's ChatCompletion structure.
    # def get_choice_text(self, rsp: litellm.ModelResponse) -> str: # rsp should be LiteLLM's ModelResponse
    #     return rsp.choices[0].message.content if rsp.choices and rsp.choices[0].message else ""

    async def async_instructor_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        max_retries: int = 2,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Optional[BaseModel]:
        """
        Performs an asynchronous LLM call using LiteLLM and parses the response
        directly into the provided Pydantic response_model using Instructor.
        """
        call_params = {
            "model": self.model,
            "messages": messages,
            "response_model": response_model,
            "max_retries": max_retries,
            "api_key": self.api_key if self.api_key and self.api_key != "sk-" else None,
            "base_url": self.base_url if self.base_url else None,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        # Add any additional kwargs passed to the method
        call_params.update(kwargs)
        call_params = {k: v for k, v in call_params.items() if v is not None} # Filter Nones
        
        logger.debug(f"LiteLLMProvider: Calling instructor_client_async.chat.completions.create with response_model: {response_model.__name__}, model: {self.model}")

        original_gemini_key_env = None
        original_google_key_env = None
        is_gemini_model = "gemini/" in self.model

        if is_gemini_model:
            original_gemini_key_env = os.environ.get('GEMINI_API_KEY')
            original_google_key_env = os.environ.get('GOOGLE_API_KEY')
            if self.api_key:
                os.environ['GEMINI_API_KEY'] = self.api_key
                os.environ['GOOGLE_API_KEY'] = self.api_key
                logger.info(f"Temporarily set os.environ['GEMINI_API_KEY'] and os.environ['GOOGLE_API_KEY'] for Gemini instructor: {self.model}")

        structured_response = None
        try:
            # Use the async client for instructor
            structured_response = await self._instructor_client_async.chat.completions.create(**call_params)
        finally:
            if is_gemini_model and self.api_key:
                if original_gemini_key_env:
                    os.environ['GEMINI_API_KEY'] = original_gemini_key_env
                elif 'GEMINI_API_KEY' in os.environ:
                    del os.environ['GEMINI_API_KEY']
                if original_google_key_env:
                    os.environ['GOOGLE_API_KEY'] = original_google_key_env
                elif 'GOOGLE_API_KEY' in os.environ:
                    del os.environ['GOOGLE_API_KEY']
                logger.info("Restored original GEMINI_API_KEY/GOOGLE_API_KEY environment variables after instructor call for Gemini.")

        if structured_response:
            raw_litellm_response = getattr(structured_response, '_raw_response', None)
            if raw_litellm_response and raw_litellm_response.usage:
                self._update_costs(raw_litellm_response.usage) #
            elif self.config.calc_usage:
                logger.warning("Usage data not directly available from instructor response. Manual calculation for instructor calls is not yet implemented in LiteLLMProvider.")
                # To implement, you'd need the prompt and the serialized response text.

        return structured_response
# END: /home/brian/digimon/Core/Provider/LiteLLMProvider.py