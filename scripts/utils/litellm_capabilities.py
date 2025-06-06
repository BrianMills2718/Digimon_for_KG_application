import json
import sys
import requests
import base64
import time
import os
import litellm
from litellm import responses, get_model_info, batch_completion, batch_completion_models, batch_completion_models_all_responses, create_assistants, get_assistants, delete_assistant, create_thread, get_thread, add_message, get_messages, run_thread, embedding, rerank, Router # get_model_info for prefix test, added embedding, added rerank, added Router
from litellm.utils import supports_pdf_input
from litellm.caching.caching import Cache  # Correct import for caching
import openai # For catching specific OpenAI-compatible exceptions
import instructor # Ensure instructor is installed: pip install instructor
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, List
import asyncio

# Set encoding to handle unicode characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure these are installed: pip install langchain-community langchain-core
try:
    from langchain_community.chat_models import ChatLiteLLM
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("WARNING: Langchain libraries not found. Skipping Langchain ChatLiteLLM test. Install with: pip install langchain-community langchain-core")

# New way to enable LiteLLM debug logging
# Set this early, before other litellm imports if they trigger config or logging.
os.environ['LITELLM_LOG'] = 'DEBUG' 

# Load environment variables from .env file (e.g., API keys)
# This allows for secure management of API keys without hardcoding them.
# Ensure your .env file has:
# OPENAI_API_KEY=your_openai_key
# GOOGLE_API_KEY=your_google_ai_studio_or_vertex_key (or GEMINI_API_KEY)
# ANTHROPIC_API_KEY=your_anthropic_key
# DEEPSEEK_API_KEY=your_deepseek_key
load_dotenv() # Loads variables from .env into os.environ

# --- API Key Constants (loaded from .env) ---
# These constants store API keys retrieved from environment variables.
# The script will check for these and skip tests if a required key is missing.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# LiteLLM often uses GOOGLE_API_KEY for Gemini, but we can define GEMINI_API_KEY as well.
# It will pick up GOOGLE_API_KEY if GEMINI_API_KEY is not explicitly set for a gemini call.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# --- Configuration ---
# Defines model names and other constants used throughout the script.
# Using constants makes it easier to update model versions or identifiers.
# OpenAI
OPENAI_GPT4O_MODEL = "gpt-4o"
O4_MINI_MODEL_NAME = "o4-mini" # LiteLLM treats this like an OpenAI model
OPENAI_WEB_SEARCH_COMPLETION_MODEL = "openai/gpt-4o-search-preview" # As per docs

# Gemini
GEMINI_MODEL_NAME_PRO = "gemini/gemini-1.5-pro" 
GEMINI_MODEL_NAME_FLASH = "gemini/gemini-1.5-flash"

# Anthropic
ANTHROPIC_SONNET_MODEL = "claude-3-sonnet-20240229"

# DeepSeek
DEEPSEEK_CHAT_MODEL = "deepseek/deepseek-chat" # Correctly prefixed for LiteLLM

# --- Pydantic Models for Function Calling / Tool Use Tests ---
# These Pydantic models define the expected structure for data extraction
# when testing function calling (tool use) capabilities with `instructor`.
class SimpleUserDetails(BaseModel):
    name: str = Field(..., description="The full name of the user.")
    age: int = Field(..., description="The age of the user.")
    city: Optional[str] = Field(None, description="The city where the user resides, if known.")
    # Note: `Optional[str]` for a data field like 'city' was working fine with Gemini.
    # The `anyOf` issue we debugged previously was for `Optional[str]` fields that *were themselves descriptions*
    # in more complex Pydantic models, leading to problematic schema generation for Gemini's tool parser.
    # For this simple data model, `Optional[str]` for `city` is standard and should be fine.

# Simpler model for Gemini to avoid any_of schema issues
class SimpleUserDetailsGemini(BaseModel):
    name: str = Field(..., description="The full name of the user.")
    age: int = Field(..., description="The age of the user.")
    city: str = Field(..., description="The city where the user resides.")

# --- Helper Function for Pydantic/Instructor based Function Calling / Tool Use Test ---
# This function demonstrates how to use LiteLLM with the `instructor` library
# to get structured output (Pydantic models) from LLMs that support tool use/function calling.
# It shows different strategies for client initialization (LiteLLM default, native OpenAI, native Anthropic).
def run_function_calling_test(model_to_test: str, client_provider_strategy: str = "litellm"): #
    print(f"\n--- Testing Function Calling with {model_to_test} (using strategy: {client_provider_strategy}) ---")
    
    # Basic API Key availability checks
    key_found = True
    if "gemini" in model_to_test.lower(): #
        if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
            print(f"INFO: Set GOOGLE_API_KEY from GEMINI_API_KEY for {model_to_test}.") #
        elif not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            key_found = False
    elif "claude" in model_to_test.lower() and not os.getenv("ANTHROPIC_API_KEY"): #
        key_found = False
    elif (model_to_test.startswith("gpt") or model_to_test == O4_MINI_MODEL_NAME) and not os.getenv("OPENAI_API_KEY"):
        key_found = False
    elif model_to_test.startswith("deepseek") and not os.getenv("DEEPSEEK_API_KEY"):
        key_found = False
    
    if not key_found:
        print(f"WARNING: Appropriate API key for {model_to_test} not found. Skipping function calling test.")
        return

    iclient = None
    actual_client_used_for_call = "litellm_default" # For logging which path was taken

    try:
        if client_provider_strategy == "instructor_openai" and (model_to_test.startswith("gpt") or model_to_test == O4_MINI_MODEL_NAME): #
            try:
                from openai import OpenAI as OpenAISDKClient # Standard OpenAI SDK
                sdk_client = OpenAISDKClient() 
                iclient = instructor.from_openai(sdk_client) #
                actual_client_used_for_call = "native_openai_via_instructor" #
                print(f"INFO: Using instructor with native OpenAI client for {model_to_test}")
            except ImportError:
                print(f"INFO: OpenAI SDK not found for direct instructor use. Falling back to LiteLLM for {model_to_test}.") #
                iclient = instructor.from_litellm(litellm.completion) #
                actual_client_used_for_call = "litellm_fallback_for_openai" #


        elif client_provider_strategy == "instructor_anthropic" and "claude" in model_to_test.lower(): #
            try:
                # Requires: pip install "instructor[anthropic]" anthropic
                import instructor.anthropic as instructor_anthropic_mod 
                from anthropic import Anthropic as AnthropicSDKClient
                sdk_client = AnthropicSDKClient() 
                iclient = instructor.from_anthropic(sdk_client.messages) # Use .messages for Claude 3+
                actual_client_used_for_call = "native_anthropic_via_instructor" #
                print(f"INFO: Using instructor with native Anthropic client for {model_to_test}")
            except ImportError:
                print(f"INFO: Native Anthropic client dependencies not found for instructor. Falling back to LiteLLM for {model_to_test}.") #
                iclient = instructor.from_litellm(litellm.completion) #
                actual_client_used_for_call = "litellm_fallback_for_anthropic" #
        
        if iclient is None: # Default for Gemini, DeepSeek, or fallbacks
            iclient = instructor.from_litellm(litellm.completion) #
            actual_client_used_for_call = "litellm_via_instructor" #
            print(f"INFO: Using instructor with LiteLLM client for {model_to_test}")


        prompt_messages = [
            {"role": "user", "content": "Extract the user's name, age and city. The user is Alex Taylor, 35 years old, and lives in Berlin."} #
        ]

        print(f"Attempting call to {model_to_test} with instructor (actual client path: {actual_client_used_for_call})...") #
        
        response_object = None
        # Native instructor clients for OpenAI/Anthropic have a slightly different API surface
        if actual_client_used_for_call == "native_anthropic_via_instructor": #
             tool_schema = SimpleUserDetails.model_json_schema()
             response_object = iclient.create( # type: ignore #
                model=model_to_test.split('/')[-1] if '/' in model_to_test else model_to_test, 
                max_tokens=1024, # Anthropic requires max_tokens
                messages=prompt_messages,
                tools=[{ 
                    "name": tool_schema.get("title", "SimpleUserDetails"),
                    "description": tool_schema.get("description", "Extracts user details like name, age, and city."), #
                    "input_schema": tool_schema
                }],
                # tool_choice={"type": "tool", "name": tool_schema.get("title", "SimpleUserDetails")}, # Let instructor handle tool choice from response_model
                response_model=SimpleUserDetails
            )
        elif iclient: # For OpenAI native, and all LiteLLM paths
            # Choose the appropriate model based on provider to avoid schema issues
            if "gemini" in model_to_test.lower():
                response_model = SimpleUserDetailsGemini
            else:
                response_model = SimpleUserDetails
                
            response_object = iclient.chat.completions.create(
                model=model_to_test,
                messages=prompt_messages,
                response_model=response_model,
                max_retries=1, 
            )
        else:
            print(f"ERROR: Instructor client (iclient) was not properly initialized for {model_to_test}.")
            return

        print(f"\n{model_to_test} - Instructor Response (Pydantic Object):")
        print(response_object)
        if response_object:
             print(f"  User Name: {response_object.name}, Age: {response_object.age}, City: {response_object.city}")

    except ImportError as e:
        print(f"{model_to_test} - ImportError during client setup or call: {e}")
    except Exception as e:
        print(f"{model_to_test} - An error occurred during function calling test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for JSON Mode Test ---
# This function tests the JSON mode capability of LLMs, where the model is instructed
# to return its output in a strict JSON format.
# LiteLLM enables this using `response_format={'type': 'json_object'}`.
def run_json_mode_test(model_to_test: str): #
    print(f"\n--- Testing JSON Mode with {model_to_test} ---")
    
    api_key_needed = True
    # Basic API Key availability checks
    if "gemini" in model_to_test.lower(): #
        if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        elif not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            api_key_needed = False
    elif "claude" in model_to_test.lower() and not os.getenv("ANTHROPIC_API_KEY"): #
        api_key_needed = False
    elif (model_to_test.startswith("gpt") or model_to_test == O4_MINI_MODEL_NAME) and not os.getenv("OPENAI_API_KEY"):
        api_key_needed = False
    elif model_to_test.startswith("deepseek") and not os.getenv("DEEPSEEK_API_KEY"):
        api_key_needed = False

    if not api_key_needed:
        print(f"WARNING: Appropriate API key for {model_to_test} not found for JSON mode test. Skipping.")
        return

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant. Your response MUST be a single, valid JSON object and nothing else. Do not include explanations or markdown ```json wrappers."}, #
        {"role": "user", "content": "Provide server health: status is 'green', uptime is 72 hours, region is 'us-west-1'. Format as JSON."} #
    ]

    try:
        print(f"Attempting call to {model_to_test} for JSON output...")
        
        api_params = {
            "model": model_to_test,
            "messages": prompt_messages,
            "response_format": {"type": "json_object"} # Request JSON mode
        }
        # Adjust temperature: o4-mini uses default; others can use a low temp for predictability
        if model_to_test != O4_MINI_MODEL_NAME: #
            api_params["temperature"] = 0.1 
            
        response = litellm.completion(**api_params)

        print(f"\n{model_to_test} - Raw LiteLLM Response Object (ModelResponse):")
        # print(response) # Can be very verbose

        raw_content = None
        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            raw_content = response.choices[0].message.content
            print(f"\n{model_to_test} - Raw Content String from Model:\n```\n{raw_content}\n```")
        else:
            print(f"\n{model_to_test} - No content found in response.")
            if response: print(f"Full response object was: {response}")
            return
        
        content_to_parse = raw_content.strip()
        # Basic stripping for markdown ```json ... ``` or ``` ... ```
        # Some models might still add it despite instructions.
        if content_to_parse.startswith("```json"): #
            content_to_parse = content_to_parse[len("```json"):].strip() #
            if content_to_parse.endswith("```"):
                content_to_parse = content_to_parse[:-len("```")].strip() #
        elif content_to_parse.startswith("```") and content_to_parse.endswith("```"): #
             content_to_parse = content_to_parse[len("```"):(len(content_to_parse) - len("```"))].strip() #
        
        try:
            parsed_json = json.loads(content_to_parse)
            print(f"\n{model_to_test} - Successfully Parsed JSON:")
            print(json.dumps(parsed_json, indent=2))
        except json.JSONDecodeError as e:
            print(f"\n{model_to_test} - FAILED to parse JSON from raw content. Error: {e}")
            print(f"Content that failed to parse was (after stripping):\n```\n{content_to_parse}\n```")

    except Exception as e:
        print(f"{model_to_test} - An error occurred during JSON mode test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Reasoning Content Test ---
# This function is designed to test if the model can follow instructions
# to provide reasoning steps before its final answer, often useful for complex queries.
# It checks for specific keywords in the response that indicate reasoning.
def run_reasoning_content_test(model_to_test: str):
    print(f"\n--- Testing Reasoning Content with {model_to_test} ---")
    
    # Check if this model supports reasoning content (thinking mode)
    if "claude" in model_to_test.lower():
        print(f"INFO: {model_to_test} does not support reasoning content (thinking mode). Skipping test.")
        return
    
    # API key checks for supported models  
    api_key_needed = True
    if model_to_test.startswith("deepseek") and os.getenv("DEEPSEEK_API_KEY"):
        api_key_needed = True
    elif (model_to_test.startswith("gpt") or model_to_test == O4_MINI_MODEL_NAME) and os.getenv("OPENAI_API_KEY"): # Though OpenAI reasoning may be limited
        api_key_needed = True  
    else:
        api_key_needed = False

    if not api_key_needed:
        print(f"WARNING: Appropriate API key for {model_to_test} not found for reasoning content test. Skipping.")
        return

    prompt_messages = [
        {"role": "user", "content": "What is the capital of France? And what is 2+2?"}
    ]

    try:
        print(f"Attempting call to {model_to_test} with reasoning_effort='low'...")
        
        # According to docs, use drop_params=True when using reasoning_effort for compatibility
        # between Anthropic and Deepseek.
        response = litellm.completion(
            model=model_to_test,
            messages=prompt_messages,
            reasoning_effort="low",
            drop_params=True
        )

        print(f"\n{model_to_test} - Raw LiteLLM Response Object (ModelResponse):")
        # print(response) # Can be verbose

        message = response.choices[0].message

        if message.content:
            print(f"\n{model_to_test} - Message Content:\n```\n{message.content}\n```")
        else:
            print(f"\n{model_to_test} - No direct message content found.")

        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"\n{model_to_test} - Reasoning Content:\n```\n{message.reasoning_content}\n```")
        else:
            print(f"\n{model_to_test} - No reasoning_content found in response message.")
            # Optionally, add a check here if the model is Anthropic to see if it *should* have had them.
            # For now, just reporting absence is fine.

        if hasattr(message, 'thinking_blocks') and message.thinking_blocks:
            print(f"\n{model_to_test} - Thinking Blocks (typically Anthropic-specific):\n```\n{json.dumps(message.thinking_blocks, indent=2)}\n```")
        else:
            print(f"\n{model_to_test} - No thinking_blocks found in response message.")
            # Optionally, add a check here if the model is Anthropic to see if it *should* have had them.
            # For now, just reporting absence is fine.

    except Exception as e:
        print(f"{model_to_test} - An error occurred during reasoning content test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Predicted Outputs Test ---
# This function tests if the model's output matches a predicted or expected output
# for a given input. This is a basic form of response validation.
def run_predicted_outputs_test(model_to_test: str):
    print(f"\n--- Testing Predicted Outputs with {model_to_test} ---")

    # This feature is documented primarily for OpenAI models.
    if not (model_to_test.startswith("gpt") or model_to_test == O4_MINI_MODEL_NAME or "openai" in model_to_test):
        print(f"INFO: Model {model_to_test} is not an OpenAI model. Skipping predicted outputs test as it's primarily documented for OpenAI.")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print(f"WARNING: OPENAI_API_KEY not found. Skipping predicted outputs test for {model_to_test}.")
        return

    original_code = """
/// <summary>
/// Represents a user with a first name, last name, and username.
/// </summary>
public class User
{
    /// <summary>
    /// Gets or sets the user's first name.
    /// </summary>
    public string FirstName { get; set; }

    /// <summary>
    /// Gets or sets the user's last name.
    /// </summary>
    public string LastName { get; set; }

    /// <summary>
    /// Gets or sets the user's username.
    /// </summary>
    public string Username { get; set; }
}
"""

    prompt_messages_for_api = [
        {
            "role": "user",
            "content": f"Given the following C# code, replace the Username property with an Email property. Respond only with the complete modified C# code, and with no markdown formatting or explanations.\n\nOriginal Code:\n```csharp\n{original_code}\n```",
        }
    ]

    try:
        print(f"Attempting call to {model_to_test} with predicted output...")
        
        completion_response = litellm.completion(
            model=model_to_test,
            messages=prompt_messages_for_api,
            prediction={"type": "content", "content": original_code},
            temperature=1.0 if model_to_test == O4_MINI_MODEL_NAME else 0.0, # O4-mini only supports temperature=1
            max_tokens=1000 # Ensure enough space for the modified code
        )

        print(f"\n{model_to_test} - Raw LiteLLM Response Object (ModelResponse):")
        # print(completion_response) # Can be verbose

        response_content = completion_response.choices[0].message.content

        if response_content:
            print(f"\n{model_to_test} - Message Content (should be modified code):\n```csharp\n{response_content.strip()}\n```")
            if "Username" in response_content and "Email" in response_content:
                print("INFO: Both 'Username' and 'Email' are in the response. Please check if modification was partial or if 'Username' remains in comments.")
            elif "Email" in response_content and "Username" not in response_content: # A simplistic check
                print("INFO: 'Email' property seems to be present and 'Username' (as a property) seems to be removed. Verification needed.")
            elif "Username" in response_content:
                 print("WARNING: 'Username' property might still be present. Modification might not have occurred as expected.")
            else:
                print("INFO: Response received. Manual verification of modification advised.")
        else:
            print(f"\n{model_to_test} - No direct message content found.")

    except Exception as e:
        print(f"{model_to_test} - An error occurred during predicted outputs test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Retries Test ---
# This function demonstrates LiteLLM's automatic retry mechanism.
# By setting `num_retries`, LiteLLM will attempt to call the API multiple times
# if transient errors occur (e.g., rate limits, temporary connection issues).
def run_retries_test(model_to_test: str):
    print(f"\n--- Testing Retries with {model_to_test} (num_retries=2) ---")

    api_key_needed = True
    if "gemini" in model_to_test.lower():
        if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        elif not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            api_key_needed = False
    elif "claude" in model_to_test.lower() and not os.getenv("ANTHROPIC_API_KEY"):
        api_key_needed = False
    elif (model_to_test.startswith("gpt") or model_to_test == O4_MINI_MODEL_NAME) and not os.getenv("OPENAI_API_KEY"):
        api_key_needed = False
    elif model_to_test.startswith("deepseek") and not os.getenv("DEEPSEEK_API_KEY"):
        api_key_needed = False

    if not api_key_needed:
        print(f"WARNING: Appropriate API key for {model_to_test} not found. Skipping retries test.")
        return

    messages = [{"role": "user", "content": "Tell me a very short story."}]

    try:
        print(f"Attempting call to {model_to_test} with num_retries=2...")
        # We can't easily simulate a failure that would trigger a retry without a flaky connection
        # or by mocking. This test primarily demonstrates setting the parameter.
        # LiteLLM's internal retry mechanism (e.g. with Tenacity) would handle transient network errors.
        response = litellm.completion(
            model=model_to_test,
            messages=messages,
            num_retries=2
        )
        print(f"{model_to_test} - Response received (retries parameter was set).")
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            print(f"Content sample: {response.choices[0].message.content[:100]}...")
        else:
            print("No content in response.")

    except Exception as e:
        print(f"{model_to_test} - An error occurred during retries test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Model Fallback Test ---
# This function tests LiteLLM's model fallback capability at the SDK level.
# If the primary model call fails (e.g., model doesn't exist, API error),
# LiteLLM can automatically try a list of specified `fallback_models`.
# This test intentionally uses a non-existent primary model to trigger the fallback.
def run_model_fallback_test(primary_model_to_fail: str, fallback_models: List[str]):
    print(f"\n--- Testing Model Fallback: Primary '{primary_model_to_fail}', Fallbacks {fallback_models} ---")

    # Ensure at least one fallback model has its API key available
    key_for_fallback_found = False
    for fb_model in fallback_models:
        if "gemini" in fb_model.lower():
            if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
                key_for_fallback_found = True; break
        elif "claude" in fb_model.lower() and os.getenv("ANTHROPIC_API_KEY"):
            key_for_fallback_found = True; break
        elif (fb_model.startswith("gpt") or fb_model == O4_MINI_MODEL_NAME) and os.getenv("OPENAI_API_KEY"):
            key_for_fallback_found = True; break
        elif fb_model.startswith("deepseek") and os.getenv("DEEPSEEK_API_KEY"):
            key_for_fallback_found = True; break
    
    if not key_for_fallback_found:
        print(f"WARNING: No API key found for any of the fallback models {fallback_models}. Skipping model fallback test.")
        return

    messages = [{"role": "user", "content": "What is the capital of Germany?"}]

    try:
        print(f"Attempting call with primary model (expected to fail): '{primary_model_to_fail}' and fallbacks: {fallback_models}...")
        # LiteLLM is expected to print "Completion with 'bad-model': got exception..."
        # and then "making completion call actual_fallback_model"
        response = litellm.completion(
            model=primary_model_to_fail,
            messages=messages,
            fallbacks=fallback_models
        )
        
        print(f"\nFallback mechanism completed. Response from model: {response.model}")
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            print(f"  Content: {response.choices[0].message.content}")
        else:
            print("  No content in response from fallback.")

    except Exception as e:
        # This might catch an error if ALL fallbacks also fail, or if there's a setup issue.
        print(f"An error occurred during model fallback test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Web Search Test ---
# This function tests the web search capability, typically available with specific models
# like OpenAI's models when appropriate tools are enabled.
# It uses `litellm.responses()` for a more detailed look at the response object.
def run_web_search_test(model_to_test_completion: str, model_to_test_responses: str):
    print(f"\n--- Testing Web Search ---")

    # Test with litellm.completion
    print(f"\n- Using litellm.completion with model: {model_to_test_completion}")
    if not os.getenv("OPENAI_API_KEY"): # Web search is primarily OpenAI for now
        print(f"WARNING: OPENAI_API_KEY not found. Skipping web search test with litellm.completion.")
    elif not litellm.supports_web_search(model=model_to_test_completion):
        print(f"INFO: Model {model_to_test_completion} does not support web search according to litellm.supports_web_search(). Skipping.")
    else:
        try:
            print(f"Attempting web search with {model_to_test_completion}...")
            response = litellm.completion(
                model=model_to_test_completion,
                messages=[{"role": "user", "content": "What are some recent breakthroughs in AI ethics?"}],
                web_search_options={"search_context_size": "low"}
            )
            print(f"{model_to_test_completion} - Web Search (completion) Response:")
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                print(f"Content sample: {response.choices[0].message.content[:200]}...")
            else:
                print("No content in response.")
            # The documentation doesn't specify how to access search results themselves from completion,
            # just that the model uses them. So we just check for a coherent response.
        except Exception as e:
            print(f"{model_to_test_completion} - Error during web search (completion) test: {e}")
            import traceback
            traceback.print_exc()

    # Test with litellm.responses
    print(f"\n- Using litellm.responses with model: {model_to_test_responses}")
    if not os.getenv("OPENAI_API_KEY"): # Web search is primarily OpenAI for now
        print(f"WARNING: OPENAI_API_KEY not found. Skipping web search test with litellm.responses.")
    # litellm.responses doesn't have a direct supports_web_search check tied to the model string in the same way for tools,
    # it depends on the tool type specified.
    else:
        try:
            print(f"Attempting web search with {model_to_test_responses} via litellm.responses...")
            response_obj = litellm.responses( # Renamed to response_obj to avoid confusion
                model=model_to_test_responses,
                input=[{"role": "user", "content": "What was a positive news story from today?"}],
                tools=[{
                    "type": "web_search_preview",
                    "search_context_size": "low"
                }]
            )
            print(f"\n{model_to_test_responses} - Web Search (responses) Full Response Object:")
            print(response_obj) # Print the whole object to see its structure
            # print(vars(response_obj)) # Alternatively, for pydantic models vars() can be useful

            # Attempt to access content based on common patterns for LiteLLM responses
            # This path might need adjustment based on the actual structure of ResponsesAPIResponse
            message_content = None
            if hasattr(response_obj, 'choices') and response_obj.choices and \
               hasattr(response_obj.choices[0], 'message') and response_obj.choices[0].message and \
               hasattr(response_obj.choices[0].message, 'content'):
                message_content = response_obj.choices[0].message.content
            elif hasattr(response_obj, 'message') and response_obj.message and \
                 hasattr(response_obj.message, 'content'): # Simpler structure if it's not a list of choices
                message_content = response_obj.message.content
            elif isinstance(response_obj, dict) and response_obj.get('choices') and \
                 response_obj['choices'][0].get('message') and response_obj['choices'][0]['message'].get('content'): # if it's a dict
                message_content = response_obj['choices'][0]['message']['content']
            elif hasattr(response_obj, 'output') and response_obj.output: # ResponsesAPIResponse structure
                # Look for message type output with content
                for output_item in response_obj.output:
                    if hasattr(output_item, 'type') and output_item.type == 'message' and \
                       hasattr(output_item, 'content') and output_item.content:
                        # Content is a list of content items, find text type
                        for content_item in output_item.content:
                            if hasattr(content_item, 'type') and content_item.type == 'output_text' and \
                               hasattr(content_item, 'text'):
                                message_content = content_item.text
                                break
                        if message_content:
                            break
            if message_content:
                print(f"\n{model_to_test_responses} - Web Search (responses) Extracted Content:")
                print(f"Content sample: {message_content[:200]}...")
            else:
                print(f"\n{model_to_test_responses} - Could not directly extract message content from litellm.responses object. Please inspect the full object structure printed above.")

        except Exception as e:
            print(f"{model_to_test_responses} - Error during web search (responses) test: {e}")
            import traceback
            traceback.print_exc()

# --- Helper Function for Pre-fix Assistant Message Test ---
# This tests how models handle pre-filled assistant messages, which can be used
# to guide the model's response or continue a conversation in a specific way.
def run_prefix_assistant_message_test(model_to_test: str, api_key: Optional[str] = None):
    print(f"\n--- Testing Pre-fix Assistant Message with {model_to_test} ---")

    # Check if model supports assistant prefill
    try:
        model_info = get_model_info(model=model_to_test)
        if not model_info.get("supports_assistant_prefill"):
            print(f"INFO: Model {model_to_test} does not support assistant prefill according to get_model_info(). Skipping.")
            return
    except Exception as e:
        print(f"Could not get model_info for {model_to_test} to check for assistant prefill support: {e}. Trying to proceed anyway.")
        # Allow to proceed, the API call will fail if not supported or key is missing

    # Determine if an API key is conceptually required for this model provider
    is_anthropic_model = "claude" in model_to_test.lower() or "anthropic" in model_to_test.lower()
    is_deepseek_model = "deepseek" in model_to_test.lower()

    if (is_anthropic_model or is_deepseek_model) and not api_key:
        print(f"WARNING: API key not provided for {model_to_test}, which requires one. Skipping prefix assistant message test.")
        return

    messages = [
        {"role": "user", "content": "Who won the FIFA World Cup in 2022?"},
        {"role": "assistant", "content": "Argentina", "prefix": True}
    ]

    try:
        print(f"Attempting call to {model_to_test} with a prefixed assistant message...")
        response = litellm.completion(
            model=model_to_test,
            messages=messages,
            api_key=api_key # Pass the provided API key
        )
        print(f"{model_to_test} - Response to prefixed message:")
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            print(f"Content: '{response.choices[0].message.content}'")
            # Example specific check, can be made more generic
            if "Argentina" in response.choices[0].message.content or "won the FIFA World Cup in 2022" in response.choices[0].message.content:
                 print("INFO: Response seems related to the prefixed content.")
            else:
                print("INFO: Response received, manual check advised for prefix logic.")
        else:
            print("No content in response.")

    except Exception as e:
        print(f"{model_to_test} - An error occurred during prefix assistant message test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Batch Completion (Many Calls to 1 Model) ---
# Tests `litellm.batch_completion()`: sends multiple, independent requests to the *same* model efficiently.
def run_batch_completion_one_model_test(model_to_test: str):
    print(f"\n--- Testing Batch Completion (Many Calls to 1 Model) with {model_to_test} ---")

    # API key check (simplified for the model being tested)
    api_key_present = False
    if "claude" in model_to_test.lower() and os.getenv("ANTHROPIC_API_KEY"):
        api_key_present = True
    elif (model_to_test.startswith("gpt") or model_to_test == O4_MINI_MODEL_NAME) and os.getenv("OPENAI_API_KEY"):
        api_key_present = True
    # Add other providers if testing with them for batching

    if not api_key_present:
        print(f"WARNING: API key for {model_to_test} not found. Skipping batch_completion test.")
        return

    list_of_message_lists = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Why is the sky blue?"}]
    ]

    try:
        print(f"Attempting batch_completion with {model_to_test} for {len(list_of_message_lists)} calls...")
        responses_list = litellm.batch_completion(
            model=model_to_test,
            messages=list_of_message_lists
        )
        
        print(f"\n{model_to_test} - Batch Completion (1 model) Responses ({len(responses_list)} received):")
        for i, response_item in enumerate(responses_list):
            if isinstance(response_item, Exception):
                print(f"  Response {i+1}: ERROR - {response_item}")
            elif response_item.choices and response_item.choices[0].message and response_item.choices[0].message.content:
                print(f"  Response {i+1} Content: {response_item.choices[0].message.content[:100]}...")
            else:
                print(f"  Response {i+1}: No content or unexpected format.")
        assert len(responses_list) == len(list_of_message_lists)
    except Exception as e:
        print(f"{model_to_test} - An error occurred during batch_completion (1 model) test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Batch Completion (1 Call to Many Models, Fastest) ---
# Tests `litellm.batch_completion_models()`: sends the *same* request to *multiple different models* and returns the fastest successful response.
def run_batch_completion_many_models_fastest_test(models_to_test: List[str]):
    print(f"\n--- Testing Batch Completion (1 Call to Many Models, Fastest) with {models_to_test} ---")

    # Ensure at least one model has an API key
    any_key_found = any([
        (os.getenv("OPENAI_API_KEY") and any(m.startswith("gpt") or m == O4_MINI_MODEL_NAME for m in models_to_test)),
        (os.getenv("ANTHROPIC_API_KEY") and any("claude" in m.lower() for m in models_to_test)),
        (os.getenv("DEEPSEEK_API_KEY") and any(m.startswith("deepseek") for m in models_to_test)),
        (os.getenv("GOOGLE_API_KEY") and any(m.startswith("gemini") for m in models_to_test))
    ])
    if not any_key_found:
        print(f"WARNING: No API keys found for any of the specified models for batch fastest. Skipping.")
        return

    messages = [{"role": "user", "content": "Tell me a fun fact about space."}]

    try:
        print(f"Attempting batch_completion_models (fastest) with {models_to_test}...")
        fastest_response = litellm.batch_completion_models(
            models=models_to_test,
            messages=messages
        )
        
        print(f"\nBatch Completion (Many Models, Fastest) - Response from model: {fastest_response.model}")
        if fastest_response.choices and fastest_response.choices[0].message and fastest_response.choices[0].message.content:
            print(f"  Content: {fastest_response.choices[0].message.content[:150]}...")
        else:
            print("  No content or unexpected format in fastest response.")
    except Exception as e:
        print(f"An error occurred during batch_completion_models (fastest) test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Batch Completion (1 Call to Many Models, All Responses) ---
# Tests `litellm.batch_completion_models_all_responses()`: sends the *same* request to *multiple different models* and returns all responses.
def run_batch_completion_many_models_all_test(models_to_test: List[str]):
    print(f"\n--- Testing Batch Completion (1 Call to Many Models, All Responses) with {models_to_test} ---")

    any_key_found = any([
        (os.getenv("OPENAI_API_KEY") and any(m.startswith("gpt") or m == O4_MINI_MODEL_NAME for m in models_to_test)),
        (os.getenv("ANTHROPIC_API_KEY") and any("claude" in m.lower() for m in models_to_test)),
        (os.getenv("DEEPSEEK_API_KEY") and any(m.startswith("deepseek") for m in models_to_test)),
        (os.getenv("GOOGLE_API_KEY") and any(m.startswith("gemini") for m in models_to_test))
    ])
    if not any_key_found:
        print(f"WARNING: No API keys found for any of the specified models for batch all. Skipping.")
        return

    messages = [{"role": "user", "content": "What's a common use case for Python?"}]

    try:
        print(f"Attempting batch_completion_models_all_responses with {models_to_test}...")
        all_responses_list = litellm.batch_completion_models_all_responses(
            models=models_to_test,
            messages=messages
        )
        
        print(f"\nBatch Completion (Many Models, All Responses) - Received {len(all_responses_list)} responses:")
        for i, response_item in enumerate(all_responses_list):
            if isinstance(response_item, Exception):
                print(f"  Response {i+1}: ERROR - {response_item}")
            elif hasattr(response_item, 'model') and response_item.choices and response_item.choices[0].message and response_item.choices[0].message.content:
                print(f"  Response {i+1} from {response_item.model}: {response_item.choices[0].message.content[:100]}...")
            else:
                print(f"  Response {i+1}: No content or unexpected format. Object: {type(response_item)}")
        assert len(all_responses_list) <= len(models_to_test) # We expect responses for models that succeed
    except Exception as e:
        print(f"An error occurred during batch_completion_models_all_responses test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Langchain ChatLiteLLM Test ---
# Demonstrates how to use LiteLLM as a backend for Langchain's `ChatLiteLLM` interface.
# This allows leveraging LiteLLM's provider support within the Langchain ecosystem.
def run_langchain_chatlitellm_test(model_to_test: str):
    print(f"\n--- Testing Langchain ChatLiteLLM with {model_to_test} ---")

    if not LANGCHAIN_AVAILABLE:
        print("INFO: Langchain libraries not available. Skipping test.")
        return

    # API key check (simplified for the model being tested)
    api_key_present = False
    if (model_to_test.startswith("gpt") or "openai" in model_to_test or model_to_test == O4_MINI_MODEL_NAME) and os.getenv("OPENAI_API_KEY"):
        api_key_present = True
    # Add checks for other providers if you test ChatLiteLLM with them
    
    if not api_key_present:
        print(f"WARNING: OpenAI API key not found. Skipping Langchain ChatLiteLLM test with {model_to_test}.")
        return

    try:
        print(f"Initializing ChatLiteLLM with model: {model_to_test}")
        # The model string for ChatLiteLLM should be in the format LiteLLM expects,
        # e.g., "gpt-3.5-turbo" or "openai/gpt-4o" if you want to be explicit.
        # Using the constants like OPENAI_GPT4O_MODEL which are already prefixed is fine.
        chat_model = ChatLiteLLM(model=model_to_test, temperature=0.1)

        messages = [
            SystemMessage(content="You are a helpful AI assistant that provides concise answers."),
            HumanMessage(content="What is the main benefit of using Langchain?")
        ]
        
        print("Invoking ChatLiteLLM model...")
        response_message = chat_model.invoke(messages)

        print(f"\n{model_to_test} - Langchain ChatLiteLLM Response:")
        if hasattr(response_message, 'content'):
            print(f"  Content: {response_message.content}")
        else:
            print(f"  Response object: {response_message} (unexpected format)")

    except Exception as e:
        print(f"{model_to_test} - An error occurred during Langchain ChatLiteLLM test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Assistants API Test ---
# Tests LiteLLM's implementation of the OpenAI Assistants API pattern.
# This involves creating an assistant, creating a thread, adding messages, running the thread, and retrieving results.
# It supports OpenAI and Azure OpenAI providers for this feature.
def run_assistants_api_test(provider: str, model_for_assistant: str):
    print(f"\n--- Testing Assistants API ({provider}) with model {model_for_assistant} ---")

    if provider != "openai": # For now, focusing on OpenAI as per docs and common usage
        print(f"INFO: Assistants API test is primarily configured for 'openai' provider. Skipping for '{provider}'.")
        return
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found. Skipping Assistants API test.")
        return

    assistant_id = None
    thread_id = None

    try:
        # 1. Create Assistant
        print("\n1. Creating Assistant...")
        assistant_name = "Test Math Tutor"
        instructions = "You are a personal math tutor. When asked a question, write and run Python code if needed to answer the question."
        created_assistant = litellm.create_assistants(
            custom_llm_provider=provider,
            model=model_for_assistant,
            name=assistant_name,
            instructions=instructions,
            tools=[{"type": "code_interpreter"}]
        )
        assistant_id = created_assistant.id
        print(f"Assistant created with ID: {assistant_id}, Name: {created_assistant.name}")
        assert created_assistant.name == assistant_name

        # 1b. Get/List Assistants (optional check)
        print("\n1b. Listing Assistants...")
        list_of_assistants = litellm.get_assistants(custom_llm_provider=provider)
        found_in_list = any(a.id == assistant_id for a in list_of_assistants.data)
        print(f"Newly created assistant found in list: {found_in_list}")
        assert found_in_list

        # 2. Create Thread
        print("\n2. Creating Thread...")
        initial_user_message = "I need to solve the equation 3x + 11 = 14. Can you help?"
        created_thread = litellm.create_thread(
            custom_llm_provider=provider,
            messages=[{"role": "user", "content": initial_user_message}]
        )
        thread_id = created_thread.id
        print(f"Thread created with ID: {thread_id}")

        # 2b. Get Thread (optional check)
        retrieved_thread = litellm.get_thread(custom_llm_provider=provider, thread_id=thread_id)
        assert retrieved_thread.id == thread_id
        print(f"Successfully retrieved thread: {retrieved_thread.id}")

        # 3. Add Message to Thread (already done via create_thread, but showing add_message explicitly)
        # If the thread was created empty, or to add subsequent messages:
        print("\n3. Adding another message to Thread (example)...")
        follow_up_message_content = "What are the steps?"
        added_msg_obj = litellm.add_message(
            custom_llm_provider=provider,
            thread_id=thread_id,
            role="user",
            content=follow_up_message_content
        )
        print(f"Message added to thread {thread_id}. Message ID: {added_msg_obj.id}")
        assert added_msg_obj.thread_id == thread_id
        # Handle different content formats - content might be a list or string
        actual_content = added_msg_obj.content
        if isinstance(actual_content, list) and len(actual_content) > 0:
            actual_content = actual_content[0].text if hasattr(actual_content[0], 'text') else str(actual_content[0])
        print(f"Expected content: '{follow_up_message_content}', Actual content: '{actual_content}'")
        # More flexible content check
        assert follow_up_message_content in str(actual_content), f"Expected content not found in: {actual_content}"
        # Original assertion
        # assert added_msg_obj.content == follow_up_message_content

        # 4. Run Assistant on Thread
        print("\n4. Running Assistant on Thread...")
        # The run object itself is complex and represents the run's lifecycle.
        # We need to poll or use events for results in a real app.
        # For this test, we'll initiate the run and then retrieve messages.
        run_obj = litellm.run_thread(
            custom_llm_provider=provider,
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        print(f"Run initiated with ID: {run_obj.id}, Status: {run_obj.status}")

        # Wait for the run to complete - OpenAI's API requires polling.
        # LiteLLM's run_thread might block until completion or might require polling.
        # Let's assume for now it might not block fully, so we add a delay and check messages.
        # A more robust way is to poll run_obj.status.
        print("Waiting for assistant to process...")
        import time
        time.sleep(15) # Increase if assistant/model is slow or involves complex tool use

        # Check run status again (optional advanced check, requires get_run function if available)
        # For now, we proceed to get messages, assuming completion or some messages are available.

        # 5. Get Messages from Thread (to see assistant's response)
        print("\n5. Retrieving messages from Thread...")
        thread_messages_response = litellm.get_messages(
            custom_llm_provider=provider,
            thread_id=thread_id
        )
        print(f"Found {len(thread_messages_response.data)} messages in the thread:")
        assistant_responded = False
        for msg in reversed(thread_messages_response.data): # Print in chronological order
            print(f"  - Role: {msg.role}, Content: {str(msg.content)[:200]}...") # Content can be a list of objects
            if msg.role == "assistant":
                assistant_responded = True
        
        if not assistant_responded:
            print("WARNING: Assistant does not seem to have responded yet or no assistant message found.")
            print(f"Final run status if available from initial run_obj: {run_obj.status}")
            # You might need to poll the run status: litellm.retrieve_run(thread_id=thread_id, run_id=run_obj.id)
            # and check for 'completed' status.

        assert assistant_responded, "Assistant did not respond in the retrieved messages."

    except Exception as e:
        print(f"An error occurred during Assistants API test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 6. Delete Assistant (Cleanup)
        if assistant_id and provider == "openai": # Ensure it's an OpenAI assistant we created
            try:
                print(f"\n6. Deleting Assistant ID: {assistant_id}...")
                litellm.delete_assistant(custom_llm_provider=provider, assistant_id=assistant_id)
                print(f"Assistant {assistant_id} deleted.")
            except Exception as e:
                print(f"Error deleting assistant {assistant_id}: {e}")
        # Note: Threads and messages usually persist unless explicitly deleted.
        # OpenAI doesn't have a direct 'delete_thread' in the same way via LiteLLM's top-level funcs,
        # but thread deletion is part of OpenAI's API. For this test, we'll skip thread deletion.

# --- Helper Function for Embedding Test ---
# Demonstrates how to generate embeddings (vector representations of text)
# using `litellm.embedding()`. It tests with various embedding models.
def run_embedding_test():
    print(f"\n--- Testing Embedding Generation ---")

    # Test Case 1: OpenAI Embedding
    print("\n- Test Case 1: OpenAI Embedding (text-embedding-3-small)")
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found. Skipping OpenAI embedding test.")
    else:
        try:
            openai_input_texts = ["Hello from LiteLLM!", "Testing OpenAI embeddings."]
            response_openai = litellm.embedding(
                model="text-embedding-3-small", 
                input=openai_input_texts,
                dimensions=256  # Test model-specific parameter
            )
            print(f"OpenAI Embedding Response (Model: {response_openai.model}):")
            if hasattr(response_openai, 'usage') and response_openai.usage:
                 print(f"  Usage: Prompt Tokens - {response_openai.usage.prompt_tokens}, Total Tokens - {response_openai.usage.total_tokens}")
            else:
                print("  Usage information not available in response.")

            assert len(response_openai.data) == len(openai_input_texts)
            for i, data_item in enumerate(response_openai.data):
                # Handle different response formats - check if it's a dict or object
                if isinstance(data_item, dict):
                    assert data_item["object"] == "embedding"
                    assert data_item["index"] == i
                    assert len(data_item["embedding"]) == 256 # Check if dimensions param worked
                    print(f"  Embedding for input {i+1} (first 3 dims): {str(data_item['embedding'][:3])[:100]}... Length: {len(data_item['embedding'])}")
                else:
                    assert data_item.object == "embedding"
                    assert data_item.index == i
                    assert len(data_item.embedding) == 256 # Check if dimensions param worked
                    print(f"  Embedding for input {i+1} (first 3 dims): {str(data_item.embedding[:3])[:100]}... Length: {len(data_item.embedding)}")
        except Exception as e:
            print(f"Error during OpenAI embedding test: {e}")
            import traceback
            traceback.print_exc()

    # Test Case 2: Cohere Embedding (example of another provider + provider-specific param)
    print("\n- Test Case 2: Cohere Embedding (embed-english-v3.0)")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    if not COHERE_API_KEY:
        print("WARNING: COHERE_API_KEY not found. Skipping Cohere embedding test.")
    else:
        try:
            cohere_input_texts = ["This is a document for search.", "Another document here."]
            # Ensure the model name includes the provider prefix for clarity with LiteLLM
            response_cohere = litellm.embedding(
                model="cohere/embed-english-v3.0", 
                input=cohere_input_texts,
                input_type="search_document"  # Provider-specific param for Cohere v3
            )
            print(f"Cohere Embedding Response (Model: {response_cohere.model}):")
            if hasattr(response_cohere, 'usage') and response_cohere.usage:
                print(f"  Usage: Prompt Tokens - {response_cohere.usage.prompt_tokens}, Total Tokens - {response_cohere.usage.total_tokens}")
            else:
                print("  Usage information not available in response.")
            assert len(response_cohere.data) == len(cohere_input_texts)
            for i, data_item in enumerate(response_cohere.data):
                # Handle different response formats - check if it's a dict or object
                if isinstance(data_item, dict):
                    assert data_item["object"] == "embedding"
                    assert data_item["index"] == i
                    print(f"  Embedding for input {i+1} (first 3 dims): {str(data_item['embedding'][:3])[:100]}... Length: {len(data_item['embedding'])}")
                else:
                    assert data_item.object == "embedding"
                    assert data_item.index == i
                    print(f"  Embedding for input {i+1} (first 3 dims): {str(data_item.embedding[:3])[:100]}... Length: {len(data_item.embedding)}")
        except Exception as e:
            print(f"Error during Cohere embedding test: {e}")
            import traceback
            traceback.print_exc()

    print("\n- Conceptual Note: Image Embeddings")
    print("  LiteLLM supports image embeddings for compatible models by passing a base64 encoded image string.")

# --- Helper Function for Exception Handling Test ---
# This function showcases how to catch and handle common LiteLLM exceptions,
# such as `AuthenticationError`, `RateLimitError`, `Timeout`, etc.
# It intentionally tries to trigger these errors to demonstrate the handling.
def run_exception_handling_test():
    print(f"\n--- Testing Exception Handling ---")

    # Test Case 1: APITimeoutError
    print("\n- Test Case 1: Triggering APITimeoutError")
    if not os.getenv("OPENAI_API_KEY"): # Requires a valid key to even attempt a call that times out
        print("WARNING: OPENAI_API_KEY not found. Skipping APITimeoutError test.")
    else:
        try:
            messages = [{"role": "user", "content": "This is a test for timeout."}]
            response = litellm.completion(
                model=OPENAI_GPT4O_MODEL, # Use a configured OpenAI model
                messages=messages,
                timeout=0.001  # Extremely low timeout to force the error
            )
            print(f"Timeout Test: Unexpected success - Response: {response}")
        except openai.APITimeoutError as e: # Catching the specific mapped exception
            print(f"Successfully caught APITimeoutError as expected.")
            print(f"  Exception Type: {type(e)}")
            print(f"  Message: {e.message if hasattr(e, 'message') else str(e)}")
            if hasattr(e, 'status_code'): # LiteLLM adds these attributes
                print(f"  Status Code: {e.status_code}")
                should_retry_status = litellm._should_retry(e.status_code) # As per docs
                print(f"  litellm._should_retry({e.status_code}) suggests: {should_retry_status}")
            if hasattr(e, 'llm_provider'):
                print(f"  LLM Provider: {e.llm_provider}")
        except Exception as e:
            print(f"APITimeoutError test failed: Did not catch openai.APITimeoutError. Caught: {type(e)} - {e}")
            import traceback
            traceback.print_exc()

    # Test Case 2: AuthenticationError
    print("\n- Test Case 2: Triggering AuthenticationError")
    # We will use a known provider (OpenAI) but with a fake key for this specific call.
    # Ensure this doesn't permanently alter os.environ if other tests rely on the correct key.
    # The 'api_key' param in litellm.completion overrides env var for that call.
    if not os.getenv("OPENAI_API_KEY"): # Check if original key exists to show this is a deliberate override
         print("INFO: OPENAI_API_KEY not set, this test might not be as illustrative of overriding a valid key.")

    try:
        messages = [{"role": "user", "content": "This is a test for auth error."}]
        response = litellm.completion(
            model=OPENAI_GPT4O_MODEL, 
            messages=messages,
            api_key="sk-thisisafakekeyandshouldfail12345", # Override with a bad API key
            timeout=5 # Give it a moment to fail auth, not timeout
        )
        print(f"AuthenticationError Test: Unexpected success - Response: {response}")
    except openai.AuthenticationError as e:
        print(f"Successfully caught AuthenticationError as expected.")
        print(f"  Exception Type: {type(e)}")
        print(f"  Message: {e.message if hasattr(e, 'message') else str(e)}")
        if hasattr(e, 'status_code'):
            print(f"  Status Code: {e.status_code}")
        if hasattr(e, 'llm_provider'):
            print(f"  LLM Provider: {e.llm_provider}")
    except Exception as e:
        print(f"AuthenticationError test failed: Did not catch openai.AuthenticationError. Caught: {type(e)} - {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Provider-Specific Parameters Test ---
# Demonstrates passing parameters that are specific to a particular LLM provider
# through LiteLLM. LiteLLM passes through unknown parameters to the underlying SDK call.
def run_provider_specific_params_test(model_to_test: str, param_key: str, param_value: any):
    print(f"\n--- Testing Provider-Specific Parameters with {model_to_test} ---")
    print(f"Attempting to pass '{param_key}={param_value}'")

    # API key check (simplified for the specific model)
    api_key_present = False
    if "claude" in model_to_test.lower() and os.getenv("ANTHROPIC_API_KEY"):
        api_key_present = True
    # Add other providers if testing specific params for them

    if not api_key_present:
        print(f"WARNING: API key for {model_to_test} not found. Skipping provider-specific param test.")
        return

    messages = [{"role": "user", "content": "Tell me a short joke."}]
    
    # Construct kwargs for provider-specific parameters
    provider_params = {param_key: param_value}

    try:
        print(f"Attempting call to {model_to_test} with custom param: {provider_params}...")
        # LiteLLM should pass unknown kwargs (not part of standard OpenAI API) to the provider
        response = litellm.completion(
            model=model_to_test,
            messages=messages,
            **provider_params # Pass the provider-specific params here
        )
        print(f"{model_to_test} - Response received with custom param passed (no direct validation of effect here, just that call succeeded).")
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            print(f"  Content sample: {response.choices[0].message.content[:100]}...")
        else:
            print("  No content in response.")
        # Note: Validating the *effect* of provider-specific params can be hard without knowing
        # the exact behavior or if the param influences observable output like logprobs (if returned).
        # This test primarily checks if LiteLLM allows passing them without error.
    except litellm.exceptions.UnsupportedParamsError as e: # Specifically catch if LiteLLM blocks it
        print(f"Caught litellm.exceptions.UnsupportedParamsError: {e}")
        print("  This means LiteLLM explicitly did not pass the parameter, or the provider rejected it as unknown via LiteLLM's mapping.")
    except Exception as e:
        print(f"{model_to_test} - An error occurred during provider-specific param test: {e}")
        print(f"  This could be due to the provider rejecting the param, or another issue.")
        import traceback
        traceback.print_exc()

# --- Helper Function for PDF Input Test ---
# This function tests the conceptual support for PDF inputs with models that can handle them.
# It uses `litellm.utils.supports_pdf_input` to check model compatibility and
# demonstrates how image/PDF data (as base64) might be passed in the messages.
def run_pdf_input_test(model_to_test: str):
    print(f"\n--- Testing PDF Input with {model_to_test} ---")

    # This feature is documented for Vertex AI, Bedrock, and Anthropic API models.
    # We'll use the ANTHROPIC_SONNET_MODEL as a placeholder.
    # Ensure your ANTHROPIC_API_KEY is set if model_to_test is a direct Anthropic model.
    # If it's a Bedrock/Vertex model, ensure those keys/auth are set.

    # First, check with the utility if the model is known to support PDF input.
    # The utility function might need specific model prefixes (e.g., "bedrock/").
    # We'll pass the raw model_to_test string.
    # The second argument to supports_pdf_input is 'client', can be None for this check.
    if not supports_pdf_input(model_to_test, None):
        print(f"INFO: Model {model_to_test} does not support PDF input according to litellm.utils.supports_pdf_input(). Skipping PDF input test.")
        print(f"      Note: This utility might be more accurate with specific provider prefixes if applicable (e.g., 'bedrock/{model_to_test}').")
        return

    api_key_needed = False
    if "anthropic" in model_to_test.lower() or "claude" in model_to_test.lower():
        if os.getenv("ANTHROPIC_API_KEY"):
            api_key_needed = True
    # Add checks for Bedrock/Vertex if model_to_test is for them
    # For example, for Bedrock:
    # elif "bedrock" in model_to_test.lower():
    # if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY") and os.getenv("AWS_REGION_NAME"):
    # api_key_needed = True

    if not api_key_needed and ("anthropic" in model_to_test.lower() or "claude" in model_to_test.lower()): # Simplified check for now
        print(f"WARNING: Appropriate API key for {model_to_test} (e.g., ANTHROPIC_API_KEY) not found. Skipping PDF input test.")
        return

    dummy_pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    question_about_pdf = "What is this document about in one sentence?"

    # Test Case 1: PDF via URL
    print(f"\n- Test Case 1: PDF Input via URL ({dummy_pdf_url})")
    try:
        content_with_url = [
            {"type": "text", "text": question_about_pdf},
            {
                "type": "file", # As per docs, though some multimodal use 'image_url' or 'document_url'
                "file": {       # This structure seems specific to the new PDF input feature
                    "file_id": dummy_pdf_url,
                    "format": "application/pdf" # Explicitly setting format
                }
            },
        ]
        messages_url = [{"role": "user", "content": content_with_url}]

        response_url = litellm.completion(
            model=model_to_test,
        )
        print(f"{model_to_test} - Response (PDF URL input):")
        if response_url.choices and response_url.choices[0].message and response_url.choices[0].message.content:
            print(f"  Content: {response_url.choices[0].message.content[:200]}...")
        else:
            print("  No content in response.")
    except Exception as e:
        print(f"Error during PDF input (URL) test for {model_to_test}: {e}")
        import traceback
        traceback.print_exc()

    # Test Case 2: PDF via Base64
    print(f"\n- Test Case 2: PDF Input via Base64")
    try:
        # Download the PDF content
        http_response = requests.get(dummy_pdf_url)
        http_response.raise_for_status()  # Ensure the request was successful
        file_data = http_response.content
        encoded_file = base64.b64encode(file_data).decode("utf-8")
        base64_data_uri = f"data:application/pdf;base64,{encoded_file}"

        content_with_base64 = [
            {"type": "text", "text": question_about_pdf},
            {
                "type": "file",
                "file": {
                    "file_data": base64_data_uri,
                    # "format": "application/pdf" # Format can also be specified here
                }
            },
        ]
        messages_base64 = [{"role": "user", "content": content_with_base64}]

        response_base64 = litellm.completion(
            model=model_to_test,
        )
        print(f"{model_to_test} - Response (PDF Base64 input):")
        if response_base64.choices and response_base64.choices[0].message and response_base64.choices[0].message.content:
            print(f"  Content: {response_base64.choices[0].message.content[:200]}...")
        else:
            print("  No content in response.")
    except Exception as e:
        print(f"Error during PDF input (Base64) test for {model_to_test}: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Conceptual Notes on Proxy File Features ---
# This function doesn't execute calls but prints information about features
# typically configured when running LiteLLM as a proxy (e.g., via a config file),
# such as `max_budget`, `rpm_limit`, `user_max_budget`.
def run_conceptual_proxy_file_features_info():
    print(f"\n--- Conceptual Notes on LiteLLM Proxy-Specific File Features ---")
    
    print("\n- Provider Files Endpoints (/files):")
    print("  This feature allows direct interaction with provider's /files endpoints (upload, list, retrieve, delete, get content)")
    print("  It is accessed by making OpenAI-SDK compatible calls to a running LiteLLM Proxy.")
    print("  Requires: LiteLLM Proxy setup with `files_settings` in config.yaml.")
    print("  SDK interaction: Use an OpenAI client pointed at the LiteLLM Proxy URL (e.g., http://localhost:4000/v1).")
    print("  Example: client.files.create(file=..., purpose=..., extra_body={'custom_llm_provider': 'openai'})")

    print("\n- [BETA] LiteLLM Managed Files:")
    print("  This is a LiteLLM Enterprise feature for reusing the same file ID across different providers via the Proxy.")
    print("  It helps manage file permissions and abstracts provider-specific file IDs.")
    print("  Requires: LiteLLM Proxy setup, PostgreSQL database (`DATABASE_URL`), and `general_settings` in config.yaml.")
    print("  SDK interaction: Use an OpenAI client pointed at the LiteLLM Proxy URL.")
    print("  Example for uploading: client.files.create(file=..., purpose=..., extra_body={'target_model_names': 'model1,model2'})")
    print("  Example for using in completion: {'type': 'file', 'file': {'file_id': 'litellm_proxy_managed_id_...'}}")

    print("\n- LiteLLM Proxy SDK Usage (General):")
    print("  To use the LiteLLM SDK to talk to a LiteLLM Proxy (for any endpoint the proxy exposes):")
    print("  1. Prefix model names: `model='litellm_proxy/your-deployed-model-name'`")
    print("  2. Set `litellm.api_base` to your proxy URL and `litellm.api_key` to your proxy key.")
    print("  3. Or, set `litellm.use_litellm_proxy = True` (uses LITELLM_PROXY_API_BASE and LITELLM_PROXY_API_KEY env vars).")

    print("\nThese features are powerful for production deployments using the LiteLLM Proxy but are not directly tested")
    print("as standalone SDK calls in this script, as they require a running proxy instance and configuration.")

# --- Helper Function for Provider-Specific Config Object Test ---
# Demonstrates using LiteLLM's provider-specific configuration objects (e.g., `litellm.OpenAIConfig`).
# These allow setting default parameters (like `max_tokens` or `temperature`) for all subsequent calls
# made to models from that provider, until the config is reset.
def run_provider_config_object_test():
    print(f"\n--- Testing Provider-Specific Config Objects (e.g., OpenAIConfig) ---")

    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found. Skipping OpenAIConfig test.")
        return

    model_to_use = O4_MINI_MODEL_NAME # Using a fast OpenAI model
    original_max_tokens_behavior_text = ""
    config_applied_text = ""

    # Step 1: Call without specific global config (or after it might have been reset)
    # Note: This relies on the global state not being previously set to a very low max_tokens.
    # For a truly isolated test, one might need to restart the interpreter or have a reset function.
    print(f"\n- Step 1: Call {model_to_use} with default/per-call max_tokens (e.g., 50)")
    try:
        messages_default = [{"role": "user", "content": "Tell me a short story about a adventurous cat."}]
        response_default = litellm.completion(
            model=model_to_use,
            messages=messages_default,
            max_tokens=50 # Explicitly set for this call for comparison baseline
        )
        if response_default.choices and response_default.choices[0].message and response_default.choices[0].message.content:
            original_max_tokens_behavior_text = response_default.choices[0].message.content
            print(f"  Response with max_tokens=50 (length {len(original_max_tokens_behavior_text)}): {original_max_tokens_behavior_text[:100]}...")
        else:
            print("  No content in default response.")
    except Exception as e:
        print(f"  Error during default call: {e}")


    # Step 2: Apply global config and make a call
    print(f"\n- Step 2: Applying litellm.OpenAIConfig(max_completion_tokens=10) and calling {model_to_use}")
    try:
        litellm.OpenAIConfig(max_completion_tokens=10) # Set a global max_completion_tokens for OpenAI calls via this config
        # Note: This change is global for subsequent litellm.completion calls to OpenAI models
        # that don't override max_tokens in the call itself.

        messages_config = [{"role": "user", "content": "Tell me a short story about a curious dog."}]
        # This call should now use max_tokens=10 due to the global OpenAIConfig
        response_config = litellm.completion(
            model=model_to_use,
            messages=messages_config
            # No max_completion_tokens here, should pick up from OpenAIConfig
        )
        print(f"{model_to_use} - Response after setting OpenAIConfig(max_completion_tokens=10):")
        if response_config.choices and response_config.choices[0].message and response_config.choices[0].message.content:
            config_applied_text = response_config.choices[0].message.content
            print(f"  Content (length {len(config_applied_text)}): {config_applied_text[:100]}...")
            if len(config_applied_text) < len(original_max_tokens_behavior_text) and len(original_max_tokens_behavior_text) > 10:
                 print(f"  INFO: Content length ({len(config_applied_text)}) is noticeably shorter than baseline ({len(original_max_tokens_behavior_text)}), suggesting max_completion_tokens=10 from config likely took effect.")
            elif len(config_applied_text) > 20: # Heuristic, 10 tokens can be > 20 chars
                 print(f"  WARNING: Content length ({len(config_applied_text)}) seems too long for max_completion_tokens=10. Config might not have applied as expected or tokenization is very efficient.")
            else:
                 print(f"  INFO: Content received. Length {len(config_applied_text)}.")

        else:
            print("  No content in response with config.")

        # Attempt to "reset" or demonstrate overriding the global config for subsequent tests' sanity.
        # This is a bit of a hack. Ideally, LiteLLM would provide context managers for such settings.
        # Setting it to a high value or a typical default.
        print("\n- Step 3: Attempting to 'reset' OpenAIConfig max_completion_tokens to a higher value (e.g., None or default).")
        litellm.OpenAIConfig(max_completion_tokens=None) # Or a high number like 1024. 'None' might make it use model default.
        # Making another call to see if it's longer than the max_tokens=10 response
        response_reset_attempt = litellm.completion(
            model=model_to_use,
            messages=[{"role": "user", "content": "Tell me a short story about a brave bird."}],
            # No max_tokens here, should pick up from the new OpenAIConfig(max_tokens=None)
        )
        if response_reset_attempt.choices and response_reset_attempt.choices[0].message and response_reset_attempt.choices[0].message.content:
            reset_text_length = len(response_reset_attempt.choices[0].message.content)
            print(f"  Response after 'resetting' OpenAIConfig (length {reset_text_length}): {response_reset_attempt.choices[0].message.content[:100]}...")
            if reset_text_length > len(config_applied_text) + 5 : # Expect it to be longer than the max_tokens=10 output
                print("  INFO: 'Reset' call produced longer text, suggesting global config was altered.")
            else:
                print("  WARNING: 'Reset' call did not produce significantly longer text than max_tokens=10. Global state may persist or model output was naturally short.")

    except Exception as e:
        print(f"Error during OpenAIConfig test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Explicitly try to set it to a less restrictive default for other tests
        # This is important because these config objects are global.
        litellm.OpenAIConfig(max_tokens=None) 
        print("INFO: OpenAIConfig max_tokens has been set to None to minimize impact on subsequent tests.")

# --- Helper Function for Rerank Test ---
# Tests the `litellm.rerank()` function, which is used to reorder a list of documents
# based on their relevance to a query, typically using a reranking model (e.g., from Cohere).
def run_rerank_test():
    print(f"\n--- Testing Reranking (litellm.rerank) ---")

    # Test Case: Cohere Rerank
    print("\n- Test Case: Cohere Rerank (cohere/rerank-english-v3.0)")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    if not COHERE_API_KEY:
        print("WARNING: COHERE_API_KEY not found. Skipping Cohere rerank test.")
        return
    
    # Ensure the model name is prefixed for LiteLLM to identify Cohere as the provider
    cohere_rerank_model = "cohere/rerank-english-v3.0"
    query = "What is the capital of Canada?"
    documents = [
        "Ottawa is the capital of Canada, located in Ontario.",
        "The Eiffel Tower is a famous landmark in Paris, France.",
        "Canada is known for its maple syrup and cold winters.",
        "The capital of France is Paris.",
        "Toronto is the largest city in Canada but not the capital."
    ]

    try:
        print(f"Attempting rerank with model '{cohere_rerank_model}'...")
        response = litellm.rerank(
            model=cohere_rerank_model,
            query=query,
            documents=documents,
            top_n=3 # Optional: Get top 3 relevant documents
        )
        
        print(f"Cohere Rerank Response (ID: {response.id}):")
        assert response.results is not None
        print(f"  Received {len(response.results)} results (expected up to top_n=3):")
        
        for i, result in enumerate(response.results):
            print(f"  Result {i+1}: Index (original) - {result.index}, Relevance Score - {result.relevance_score:.4f}")
            print(f"    Document: \"{documents[result.index]}\"") # Access original document using index
            if i == 0: # First result should be most relevant
                assert "Ottawa" in documents[result.index].lower()

    except Exception as e:
        print(f"Error during Cohere rerank test: {e}")
        import traceback
        traceback.print_exc()

# --- Helper Function for Router Basic Tests (Async) ---
# This function tests the basic functionalities of the LiteLLM Router.
# It demonstrates initializing a Router with a list of model deployments,
# making synchronous and asynchronous completion calls through the router,
# and basic routing strategies.
async def run_router_basic_tests(): # Changed to async
    print(f"\n--- Testing LiteLLM Router (SDK) - Basics, Retries, Strategy ---")

    # Define a model list for the router.
    # For SDK testing, these can point to actual models you have keys for.
    # We can use the same actual model with different aliases or slightly different params
    # to simulate different "deployments" for the router.
    
    # Ensure API keys are available for the models used in the router_model_list
    # For simplicity, this example will focus on OpenAI models.
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found. Skipping most Router tests.")
        return

    router_model_list = [
        {
            "model_name": "grouped-openai-fast", # Alias for this group of deployments
            "litellm_params": {
                "model": O4_MINI_MODEL_NAME, # Actual LiteLLM model string
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "deployment_name": "o4-mini-deployment-1" # Custom identifier
        },
        {
            "model_name": "grouped-openai-fast", # Same alias, different "deployment"
            "litellm_params": {
                "model": O4_MINI_MODEL_NAME, # Could be another O4_MINI_MODEL_NAME or even OPENAI_GPT4O_MODEL if different performance/limits
                "max_tokens": None, # Explicitly nullify to prevent conflicts with max_completion_tokens
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "deployment_name": "o4-mini-deployment-2"
        },
        {
            "model_name": "gpt-4o-default",
            "litellm_params": {
                "model": OPENAI_GPT4O_MODEL,
                "max_tokens": None, # Explicitly nullify to prevent conflicts with max_completion_tokens
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
             "deployment_name": "gpt4o-main"
        }
    ]

    print("\n- Test 1: Initializing Router and basic acompletion call")
    try:
        # Default routing_strategy is "simple-shuffle"
        # num_retries can be set for the router globally
        router = Router(
            model_list=router_model_list,
            num_retries=1, # Router-level retries for failed calls to a deployment
            set_verbose=False # Can be True for more detailed LiteLLM Router logs
        )
        
        messages = [{"role": "user", "content": "Hello Router! Tell me a fact."}]
        
        print("Attempting router.acompletion()...")
        # Use max_completion_tokens for newer OpenAI models instead of max_tokens
        response_async = await router.acompletion(
            model="grouped-openai-fast", 
            messages=messages,
            max_completion_tokens=150  # Use max_completion_tokens instead of max_tokens
        )
        print("Async Router Response:")
        if response_async and response_async.choices and response_async.choices[0].message:
            print(f"  Model used: {response_async.model}") # Will show the underlying model string
            print(f"  Content: {response_async.choices[0].message.content[:100]}...")
            # To see which specific deployment was chosen, LiteLLM logs might show it,
            # or a custom callback would be needed as per docs.
            # response_async might have _response_raw which sometimes contains more deployment info.
            if hasattr(response_async, '_response_raw') and response_async._response_raw and \
               hasattr(response_async._response_raw, 'headers') and response_async._response_raw.headers:
                print(f"  Response headers (raw): {response_async._response_raw.headers.get('x-litellm-deployment-id', 'Not set')}")


        else:
            print("  No valid response or content from async completion.")

    except Exception as e:
        print(f"Error during basic router acompletion test: {e}")
        import traceback
        traceback.print_exc()

    print("\n- Test 2: Router synchronous completion call")
    try:
        # Re-initialize router or use existing one if state is not an issue for this test
        router_sync = Router(model_list=router_model_list, num_retries=1)
        messages_sync = [{"role": "user", "content": "Hello Sync Router! Give me a quote."}]
        
        print("Attempting router.completion()...")
        # Use max_completion_tokens for newer OpenAI models instead of max_tokens
        response_sync = router_sync.completion(
            model="gpt-4o-default", 
            messages=messages_sync,
            max_completion_tokens=50  # Use max_completion_tokens instead of max_tokens
        )
        print("Sync Router Response:")
        if response_sync and response_sync.choices and response_sync.choices[0].message:
            print(f"  Model used: {response_sync.model}")
            print(f"  Content: {response_sync.choices[0].message.content[:100]}...")
        else:
            print("  No valid response or content from sync completion.")
            
    except Exception as e:
        print(f"Error during basic router completion test: {e}")
        import traceback
        traceback.print_exc()

    print("\n- Test 3: Router retries (conceptual - difficult to force a retryable error cleanly in SDK)")
    # We've set num_retries=1 (or more) in Router init.
    # Forcing a retryable error like a transient network issue or a specific 429 that the router retries
    # is hard to do reliably in a standalone script without mocking.
    # The `run_model_fallback_test` already demonstrates LiteLLM handling certain failures and trying alternatives.
    # The router's `num_retries` applies to retrying the *same* chosen deployment if it fails,
    # before considering fallbacks or giving up on that deployment for the current call.
    print("  Router `num_retries` is set. LiteLLM Router will attempt retries on a failing deployment up to this count.")
    print("  (This test is conceptual as forcing specific retryable errors is complex here).")

# --- Helper Function for Router Advanced Tests (Async) ---
# This function tests more advanced features of the LiteLLM Router, including:
# - Fallbacks: If a model call fails, the router can automatically try other deployments.
# - Caching: The router can cache responses to identical requests to save costs and improve latency.
# - Prioritization: The router can prioritize certain deployments based on specified criteria.
# - Cooldowns and Timeouts: Handling for unavailable deployments.
async def run_router_advanced_tests():
    print(f"\n--- Testing LiteLLM Router (SDK) - Fallbacks, Caching, Prioritization ---")

    if not os.getenv("OPENAI_API_KEY"): # Fallback also uses OpenAI for this example
        print("WARNING: OPENAI_API_KEY not found. Skipping Router advanced tests.")
        return
    # Potentially add key check for ANTHROPIC if it's a fallback target.
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY not found. Some fallback scenarios might not be fully testable.")


    # Define a model list with a "bad" deployment to trigger fallbacks
    # and some working deployments.
    router_model_list_for_fallback = [
        {
            "model_name": "primary-group",
            "litellm_params": {
                "model": "openai/this-is-a-fake-openai-model-string", # Invalid model to force failure
                "api_key": "bad-key-for-primary", # Will cause auth error
            },
            "deployment_name": "fake-deployment-1"
        },
        {
            "model_name": "fallback-group-openai",
            "litellm_params": {
                "model": OPENAI_GPT4O_MODEL,
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "deployment_name": "gpt4o-fallback-deploy"
        },
        {
            "model_name": "fallback-group-anthropic", # Another potential fallback
            "litellm_params": {
                "model": ANTHROPIC_SONNET_MODEL,
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            },
            "deployment_name": "sonnet-fallback-deploy"
        }
    ]

    print("\n- Test 4: Router-level Fallbacks")
    try:
        # Fallback from 'primary-group' to 'fallback-group-openai', then to 'fallback-group-anthropic'
        router_with_fallbacks = Router(
            model_list=router_model_list_for_fallback,
            fallbacks=[
                {"primary-group": ["fallback-group-openai", "fallback-group-anthropic"]},
            ],
            num_retries=0 # Set to 0 to quickly go to fallbacks for this test
        )
        
        messages = [{"role": "user", "content": "Tell me about routing fallbacks."}]
        print("Attempting router.acompletion() with a primary model expected to fail, triggering fallbacks...")
        # Use max_completion_tokens for newer OpenAI models instead of max_tokens
        response_fallback = await router_with_fallbacks.acompletion(
            model="primary-group", 
            messages=messages,
            max_completion_tokens=100  # Use max_completion_tokens instead of max_tokens
        )
        
        print("Router Fallback Response:")
        if response_fallback and response_fallback.choices and response_fallback.choices[0].message:
            print(f"  Successfully fell back to model: {response_fallback.model}")
            print(f"  Content: {response_fallback.choices[0].message.content[:100]}...")
            # Check if it fell back to one of the expected groups
            assert OPENAI_GPT4O_MODEL in response_fallback.model or ANTHROPIC_SONNET_MODEL in response_fallback.model
        else:
            print("  No valid response or content from fallback.")
            assert False, "Fallback did not yield a valid response."

    except Exception as e:
        print(f"Error during router fallback test: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Router fallback test raised an unexpected exception: {e}"


    print("\n- Test 5: Router-level Caching (In-Memory)")
    # Use a simpler model list for caching demonstration
    router_model_list_for_caching = [{
        "model_name": "cache-test-model",
        "litellm_params": {"model": O4_MINI_MODEL_NAME, "api_key": os.getenv("OPENAI_API_KEY")}
    }]
    
    try:
        # Router handles its own caching internally when cache_responses=True is set
        router_with_cache = Router(
            model_list=router_model_list_for_caching,
            cache_responses=True,  # Enable in-memory cache for this router instance
            set_verbose=False 
        )
        
        # Unique content for the first call to ensure it's not a hit from a previous unrelated test
        cache_test_messages_router = [{"role": "user", "content": f"Router cache test: What is the weather like today? {time.time()}"}] 
        
        print("  Attempting first call to router (should not be cached by this router instance yet)...")
        response_cache1_router = await router_with_cache.acompletion(
            model="cache-test-model", 
            messages=cache_test_messages_router,
            max_completion_tokens=50
        )
        response1_id_router = response_cache1_router.id if hasattr(response_cache1_router, 'id') else None
        print(f"  Router - First response ID: {response1_id_router}, Content: {response_cache1_router.choices[0].message.content[:30]}...")

        print("  Attempting second call to router with identical messages (should be cached by this router instance)...")
        response_cache2_router = await router_with_cache.acompletion(
            model="cache-test-model", 
            messages=cache_test_messages_router,
            max_completion_tokens=50
        )
        response2_id_router = response_cache2_router.id if hasattr(response_cache2_router, 'id') else None
        print(f"  Router - Second response ID: {response2_id_router}, Content: {response_cache2_router.choices[0].message.content[:30]}...")

        # Check for cache hit indicators
        cache_hit_header = False
        if hasattr(response_cache2_router, '_response_raw') and \
           hasattr(response_cache2_router._response_raw, 'headers') and \
           response_cache2_router._response_raw.headers.get("x-litellm-cache-hit") == "True":
            cache_hit_header = True
            print("  INFO: Router - Second call was a cache hit (confirmed by x-litellm-cache-hit header)!")
        
        if not cache_hit_header and response1_id_router == response2_id_router and \
           response_cache1_router.choices[0].message.content == response_cache2_router.choices[0].message.content:
             print("  INFO: Router - Second call returned identical content and ID, likely a cache hit.")
        elif not cache_hit_header:
            print("  WARNING: Router - Second call does not appear to be a cache hit. Cache might not be working as expected or params differed.")
            
        print("  Router caching test completed.")

    except Exception as e:
        print(f"  Error during router caching test: {e}")
        import traceback
        traceback.print_exc()

    print("\n- Test 6: Request Prioritization (Conceptual within Router, testing param passthrough)")
    # This test checks if the 'priority' param is passed. OpenAI API does not support it,
    # so we expect a BadRequestError. Other custom LLM providers might.
    model_list_prio = [
        {
            "model_name": "priority-test-model", # This will map to an OpenAI model
            "litellm_params": {
                "model": O4_MINI_MODEL_NAME,
                "max_tokens": None,
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        }
    ]
    router_for_prio = Router(model_list=model_list_prio, num_retries=1) # Low retries for this specific test

    try:
        print("  Attempting acompletion with priority=0 (high)...")
        # Ensure 'messages' is defined and accessible here from previous parts of the function
        response_prio = await router_for_prio.acompletion(
            model="priority-test-model",
            messages=[{"role": "user", "content": "This is a high priority request."}], 
            priority=0,        # This parameter is not supported by OpenAI
            max_completion_tokens=10
        )
        print(f"Prioritization test response (should not be reached for OpenAI): {response_prio.choices[0].message.content[:50]}...")
        print("  WARNING: Test for 'priority' parameter with OpenAI model did not fail as expected. The parameter might have been ignored or supported.")
    except litellm.exceptions.BadRequestError as e:
        # Check if the error message contains the model group string from the router error
        # Example error: litellm.BadRequestError: OpenAIException - Unknown parameter: 'priority'.. Received Model Group=priority-test-model
        error_str = str(e)
        expected_model_group_in_error = "priority-test-model"
        if "Unknown parameter: 'priority'" in error_str and expected_model_group_in_error in error_str.split("Model Group=")[-1]:
            print(f"  SUCCESS: Correctly caught expected BadRequestError for 'priority' parameter with OpenAI model: {error_str.splitlines()[0]}")
        else:
            print(f"  Caught BadRequestError, but not the one expected for 'priority'. Error: {e}")
            # import traceback
            # traceback.print_exc()
    except Exception as e:
        # This will catch other errors, like if 'messages' is not defined
        print(f"  An unexpected error occurred during request prioritization test: {e}")
        import traceback
        traceback.print_exc()

# The main execution block that calls all the test functions.
if __name__ == "__main__":
    print("Starting comprehensive LLM standalone tests with LITELLM_LOG=DEBUG ...") #
    
    def print_section_header(title: str):
        print("\n" + "="*20 + f" {title} " + "="*20 + "\n")

    # --- Reliability Tests (SDK-level, not Router) ---
    print_section_header("Reliability: Retries and Fallbacks")
    # Test retries with a generally reliable model
    run_retries_test(O4_MINI_MODEL_NAME) # Using a fast OpenAI model for this
    # Test model fallbacks - primary model is fake, fallback is a working OpenAI model
    run_model_fallback_test(primary_model_to_fail="openai/this-model-does-not-exist-gpt", fallback_models=[OPENAI_GPT4O_MODEL, ANTHROPIC_SONNET_MODEL])
    # Test model fallbacks - primary model is fake, first fallback is also fake, second fallback is a working model
    run_model_fallback_test(primary_model_to_fail="openai/fake-model-1", fallback_models=["openai/fake-model-2", DEEPSEEK_CHAT_MODEL])

    # --- Batching Tests ---
    print_section_header("Batching Completion Tests")
    # Test batch_completion (many calls to 1 model) - using Anthropic as per docs example
    run_batch_completion_one_model_test(ANTHROPIC_SONNET_MODEL)
    # Test batch_completion_models (fastest) - using a mix of available models
    # Ensure you have API keys for at least one of these for the test to run meaningfully
    run_batch_completion_many_models_fastest_test([OPENAI_GPT4O_MODEL, ANTHROPIC_SONNET_MODEL, DEEPSEEK_CHAT_MODEL, GEMINI_MODEL_NAME_FLASH])
    # Test batch_completion_models_all_responses (all) - using a mix
    run_batch_completion_many_models_all_test([O4_MINI_MODEL_NAME, ANTHROPIC_SONNET_MODEL, GEMINI_MODEL_NAME_FLASH])

    # --- Embedding Tests ---
    print_section_header("Embedding Tests")
    run_embedding_test()

    # --- Exception Handling Tests ---
    print_section_header("Exception Handling Tests")
    run_exception_handling_test()

    # --- Integrations Tests ---
    print_section_header("Integrations Tests")
    # Test Langchain ChatLiteLLM - using an OpenAI model as per docs example
    # Ensure OPENAI_API_KEY is set
    if LANGCHAIN_AVAILABLE: # Check if Langchain was imported successfully
        run_langchain_chatlitellm_test(OPENAI_GPT4O_MODEL)
    else:
        print("Skipping Langchain test as libraries are not available.")
    
    # Test Assistants API (OpenAI)
    # Using gpt-4o as it's generally good with instructions and tools like code_interpreter
    run_assistants_api_test(provider="openai", model_for_assistant=OPENAI_GPT4O_MODEL)

    # --- OpenAI Tests ---
    print_section_header("OpenAI Models")
    run_web_search_test(OPENAI_WEB_SEARCH_COMPLETION_MODEL, OPENAI_GPT4O_MODEL)
    run_function_calling_test(OPENAI_GPT4O_MODEL, client_provider_strategy="instructor_openai")
    run_json_mode_test(OPENAI_GPT4O_MODEL)
    run_json_mode_test(O4_MINI_MODEL_NAME)
    run_predicted_outputs_test(OPENAI_GPT4O_MODEL)
    run_provider_config_object_test() # NEW LINE ADDED HERE

    # --- Gemini Tests ---
    print_section_header("Gemini Models")
    # For Gemini, instructor.from_litellm() is the proven path with LiteLLM v1.72.0+
    run_function_calling_test(GEMINI_MODEL_NAME_PRO, client_provider_strategy="litellm")
    run_function_calling_test(GEMINI_MODEL_NAME_FLASH, client_provider_strategy="litellm")
    # It's good practice to also test JSON mode for Gemini if needed.
    run_json_mode_test(GEMINI_MODEL_NAME_FLASH)

    # --- Anthropic Tests ---
    print_section_header("Anthropic Models")
    # The run_function_calling_test will attempt native Anthropic client first, then fallback to LiteLLM.
    # Ensure `pip install "instructor[anthropic]" anthropic` for the native path.
    run_function_calling_test(ANTHROPIC_SONNET_MODEL, client_provider_strategy="instructor_anthropic")
    run_json_mode_test(ANTHROPIC_SONNET_MODEL)
    run_prefix_assistant_message_test(ANTHROPIC_SONNET_MODEL, api_key=ANTHROPIC_API_KEY)
    # Test provider-specific parameters for Anthropic
    run_provider_specific_params_test(ANTHROPIC_SONNET_MODEL, param_key="top_k", param_value=3)
    
    # --- DeepSeek Tests ---
    # Note: DeepSeek calls require sufficient balance on your DeepSeek account.
    print_section_header("DeepSeek Models")
    # For DeepSeek, instructor.from_litellm() is the standard path.
    run_function_calling_test(DEEPSEEK_CHAT_MODEL, client_provider_strategy="litellm")
    run_json_mode_test(DEEPSEEK_CHAT_MODEL)
    run_prefix_assistant_message_test(DEEPSEEK_CHAT_MODEL, api_key=DEEPSEEK_API_KEY)
    
    # --- Document Input Tests ---
    print_section_header("Document Input Tests")
    # We use ANTHROPIC_SONNET_MODEL here. If this exact model string isn't supported
    # by litellm.utils.supports_pdf_input or for PDF input directly,
    # this test might be skipped or fail.
    # For actual Bedrock/Vertex models, you'd use "bedrock/..." or "vertex_ai/..." model names
    # and ensure corresponding AWS/Google credentials are set.
    run_pdf_input_test(ANTHROPIC_SONNET_MODEL) # Example with Anthropic

    # --- Proxy Feature Notes ---
    print_section_header("Conceptual Notes on Proxy Features")
    run_conceptual_proxy_file_features_info()
    
    # --- Rerank Tests ---
    print_section_header("Rerank Tests")
    run_rerank_test()
    
    # --- Router Tests (SDK) ---
    print_section_header("Router Tests (SDK)")
    asyncio.run(run_router_basic_tests())
    asyncio.run(run_router_advanced_tests())

    print("\nLLM isolated tests finished.")

    print("Shutting down LiteLLM to clean up resources...")
    try:
        if hasattr(litellm, 'shutdown'):
            print("Shutting down LiteLLM to clean up resources...")
            litellm.shutdown()
        else:
            print("LiteLLM shutdown not available in this version.")
    except Exception as e:
        print(f"Note: Error during shutdown: {e}")
    print("LiteLLM shutdown complete.") #