# START: /home/brian/digimon/testing/standalone_litellm_example.py
import os
import sys
import json
import asyncio
from typing import Optional, List
from pydantic import BaseModel, Field
from pathlib import Path

# --- Add project root to sys.path ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
if not (project_root / "Option").is_dir():
    cwd_path = Path(os.getcwd()).resolve()
    if (cwd_path / "Option").is_dir() and (cwd_path / "Core").is_dir():
        project_root = cwd_path
    else:
        print(f"WARNING: Could not confidently determine project root. Using {project_root}. 'Option' dir not found there.")
sys.path.insert(0, str(project_root))
print(f"DEBUG: Project root added to sys.path: {project_root}")

try:
    from Option.Config2 import Config as DigimonConfig
    print("DEBUG: Successfully imported DigimonConfig from Option.Config2")
except ImportError as e:
    print(f"ERROR: Could not import DigimonConfig from Option.Config2. Error: {e}")
    DigimonConfig = None
except Exception as e_general:
     print(f"ERROR: An unexpected error occurred during DigimonConfig import: {e_general}")
     DigimonConfig = None

import litellm
import instructor
import yaml

# --- LiteLLM Global Settings ---
# As suggested by your IDE and LiteLLM for o-series models / general robustness
litellm.drop_params = True 
print("INFO: litellm.drop_params set to True")

# Optional: Enable verbose logging from LiteLLM for debugging
# This is the new way to set verbose logs if needed:
# os.environ['LITELLM_LOG'] = 'DEBUG' 
# litellm.set_verbose = True # Deprecated, but some old versions might still respond

# --- API Keys (Provided by User for this Standalone Example) ---
# These will be used if not found in Config2.yaml or for non-OpenAI models
USER_PROVIDED_OPENAI_API_KEY = "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"
USER_PROVIDED_ANTHROPIC_API_KEY = "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"
USER_PROVIDED_GEMINI_API_KEY = "AIzaSyDXaLhSWAQhGNHZqdbvY-qFB0jxyPbiiow"
#USER_PROVIDED_GEMINI_API_KEY = "AIzaSyC1oX-7QsJXgQE2EgJNXP4vgWZ-_396yHg"

# Function to load DigimonConfig
def load_digimon_config():
    # (Keep your existing load_digimon_config function as it was, it's working)
    if not DigimonConfig:
        print("WARNING: DigimonConfig class not loaded. Cannot load from Config2.yaml using it.")
        return None
    try:
        config_path = project_root / "Option" / "Config2.yaml"
        if config_path.exists():
            print(f"Attempting to load config from: {config_path}")
            config = DigimonConfig.default()
            print(f"Successfully loaded DigimonConfig. LLM API Key from config: {'*' * 5 + config.llm.api_key[-5:] if config.llm and config.llm.api_key and config.llm.api_key != 'sk-' else 'Not found or placeholder'}")
            return config
        else:
            print(f"WARNING: {config_path} not found. Cannot load API keys from project config.")
            return None
    except Exception as e:
        print(f"ERROR loading DigimonConfig: {e}", exc_info=True)
        return None

# --- 1. Basic Text Completion Example (OpenAI o4-mini with temp 0.0 from Config) ---
def basic_openai_o4mini_completion(config: Optional[DigimonConfig]):
    print("\n--- 1. Basic Text Completion Example (OpenAI o4-mini with temp 0.0 from Config) ---")
    model_to_use = "openai/o4-mini-2025-04-16" # Specify o4-mini
    api_key_to_use = None
    temperature_to_use = 0.0 # Test case for o4-mini
    max_tokens_to_use = 50

    if config and config.llm:
        print(f"Configuring from DigimonConfig for model: {model_to_use}")
        api_key_to_use = config.llm.api_key if config.llm.api_key and config.llm.api_key != "sk-" else None
        # We want to test o4-mini with temperature 0.0, assuming litellm.drop_params handles it
        # or the config itself for o4-mini would be set to temp 1.0
        # For this test, let's explicitly use temp 0.0 if config has it for a different model,
        # otherwise default to 0.0 to demonstrate the o4-mini specific behavior.
        temperature_to_use = config.llm.temperature if hasattr(config.llm, 'temperature') else 0.0
        max_tokens_to_use = config.llm.max_token if hasattr(config.llm, 'max_token') else 50
        
        if model_to_use == "openai/o4-mini-2025-04-16" and temperature_to_use != 1.0:
            print(f"INFO: Testing {model_to_use} with configured/default temperature {temperature_to_use}. Expecting litellm.drop_params=True to handle if 0.0, or model API to correct/default to 1.0.")
    
    if not api_key_to_use:
        print("API key not found in config or is placeholder. Using USER_PROVIDED_OPENAI_API_KEY.")
        api_key_to_use = USER_PROVIDED_OPENAI_API_KEY
    
    print(f"Using Model: {model_to_use}, Temperature: {temperature_to_use}, Max Tokens: {max_tokens_to_use}")

    try:
        response = litellm.completion(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            api_key=api_key_to_use,
            temperature=temperature_to_use,
            max_tokens=max_tokens_to_use
        )
        content = response.choices[0].message.content
        print(f"LLM Response (OpenAI {model_to_use}): {content}")
        print(f"Usage from response: {response.usage}")
    except Exception as e:
        print(f"Error in basic_openai_o4mini_completion: {e}")

# --- (Keep Anthropic and Gemini examples as they were, they use direct keys) ---
# Make sure USER_PROVIDED_ANTHROPIC_API_KEY and USER_PROVIDED_GEMINI_API_KEY are filled in.

# --- 2. Anthropic (Claude) Example ---
def anthropic_claude_completion():
    print("\n--- 2. Anthropic (Claude) Example ---")
    # ... (no changes needed here if USER_PROVIDED_ANTHROPIC_API_KEY is set correctly) ...
    if not USER_PROVIDED_ANTHROPIC_API_KEY:
        print("Anthropic API key not provided. Skipping example.")
        return
    try:
        response = litellm.completion(
            model="claude-3-haiku-20240307", 
            messages=[{"role": "user", "content": "Explain the concept of a black hole in simple terms."}],
            api_key=USER_PROVIDED_ANTHROPIC_API_KEY,
            temperature=0.2,
            max_tokens=200
        )
        content = response.choices[0].message.content
        print(f"LLM Response (Claude): {content}")
    except Exception as e:
        print(f"Error in anthropic_claude_completion: {e}")


# --- 3. Google Gemini Example ---
def google_gemini_completion():
    print("\n--- 3. Google Gemini Example ---")
    if not USER_PROVIDED_GEMINI_API_KEY or USER_PROVIDED_GEMINI_API_KEY == "AIzaSyDXaLhSWAQhGNHZqdbvY-qFB0jxyPbiiow":
        print(f"Using Gemini API Key: {'*' * 5 + USER_PROVIDED_GEMINI_API_KEY[-5:]}")
    else:
        print("Actual Gemini API key is not shown here for brevity but should be set in USER_PROVIDED_GEMINI_API_KEY")

    if not USER_PROVIDED_GEMINI_API_KEY:
        print("Gemini API key (USER_PROVIDED_GEMINI_API_KEY) is empty. Skipping example.")
        return

    original_gemini_key_env = os.environ.get('GEMINI_API_KEY')
    original_google_key_env = os.environ.get('GOOGLE_API_KEY')

    try:
        os.environ['GEMINI_API_KEY'] = USER_PROVIDED_GEMINI_API_KEY
        os.environ['GOOGLE_API_KEY'] = USER_PROVIDED_GEMINI_API_KEY
        print(f"Attempting Gemini call with model: gemini/gemini-2.0-flash")
        print(f"Set os.environ['GEMINI_API_KEY'] and os.environ['GOOGLE_API_KEY']")

        response = litellm.completion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": "What are three interesting facts about the planet Mars?"}],
            api_key=USER_PROVIDED_GEMINI_API_KEY,
            temperature=0.3,
            max_tokens=250
        )
        content = response.choices[0].message.content
        print(f"LLM Response (Gemini): {content}")

    except Exception as e:
        print(f"Error in google_gemini_completion: {e}")
    finally:
        if original_gemini_key_env:
            os.environ['GEMINI_API_KEY'] = original_gemini_key_env
        elif 'GEMINI_API_KEY' in os.environ:
            del os.environ['GEMINI_API_KEY']
        
        if original_google_key_env:
            os.environ['GOOGLE_API_KEY'] = original_google_key_env
        elif 'GOOGLE_API_KEY' in os.environ:
            del os.environ['GOOGLE_API_KEY']
        print("Restored original GEMINI_API_KEY/GOOGLE_API_KEY environment variables if they existed.")

# --- 4. Pydantic Model with Instructor (using OpenAI from Config) ---
# (Keep pydantic_with_instructor_openai and its Pydantic models as they were,
#  it should now use the config correctly, and we can see if drop_params affects it)
class PlanStep(BaseModel): # Ensure BaseModel is imported from pydantic
    step_id: str = Field(description="A unique identifier for this plan step.")
    tool_id: str = Field(description="The ID of the tool to be executed in this step.")
    reasoning: str = Field(description="Brief reasoning for choosing this tool and parameters.")

class GeneratedPlan(BaseModel):
    plan_id: str = Field(description="A unique ID for the overall plan.")
    user_query: str = Field(description="The original user query.")
    steps: List[PlanStep] = Field(description="A list of steps to execute.")
    
def pydantic_with_instructor_openai(config: Optional[DigimonConfig]):
    print("\n--- 4. Pydantic Model with Instructor (OpenAI from Config) ---")
    # ... (This function was mostly fine, ensure it uses a capable model like gpt-4o for instructor)
    # Let's use o4-mini here as well to test its behavior with instructor and drop_params
    model_to_use = "openai/o4-mini-2025-04-16"
    api_key_to_use = None
    temperature_to_use = 0.0 # Test o4-mini behavior

    if config and config.llm:
        print(f"Configuring from DigimonConfig for Instructor with model: {model_to_use}")
        api_key_to_use = config.llm.api_key if config.llm.api_key and config.llm.api_key != "sk-" else None
        # If config has a different model specified and it's gpt-4, prefer it for instructor
        if "gpt-4" in (config.llm.model or ""):
            model_to_use = config.llm.model
        temperature_to_use = config.llm.temperature if hasattr(config.llm, 'temperature') else 0.0
        
        if model_to_use == "openai/o4-mini-2025-04-16" and temperature_to_use != 1.0:
             print(f"INFO: Testing Instructor with {model_to_use} and configured/default temperature {temperature_to_use}. Expecting litellm.drop_params=True to handle.")

    if not api_key_to_use:
        print("API key not found in config or is placeholder for Instructor. Using USER_PROVIDED_OPENAI_API_KEY.")
        api_key_to_use = USER_PROVIDED_OPENAI_API_KEY
        if "gpt-4" not in model_to_use: # Ensure a capable model if falling back
             model_to_use = "gpt-4o" # Fallback to gpt-4o for instructor

    print(f"Using Model: {model_to_use}, Temperature: {temperature_to_use} for Instructor.")
    try:
        instructor_client = instructor.from_litellm(litellm.completion)
        response_pydantic: GeneratedPlan = instructor_client.chat.completions.create(
            model=model_to_use,
            response_model=GeneratedPlan,
            messages=[
                {"role": "system", "content": "You are a planning agent. Create a plan based on the user query."},
                {"role": "user", "content": "Find entities related to 'solar power' and then get their one-hop neighbors."}
            ],
            api_key=api_key_to_use,
            max_retries=2,
            temperature=temperature_to_use # Test with 0.0 for o4-mini
        )
        print(f"LLM Response (Pydantic Object - GeneratedPlan):\n{response_pydantic.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error in pydantic_with_instructor_openai: {e}")


# --- 5. Asynchronous Examples (Keep async_openai_completion and async_pydantic_with_instructor as they were,
# ensuring they also use api_key from config or fallback, and test o4-mini behavior)

async def async_openai_completion(config: Optional[DigimonConfig]):
    print("\n--- 5. Asynchronous Text Completion Example (OpenAI o4-mini from Config) ---")
    # ... (similar logic as basic_openai_o4mini_completion for model, key, temp)
    model_to_use = "openai/o4-mini-2025-04-16"
    api_key_to_use = None
    temperature_to_use = 0.0
    max_tokens_to_use = 100

    if config and config.llm:
        print(f"Configuring async from DigimonConfig for model: {model_to_use}")
        api_key_to_use = config.llm.api_key if config.llm.api_key and config.llm.api_key != "sk-" else None
        temperature_to_use = config.llm.temperature if hasattr(config.llm, 'temperature') else 0.0
        max_tokens_to_use = config.llm.max_token if hasattr(config.llm, 'max_token') else 100
        if model_to_use == "openai/o4-mini-2025-04-16" and temperature_to_use != 1.0:
            print(f"INFO: Testing async {model_to_use} with configured/default temperature {temperature_to_use}.")
    
    if not api_key_to_use:
        print("API key not found in async config or is placeholder. Using USER_PROVIDED_OPENAI_API_KEY.")
        api_key_to_use = USER_PROVIDED_OPENAI_API_KEY
        
    print(f"Using Async Model: {model_to_use}, Temperature: {temperature_to_use}, Max Tokens: {max_tokens_to_use}")
    try:
        response = await litellm.acompletion(
            model=model_to_use,
            messages=[
                {"role": "user", "content": "Explain async/await in Python in one sentence."}
            ],
            api_key=api_key_to_use,
            temperature=temperature_to_use,
            max_tokens=max_tokens_to_use
        )
        content = response.choices[0].message.content
        print(f"LLM Response (Async OpenAI {model_to_use}): {content}")
    except Exception as e:
        print(f"Error in async_openai_completion: {e}")

class UserDetails(BaseModel): # Ensure Pydantic models are defined before use in async instructor
     name: str = Field(description="The user's name.")
     age: int = Field(description="The user's age.")
     occupation: Optional[str] = Field(None, description="The user's occupation.") # Made optional
     location: Optional[str] = Field(None, description="The user's location.") # Made optional
     interests: List[str] = Field(default_factory=list, description="A list of the user's interests.")


async def async_pydantic_with_instructor(config: Optional[DigimonConfig]):
    print("\n--- 6. Asynchronous Pydantic Model with Instructor (OpenAI o4-mini from Config) ---")
    # ... (similar logic as pydantic_with_instructor_openai for model, key, temp)
    model_to_use = "openai/o4-mini-2025-04-16"
    api_key_to_use = None
    temperature_to_use = 0.0

    if config and config.llm:
        print(f"Configuring async instructor from DigimonConfig for model: {model_to_use}")
        api_key_to_use = config.llm.api_key if config.llm.api_key and config.llm.api_key != "sk-" else None
        if "gpt-4" in (config.llm.model or ""): # Prefer more capable model for instructor
             model_to_use = config.llm.model 
        temperature_to_use = config.llm.temperature if hasattr(config.llm, 'temperature') else 0.0
        if model_to_use == "openai/o4-mini-2025-04-16" and temperature_to_use != 1.0:
            print(f"INFO: Testing async instructor with {model_to_use} and configured/default temperature {temperature_to_use}.")

    if not api_key_to_use:
        print("API key not found in async instructor config or is placeholder. Using USER_PROVIDED_OPENAI_API_KEY.")
        api_key_to_use = USER_PROVIDED_OPENAI_API_KEY
        if "gpt-4" not in model_to_use and "o4-mini" not in model_to_use :
             model_to_use = "gpt-4o" # Fallback for instructor

    print(f"Using Async Instructor Model: {model_to_use}, Temperature: {temperature_to_use}")
    try:
        aclient = instructor.from_litellm(litellm.acompletion) 
        response_pydantic: UserDetails = await aclient.chat.completions.create(
            model=model_to_use,
            response_model=UserDetails,
            messages=[
                {"role": "user", "content": "User info: Async Alice, 27, cloud engineer in Seattle, loves hiking and serverless."}
            ],
            api_key=api_key_to_use,
            max_retries=2,
            temperature=temperature_to_use
        )
        print(f"LLM Response (Async Pydantic Object UserDetails): {response_pydantic.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error in async_pydantic_with_instructor: {e}")

async def main_async_examples(digimon_cfg):
    await async_openai_completion(config=digimon_cfg)
    await async_pydantic_with_instructor(config=digimon_cfg)

if __name__ == "__main__":
    print("Running LiteLLM Standalone Examples (v2)...")
    
    loaded_config = load_digimon_config()

    # Test OpenAI o4-mini with potentially problematic temperature
    basic_openai_o4mini_completion(config=loaded_config)
    
    anthropic_claude_completion()
    google_gemini_completion() # This will likely still fail if the key is the issue
    
    pydantic_with_instructor_openai(config=loaded_config) # Test with o4-mini / configured model
    
    print("\nRunning Asynchronous LiteLLM Examples (v2)...")
    asyncio.run(main_async_examples(digimon_cfg=loaded_config))

    print("\n--- Examples Finished (v2) ---")
# END: /home/brian/digimon/testing/standalone_litellm_example.py