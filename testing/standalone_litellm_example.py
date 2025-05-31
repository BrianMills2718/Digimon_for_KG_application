# START: /home/brian/digimon/testing/standalone_litellm_example.py
import os
import sys
import json
import asyncio
from typing import Optional, List
from pydantic import BaseModel, Field
from pathlib import Path

# --- Add project root to sys.path to allow imports from Option ---
# This assumes this script is in /home/brian/digimon/testing/
# If you run it from /home/brian/digimon using `python testing/standalone_litellm_example.py`
# then __file__ will be "testing/standalone_litellm_example.py" relative to /home/brian/digimon if not resolved.

# Let's try to be robust:
current_script_path = Path(__file__).resolve() # e.g., /home/brian/digimon/testing/standalone_litellm_example.py
project_root = current_script_path.parent.parent # Should be /home/brian/digimon

# Double check if 'Option' directory exists, if not, maybe we are already in project root
if not (project_root / "Option").is_dir():
    # This case could happen if the script is run from project root and __file__ is just the script name
    # or if it's not in the testing subfolder.
    # Let's try using current working directory if it seems like a project root.
    cwd_path = Path(os.getcwd()).resolve()
    if (cwd_path / "Option").is_dir() and (cwd_path / "Core").is_dir():
        project_root = cwd_path
    else: # Fallback if still not found, which might lead to import error for DigimonConfig
        print(f"WARNING: Could not confidently determine project root. Using {project_root}. 'Option' dir not found there.")


sys.path.insert(0, str(project_root))
print(f"DEBUG: Project root added to sys.path: {project_root}")
print(f"DEBUG: sys.path: {sys.path}")

try:
    from Option.Config2 import Config as DigimonConfig # Your project's Config class [cite: uploaded:wsl_digimon_copy_for_gemini2/Option/Config2.py]
    print("DEBUG: Successfully imported DigimonConfig from Option.Config2")
except ImportError as e:
    print(f"ERROR: Could not import DigimonConfig from Option.Config2. Error: {e}")
    print("Please ensure standalone_litellm_example.py is in the 'testing' directory,")
    print("or adjust the project_root path calculation at the top of this script.")
    DigimonConfig = None
except Exception as e_general:
     print(f"ERROR: An unexpected error occurred during DigimonConfig import: {e_general}")
     DigimonConfig = None


# Make sure to install litellm and instructor:
# pip install litellm instructor PyYAML
import litellm
import instructor # For Pydantic model parsing
import yaml # For loading Config2.yaml if DigimonConfig loader fails or for direct load

# --- Configuration ---
# Optional: Enable verbose logging from LiteLLM for debugging
litellm.set_verbose = True
# litellm.drop_params = True # If models complain about unsupported params like response_format

# --- API Keys (Provided by User for this Standalone Example) ---
# These will be used if not found in Config2.yaml or for non-OpenAI models
USER_PROVIDED_OPENAI_API_KEY = "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"
USER_PROVIDED_ANTHROPIC_API_KEY = "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"
USER_PROVIDED_GEMINI_API_KEY = "AIzaSyDXaLhSWAQhGNHZqdbvY-qFB0jxyPbiiow"

# Function to load DigimonConfig (attempt)
def load_digimon_config():
    if not DigimonConfig:
        print("WARNING: DigimonConfig class not loaded. Cannot load from Config2.yaml using it.")
        return None
    try:
        # Assuming Option/Config2.yaml is relative to project_root
        config_path = project_root / "Option" / "Config2.yaml"
        if config_path.exists():
            print(f"Attempting to load config from: {config_path}")
            # Using the default() method from your Config2.py which loads the YAML
            # This might print its own INFO/WARNING messages.
            config = DigimonConfig.default()
            print(f"Successfully loaded DigimonConfig. LLM API Key from config: {'*' * 5 + config.llm.api_key[-5:] if config.llm and config.llm.api_key else 'Not found'}")
            return config
        else:
            print(f"WARNING: {config_path} not found. Cannot load API keys from project config.")
            return None
    except Exception as e:
        print(f"ERROR loading DigimonConfig: {e}")
        return None

# --- 1. Basic Text Completion Example (OpenAI from Config) ---
def basic_openai_completion(config: Optional[DigimonConfig]):
    print("\n--- 1. Basic Text Completion Example (OpenAI from Config) ---")
    if not (config and config.llm and config.llm.api_key):
        print("OpenAI API key not found in loaded config. Using user-provided key for this example.")
        api_key_to_use = USER_PROVIDED_OPENAI_API_KEY
        model_to_use = "gpt-3.5-turbo" # Fallback model
    else:
        api_key_to_use = config.llm.api_key
        model_to_use = config.llm.model or "gpt-3.5-turbo"
        print(f"Using OpenAI model: {model_to_use} from config.")

    try:
        response = litellm.completion(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            api_key=api_key_to_use,
            temperature=0.1,
            max_tokens=50
        )
        content = response.choices[0].message.content
        print(f"LLM Response (OpenAI): {content}")
    except Exception as e:
        print(f"Error in basic_openai_completion: {e}")

# --- 2. Anthropic (Claude) Example ---
def anthropic_claude_completion():
    print("\n--- 2. Anthropic (Claude) Example ---")
    if not USER_PROVIDED_ANTHROPIC_API_KEY:
        print("Anthropic API key not provided. Skipping example.")
        return
    try:
        response = litellm.completion(
            model="claude-3-haiku-20240307", # Or "claude-3-sonnet-20240229", "claude-3-opus-20240229"
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
    if not USER_PROVIDED_GEMINI_API_KEY:
        print("Gemini API key not provided. Skipping example.")
        return
    try:
        # LiteLLM often prefers GOOGLE_API_KEY env var for Gemini.
        # We can set it temporarily for this call if not already set.
        original_google_key = os.environ.get('GOOGLE_API_KEY')
        os.environ['GOOGLE_API_KEY'] = USER_PROVIDED_GEMINI_API_KEY
        print(f"Temporarily set GOOGLE_API_KEY for Gemini call.")

        response = litellm.completion(
            model="gemini/gemini-1.5-flash-latest", # or "gemini-pro" etc.
            messages=[{"role": "user", "content": "What are three interesting facts about the planet Mars?"}],
            # api_key=USER_PROVIDED_GEMINI_API_KEY, # Usually not needed if GOOGLE_API_KEY is set
            temperature=0.3,
            max_tokens=250
        )
        content = response.choices[0].message.content
        print(f"LLM Response (Gemini): {content}")

        # Restore original GOOGLE_API_KEY if it was set
        if original_google_key:
            os.environ['GOOGLE_API_KEY'] = original_google_key
        else:
            del os.environ['GOOGLE_API_KEY'] # Clean up if we set it

    except Exception as e:
        print(f"Error in google_gemini_completion: {e}")
        # Restore env var in case of error too
        if 'original_google_key' in locals(): # Check if it was defined
             if original_google_key:
                 os.environ['GOOGLE_API_KEY'] = original_google_key
             elif 'GOOGLE_API_KEY' in os.environ: # Only delete if we might have set it
                 del os.environ['GOOGLE_API_KEY']


# --- 4. Pydantic Model with Instructor (using OpenAI from Config) ---
class PlanStep(BaseModel):
    step_id: str = Field(description="A unique identifier for this plan step.")
    tool_id: str = Field(description="The ID of the tool to be executed in this step.")
    reasoning: str = Field(description="Brief reasoning for choosing this tool and parameters.")

class GeneratedPlan(BaseModel):
    plan_id: str = Field(description="A unique ID for the overall plan.")
    user_query: str = Field(description="The original user query.")
    steps: List[PlanStep] = Field(description="A list of steps to execute.")

def pydantic_with_instructor_openai(config: Optional[DigimonConfig]):
    print("\n--- 4. Pydantic Model with Instructor (OpenAI from Config) ---")
    if not (config and config.llm and config.llm.api_key):
        print("OpenAI API key not found in loaded config for Instructor. Using user-provided key.")
        api_key_to_use = USER_PROVIDED_OPENAI_API_KEY
        # Use a capable model for instructor, gpt-4o or gpt-4-turbo if available
        model_to_use = "gpt-4o" # Fallback to gpt-4o if config model not suitable
    else:
        api_key_to_use = config.llm.api_key
        # Prefer more capable models for instructor if available, otherwise use config model
        model_to_use = config.llm.model if "gpt-4" in config.llm.model else "gpt-4o"
        print(f"Using OpenAI model: {model_to_use} from config (or overridden for capability) for Instructor.")


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
            temperature=0.0
        )
        print(f"LLM Response (Pydantic Object - GeneratedPlan):\n{response_pydantic.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error in pydantic_with_instructor_openai: {e}")

# --- 5. Asynchronous Text Completion Example (OpenAI from Config) ---

# --- 5b. Asynchronous Pydantic Model with Instructor ---
class UserDetails(BaseModel):
    name: str = Field(description="The user's name.")
    age: int = Field(description="The user's age.")
    occupation: str = Field(description="The user's occupation.")
    location: str = Field(description="The user's location.")
    interests: list[str] = Field(description="A list of the user's interests.")

async def async_pydantic_with_instructor(config: Optional[DigimonConfig]):
    print("\n--- 5. Asynchronous Pydantic Model with Instructor ---")

    if not (config and config.llm and config.llm.api_key):
        print("OpenAI API key not found in loaded config for Async Instructor. Using user-provided key.")
        api_key_to_use = USER_PROVIDED_OPENAI_API_KEY
        model_to_use = "gpt-4o" # Capable model for instructor
    else:
        api_key_to_use = config.llm.api_key
        model_to_use = config.llm.model if "gpt-4" in config.llm.model else "gpt-4o" # Prefer capable model
        print(f"Using OpenAI model: {model_to_use} from config (or overridden) for Async Instructor.")

    try:
        aclient = instructor.from_litellm(litellm.acompletion)

        response_pydantic: UserDetails = await aclient.chat.completions.create(
            model=model_to_use, # USE DYNAMIC MODEL
            response_model=UserDetails,
            messages=[
                {"role": "user", "content": "User info: Async Alice, 27, cloud engineer in Seattle, loves hiking and serverless."}
            ],
            api_key=api_key_to_use, # PASS API KEY
            max_retries=2,
            temperature=0.0
        )
        print(f"LLM Response (Async Pydantic Object): {response_pydantic}")
        print(f"Name: {response_pydantic.name}, Age: {response_pydantic.age}, Interests: {response_pydantic.interests}")
    except Exception as e:
        print(f"Error in async_pydantic_with_instructor: {e}")

async def async_openai_completion(config: Optional[DigimonConfig]):
    print("\n--- 5. Asynchronous Text Completion Example (OpenAI from Config) ---")
    if not (config and config.llm and config.llm.api_key):
        print("OpenAI API key not found in loaded config for async. Using user-provided key.")
        api_key_to_use = USER_PROVIDED_OPENAI_API_KEY
        model_to_use = "gpt-3.5-turbo"
    else:
        api_key_to_use = config.llm.api_key
        model_to_use = config.llm.model or "gpt-3.5-turbo"
        print(f"Using OpenAI model: {model_to_use} from config for async.")

    try:
        response = await litellm.acompletion(
            model=model_to_use,
            messages=[
                {"role": "user", "content": "Explain async/await in Python in one sentence."}
            ],
            api_key=api_key_to_use,
            temperature=0.1,
            max_tokens=100
        )
        content = response.choices[0].message.content
        print(f"LLM Response (Async OpenAI): {content}")
    except Exception as e:
        print(f"Error in async_openai_completion: {e}")

async def main_async_examples(digimon_cfg):
    await async_openai_completion(config=digimon_cfg)
    await async_pydantic_with_instructor(config=digimon_cfg)  # Added as per instructions

if __name__ == "__main__":
    print("Running LiteLLM Standalone Examples...")
    
    loaded_config = load_digimon_config()

    # Synchronous Examples
    basic_openai_completion(config=loaded_config)
    anthropic_claude_completion()
    google_gemini_completion()
    pydantic_with_instructor_openai(config=loaded_config)
    
    # Asynchronous Examples
    print("\nRunning Asynchronous LiteLLM Examples...")
    asyncio.run(main_async_examples(digimon_cfg=loaded_config))

    print("\n--- Examples Finished ---")
# END: /home/brian/digimon/testing/standalone_litellm_example.py