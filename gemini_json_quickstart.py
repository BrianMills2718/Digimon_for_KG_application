import os
import json
from google import genai

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise EnvironmentError("Please set the GOOGLE_API_KEY environment variable.")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.5-flash-preview-04-17"

# --- Example 1: Constrained Output in Prompt ---
prompt = """
  List a few popular cookie recipes using this JSON schema:

  Recipe = {'recipe_name': str}
  Return: list[Recipe]
"""

raw_response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config={
        'response_mime_type': 'application/json'
    },
)

print("Prompt-based raw response:")
print(raw_response.text)
try:
    response = json.loads(raw_response.text)
    print(json.dumps(response, indent=4))
except Exception as e:
    print("Could not parse prompt-based response as JSON:", e)

# --- Example 2: Schema-based Output (Recommended for Gemini 1.5+) ---
try:
    import typing_extensions as typing
except ImportError:
    import typing as typing

class Recipe(typing.TypedDict):
    recipe_name: str
    recipe_description: str
    recipe_ingredients: list[str]

schema_prompt = "List a few imaginative cookie recipes along with a one-sentence description as if you were a gourmet restaurant and their main ingredients."

result = client.models.generate_content(
    model=MODEL_ID,
    contents=schema_prompt,
    config={
        'response_mime_type': 'application/json',
        'response_schema': list[Recipe],
    },
)

print("\nSchema-based raw response:")
print(result.text)
try:
    response2 = json.loads(result.text)
    print(json.dumps(response2, indent=4))
except Exception as e:
    print("Could not parse schema-based response as JSON:", e)