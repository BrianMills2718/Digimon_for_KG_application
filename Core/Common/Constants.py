import os
from pathlib import Path

from loguru import logger

Process_tickers = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


Default_text_separator = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]

# def get_metagpt_package_root():
#     """Get the root directory of the installed package."""
#     package_root = Path(metagpt.__file__).parent.parent
#     for i in (".git", ".project_root", ".gitignore"):
#         if (package_root / i).exists():
#             break
#     else:
#         package_root = Path.cwd()

#     logger.info(f"Package root set to {str(package_root)}")
#     return package_root

def get_root():
    """Get the project root directory."""
    # Check if a project root is specified in the environment variable
    project_root_env = os.getenv("METAGPT_PROJECT_ROOT")
    if project_root_env:
        project_root = Path(project_root_env)
        logger.info(f"PROJECT_ROOT set from environment variable to {str(project_root)}")
    else:
        # Fallback to package root if no environment variable is set
        project_root = get_metagpt_package_root()
    return project_root

GRAPHRAG_ROOT = "test"
METAGPT_ROOT = 'test'  # Dependent on METAGPT_PROJECT_ROOT

CONFIG_ROOT = Path.home() / ".metagpt"


# Timeout
USE_CONFIG_TIMEOUT = 0  # Using llm.timeout configuration.
LLM_API_TIMEOUT = 300

# Split tokens
GRAPH_FIELD_SEP = "<SEP>"

DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"

IGNORED_MESSAGE_ID = "0"
