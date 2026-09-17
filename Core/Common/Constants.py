import os
from pathlib import Path

from loguru import logger
from enum import Enum

Process_tickers = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


Default_text_separator = [
    "\n\n",
    "\r\n\r\n",
    "\n",
    "\r\n",
    "。",
    "．",
    ".",
    "！",
    "!",
    "？",
    "?",
    " ",
    "\t",
    "\u3000",
    "\u200b",
]


def get_package_root():
    return Path.cwd()


def get_root():
    """Get the project root directory."""
    project_root_env = os.getenv("METAGPT_PROJECT_ROOT")
    if project_root_env:
        project_root = Path(project_root_env)
        logger.info(
            f"PROJECT_ROOT set from environment variable to {str(project_root)}"
        )
    else:
        project_root = get_package_root()
    return project_root


GRAPHRAG_ROOT = get_root()
CONFIG_ROOT = Path.home() / "Option"

USE_CONFIG_TIMEOUT = 0
LLM_API_TIMEOUT = 300

GRAPH_FIELD_SEP = "<SEP>"

DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"

IGNORED_MESSAGE_ID = "0"

MESSAGE_ROUTE_FROM = "sent_from"
MESSAGE_ROUTE_TO = "send_to"
MESSAGE_ROUTE_CAUSE_BY = "cause_by"
MESSAGE_META_ROLE = "role"
MESSAGE_ROUTE_TO_ALL = "<all>"
MESSAGE_ROUTE_TO_NONE = "<none>"

NODE_PATTERN = r"Node\(id='(.*?)', type='(.*?)'\)"
REL_PATTERN = r"Relationship\(subj=Node\(id='(.*?)', type='(.*?)'\), obj=Node\(id='(.*?)', type='(.*?)'\), type='(.*?)'\)"

# PassageGraph's WAT integration requires a service token. Keep credentials out
# of source control and let local/CI environments opt into this graph type.
GCUBE_TOKEN = os.getenv("GCUBE_TOKEN", "").strip()

hex_color = "#ea6eaf"
r = int(hex_color[1:3], 16)
g = int(hex_color[3:5], 16)
b = int(hex_color[5:7], 16)
ANSI_COLOR = f"\033[38;2;{r};{g};{b}m"
TOKEN_TO_CHAR_RATIO = 4


class Retriever(Enum):
    ENTITY = "entity"
    RELATION = "relationship"
    CHUNK = "chunk"
    COMMUNITY = "community"
    SUBGRAPH = "subgraph"
