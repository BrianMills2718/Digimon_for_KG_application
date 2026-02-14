"""Meta: LLM entity extraction operator.

Use LLM to extract entity mentions from a query text.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def meta_extract_entities(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": SlotValue(QUERY_TEXT)}
    Outputs: {"entities": SlotValue(ENTITY_SET)}
    """
    query = inputs["query"].data

    try:
        prompt = (
            "Extract all named entities from the following question. "
            "Return a JSON list of strings.\n\n"
            f"Question: {query}\n\n"
            "Entities (JSON list):"
        )
        result = await ctx.llm.aask(msg=[{"role": "user", "content": prompt}])

        # Parse JSON list from response
        # Try to find a JSON array in the response
        import re
        match = re.search(r"\[.*?\]", result, re.DOTALL)
        if match:
            names = json.loads(match.group())
        else:
            # Fallback: split by commas
            names = [n.strip().strip('"').strip("'") for n in result.split(",")]

        records = [EntityRecord(entity_name=n) for n in names if n.strip()]
        return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=records, producer="meta.extract_entities")}

    except Exception as e:
        logger.exception(f"meta_extract_entities failed: {e}")
        return {"entities": SlotValue(kind=SlotKind.ENTITY_SET, data=[], producer="meta.extract_entities")}
