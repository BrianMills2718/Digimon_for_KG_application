"""Meta: LLM entity extraction operator."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1 :] if first_newline >= 0 else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _names_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        for key in ("entities", "named_entities", "entity_names"):
            if key in payload:
                return _names_from_payload(payload[key])
        return []

    if not isinstance(payload, list):
        return []

    names = []
    for item in payload:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(
                item.get("entity_name")
                or item.get("name")
                or item.get("entity")
                or ""
            ).strip()
        else:
            name = ""
        if name and name not in names:
            names.append(name)
    return names


def parse_entity_names(response: Any) -> list[str]:
    """Parse common structured LLM entity-extraction responses."""
    if isinstance(response, (list, dict)):
        return _names_from_payload(response)

    text = _strip_fences(str(response or ""))
    if not text:
        return []

    # Preferred path: complete JSON response.
    try:
        names = _names_from_payload(json.loads(text))
        if names:
            return names
    except json.JSONDecodeError:
        pass

    # Tolerate explanatory text around a JSON array/object.
    for pattern in (r"\[[\s\S]*?\]", r"\{[\s\S]*?\}"):
        for match in re.finditer(pattern, text):
            try:
                names = _names_from_payload(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
            if names:
                return names

    # Final conservative fallback: quoted strings only. Do not comma-split
    # arbitrary prose into fake entities.
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    return list(dict.fromkeys(value.strip() for value in quoted if value.strip()))


async def meta_extract_entities(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": QUERY_TEXT}
    Outputs: {"entities": ENTITY_SET}
    """
    query = inputs["query"].data

    try:
        prompt = (
            "Extract all named entities from the following question. "
            "Return a JSON list of strings.\n\n"
            f"Question: {query}\n\n"
            "Entities (JSON list):"
        )
        response = await ctx.llm.aask(
            msg=[{"role": "user", "content": prompt}]
        )
        names = parse_entity_names(response)
        if not names:
            logger.warning(
                f"meta.extract_entities produced no parseable entities for query: {query!r}"
            )

        records = [EntityRecord(entity_name=name) for name in names]
        return {
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=records,
                producer="meta.extract_entities",
            )
        }

    except Exception as exc:
        logger.exception(f"meta_extract_entities failed: {exc}")
        return {
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[],
                producer="meta.extract_entities",
                metadata={"error": str(exc)},
            )
        }
