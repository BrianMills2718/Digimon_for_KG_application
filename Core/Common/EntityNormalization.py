"""Canonical normalization helpers for graph identities and extracted text."""

from __future__ import annotations

import html
import unicodedata
from typing import Any


def normalize_graph_text(value: Any) -> Any:
    """Clean extracted graph text without deleting Unicode semantics.

    HTML entities are decoded, Unicode is normalized with NFKC, control
    characters become spaces, and whitespace is collapsed. Punctuation and
    Unicode letters/symbols are otherwise preserved because descriptions and
    keywords are semantic evidence, not identifiers.
    """
    if not isinstance(value, str):
        return value

    text = unicodedata.normalize("NFKC", html.unescape(value.strip()))
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("C") and not char.isspace():
            cleaned.append(" ")
        else:
            cleaned.append(char)
    return " ".join("".join(cleaned).split())


def normalize_entity_id(value: Any) -> Any:
    """Return a stable case-insensitive Unicode graph identifier.

    Non-string values are returned unchanged. Unicode letters, numbers, and
    combining marks are preserved; punctuation/symbols become spaces; whitespace
    is collapsed. ASCII behavior remains equivalent to the old ``clean_str``
    identity path for ordinary names.
    """
    if not isinstance(value, str):
        return value

    text = normalize_graph_text(value).casefold()
    normalized = []
    for char in text:
        category = unicodedata.category(char)
        if category and category[0] in {"L", "N", "M"}:
            normalized.append(char)
        elif char.isspace():
            normalized.append(" ")
        else:
            normalized.append(" ")

    return " ".join("".join(normalized).split())
