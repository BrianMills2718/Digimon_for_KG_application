"""Canonical normalization for graph entity/relation identifiers.

Entity identity must not assume ASCII. DIGIMON datasets may contain accented,
Cyrillic, CJK, and other Unicode names; deleting those characters can collapse
unrelated entities or erase them entirely. This helper normalizes presentation
while preserving Unicode letters, numbers, and combining marks.
"""

from __future__ import annotations

import html
import unicodedata
from typing import Any


def normalize_entity_id(value: Any) -> Any:
    """Return a stable lowercase/casefolded Unicode graph identifier.

    Non-string values are returned unchanged for compatibility with legacy
    callers. For strings:
    - HTML entities are decoded;
    - Unicode is normalized with NFKC;
    - casefolding gives stable case-insensitive identity;
    - Unicode letters/numbers/marks are preserved;
    - punctuation, symbols, and controls become spaces;
    - whitespace is collapsed.

    ASCII behavior remains equivalent to the old ``clean_str`` identity path
    for ordinary names such as ``"Scott Derrickson"`` or ``"ACME, Inc."``.
    """
    if not isinstance(value, str):
        return value

    text = unicodedata.normalize("NFKC", html.unescape(value.strip())).casefold()
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
