"""Lightweight, corpus-agnostic query expansion for entity retrieval."""

import re
from typing import List, Set

from Core.Common.Logger import logger


class QueryExpander:
    """Generate small generic search variants without corpus-specific knowledge."""

    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "what", "who",
        "when", "where", "why", "how", "of", "to", "in", "for", "on",
        "at", "by", "and", "or", "with", "about",
    }

    QUERY_PATTERNS = (
        r"what (?:is|are) (.+?)(?:\?|$)",
        r"tell me about (.+?)(?:\?|$)",
        r"who (?:was|is|were) (.+?)(?:\?|$)",
        r"describe (.+?)(?:\?|$)",
    )

    def expand_query(self, query: str) -> List[str]:
        """Return generic variants while always preserving the original query."""
        query_lower = query.lower().strip()
        expanded_terms: Set[str] = {query_lower}

        subject = self._extract_subject(query_lower)
        if subject:
            expanded_terms.update(self._subject_variants(subject))

        words = re.findall(r"[a-z0-9][a-z0-9'-]*", query_lower)
        expanded_terms.update(
            word for word in words
            if word not in self.STOP_WORDS and len(word) > 2
        )

        result = sorted(expanded_terms, key=lambda value: (value != query_lower, value))
        logger.info(
            "Query expansion: '{}' -> {} generic term(s): {}",
            query,
            len(result),
            result[:10],
        )
        return result

    def _extract_subject(self, query: str) -> str | None:
        for pattern in self.QUERY_PATTERNS:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _subject_variants(subject: str) -> Set[str]:
        variants = {subject}
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", subject).strip()
        if cleaned:
            variants.add(cleaned)

        # Tiny morphology heuristic only; do not inject domain facts/synonyms.
        if cleaned and " " not in cleaned:
            if cleaned.endswith("s") and len(cleaned) > 3:
                variants.add(cleaned[:-1])
            elif len(cleaned) > 2:
                variants.add(cleaned + "s")
        return variants


query_expander = QueryExpander()
