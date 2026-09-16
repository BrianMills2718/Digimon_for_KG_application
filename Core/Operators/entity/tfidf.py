"""Entity TF-IDF ranking operator.

Uses scikit-learn directly so the maintained operator does not depend on the
legacy LlamaIndex TFIDFStore implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


async def entity_tfidf(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": QUERY_TEXT, "entities"?: ENTITY_SET}
    Outputs: {"entities": ENTITY_SET}
    Params:  {"top_k": int}
    """
    query = inputs["query"].data
    seed = inputs.get("entities")
    top_k = (params or {}).get("top_k", ctx.config.top_k)

    try:
        if seed and seed.data:
            candidates = seed.data
            names = [record.entity_name for record in candidates]
            descriptions = [
                " ".join(
                    part
                    for part in (
                        record.entity_name,
                        record.entity_type,
                        record.description,
                    )
                    if part
                )
                for record in candidates
            ]
            source_ids = [record.source_id for record in candidates]
            entity_types = [record.entity_type for record in candidates]
        else:
            names = list(await ctx.graph.get_nodes())
            node_data = [await ctx.graph.get_node(name) for name in names]
            descriptions = []
            source_ids = []
            entity_types = []
            for name, data in zip(names, node_data):
                data = data or {}
                entity_type = data.get("entity_type", "")
                description = data.get("description", "")
                descriptions.append(
                    " ".join(
                        part for part in (str(name), entity_type, description) if part
                    )
                )
                source_ids.append(data.get("source_id", ""))
                entity_types.append(entity_type)

        if not names:
            return {
                "entities": SlotValue(
                    kind=SlotKind.ENTITY_SET,
                    data=[],
                    producer="entity.tfidf",
                )
            }

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(descriptions)
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, matrix).reshape(-1)

        ranked_indices = similarities.argsort()[::-1][: min(top_k, len(names))]
        records = [
            EntityRecord(
                entity_name=str(names[index]),
                source_id=source_ids[index],
                entity_type=entity_types[index],
                description=descriptions[index],
                score=float(similarities[index]),
                extra={"tfidf_rank": rank, "candidate_index": int(index)},
            )
            for rank, index in enumerate(ranked_indices)
        ]

        return {
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=records,
                producer="entity.tfidf",
            )
        }
    except Exception as exc:
        logger.exception(f"entity_tfidf failed: {exc}")
        return {
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[],
                producer="entity.tfidf",
            )
        }
