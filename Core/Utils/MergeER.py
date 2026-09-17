from collections import Counter
from typing import List

from Core.Common.Constants import GRAPH_FIELD_SEP


def _merge_unique_strings(*collections) -> list[str]:
    """Merge provenance-like string collections deterministically."""
    values = {
        str(value)
        for collection in collections
        for value in (collection or [])
        if value is not None and str(value) != ""
    }
    return sorted(values)


class MergeEntity:
    merge_keys = ["source_id", "entity_type", "description"]
    merge_function = None

    @staticmethod
    def merge_source_ids(existing_source_ids: List[str], new_source_ids):
        return GRAPH_FIELD_SEP.join(
            _merge_unique_strings(existing_source_ids, new_source_ids)
        )

    @staticmethod
    def merge_types(existing_entity_types: List[str], new_entity_types):
        merged_entity_types = existing_entity_types + new_entity_types
        entity_type_counts = Counter(merged_entity_types)
        return entity_type_counts.most_common(1)[0][0] if entity_type_counts else ""

    @staticmethod
    def merge_descriptions(entity_relationships: List[str], new_descriptions):
        return GRAPH_FIELD_SEP.join(
            _merge_unique_strings(entity_relationships, new_descriptions)
        )

    @classmethod
    async def merge_info(cls, merge_keys, nodes_data, merge_dict):
        """Legacy tuple-style merge helper retained for compatibility."""
        if len(nodes_data) == 0:
            return []
        if cls.merge_function is None:
            cls.merge_function = {
                "source_id": cls.merge_source_ids,
                "entity_type": cls.merge_types,
                "description": cls.merge_descriptions,
            }

        result = []
        for merge_key in merge_keys:
            if merge_key in merge_dict and merge_key in cls.merge_function:
                result.append(
                    cls.merge_function[merge_key](
                        nodes_data.get(merge_key, []),
                        merge_dict.get(merge_key, []),
                    )
                )
            else:
                result.append("")
        return tuple(result)


class MergeRelationship:
    merge_keys = ["source_id", "weight", "description", "keywords", "relation_name"]
    merge_function = None

    @staticmethod
    def merge_weight(merge_weight, new_weight):
        return sum((new_weight or []) + (merge_weight or []))

    @staticmethod
    def merge_descriptions(entity_relationships, new_descriptions):
        return GRAPH_FIELD_SEP.join(
            _merge_unique_strings(entity_relationships, new_descriptions)
        )

    @staticmethod
    def merge_source_ids(existing_source_ids: List[str], new_source_ids):
        return GRAPH_FIELD_SEP.join(
            _merge_unique_strings(existing_source_ids, new_source_ids)
        )

    @staticmethod
    def merge_keywords(keywords: List[str], new_keywords):
        return GRAPH_FIELD_SEP.join(
            _merge_unique_strings(keywords, new_keywords)
        )

    @staticmethod
    def merge_relation_name(relation_name, new_relation_name):
        return GRAPH_FIELD_SEP.join(
            _merge_unique_strings(relation_name, new_relation_name)
        )

    @classmethod
    async def merge_info(cls, edges_data, merge_dict):
        """Legacy tuple-style relationship merge helper retained for compatibility."""
        if len(edges_data) == 0:
            return []
        if cls.merge_function is None:
            cls.merge_function = {
                "weight": cls.merge_weight,
                "description": cls.merge_descriptions,
                "source_id": cls.merge_source_ids,
                "keywords": cls.merge_keywords,
                "relation_name": cls.merge_relation_name,
            }

        result = []
        for merge_key in cls.merge_keys:
            if merge_key in merge_dict and merge_key in cls.merge_function:
                result.append(
                    cls.merge_function[merge_key](
                        edges_data.get(merge_key, []),
                        merge_dict.get(merge_key, []),
                    )
                )
            else:
                result.append("")
        return tuple(result)
