from collections import Counter
from typing import List
from Core.Common.Constants import GRAPH_FIELD_SEP


class MergeER:
    def __init__(self):
        self.merge_function = {}
        self.merge_types = []

    @staticmethod
    async def merge_info(self, nodes_data, **kwargs):
        """
        Merge entity information for a specific entity name, including source IDs, entity types, and descriptions.
        If an existing key is present in the data, merge the information; otherwise, use the new insert data.
        """
        if len(nodes_data) == 0:
            return []
        result = []
        for merge_key in self.merge_types:
            if merge_key in kwargs:
                result.append(self.merge_function(kwargs[merge_key])(kwargs[merge_key], nodes_data[merge_key]))
            else:
                result.append(self.merge_function(nodes_data[merge_key]))

        return tuple(result)


class MergeEntity(MergeER):
    def __init__(self):
        super().__init__()
        self.merge_function = {
            "source_id": self.merge_source_ids,
            "entity_type": self.merge_types,
            "description": self.merge_descriptions,
        }
        self.merge_types = ["source_id", "entity_type", "description"]

    @staticmethod
    def merge_source_ids(existing_source_ids: List[str], new_source_ids):
        merged_source_ids = list(set(new_source_ids) | set(existing_source_ids))
        return GRAPH_FIELD_SEP.join(merged_source_ids)

    @staticmethod
    def merge_types(existing_entity_types: List[str], new_entity_types):
        # Use the most frequency entity type as the new entity
        merged_entity_types = existing_entity_types + new_entity_types
        entity_type_counts = Counter(merged_entity_types)
        most_common_type = entity_type_counts.most_common(1)[0][0] if entity_type_counts else ''
        return most_common_type

    @staticmethod
    def merge_descriptions(entity_relationships: List[str], new_descriptions):
        merged_descriptions = list(set(new_descriptions) | set(entity_relationships))
        description = GRAPH_FIELD_SEP.join(sorted(merged_descriptions))
        return description


class MergeRelationship(MergeER):

    def __init__(self):
        super().__init__()
        self.merge_function = {
            "weight": self.merge_weight,
            "description": self.merge_description,
            "source_id": self.merge_source_ids,
            "keywords": self.merge_keywords,
            "relation_name": self.merge_relation_name
        }
        self.merge_types = ["weight", "description", "source_id", "keywords"]

    @staticmethod
    def merge_weight(merge_weight, new_weight):
        return new_weight + merge_weight

    @staticmethod
    def merge_description(entity_relationships, new_descriptions):
        return GRAPH_FIELD_SEP.join(
            sorted(set(new_descriptions) | entity_relationships)
        )

    @staticmethod
    def merge_source_ids(existing_source_ids: List[str], new_source_ids):
        return GRAPH_FIELD_SEP.join(
            set(new_source_ids + existing_source_ids)
        )

    @staticmethod
    def merge_keywords(keywords: List[str], new_keywords):
        return GRAPH_FIELD_SEP.join(
            set(keywords + new_keywords)
        )

    @staticmethod
    def merge_relation_name(relation_name, new_relation_name):
        return GRAPH_FIELD_SEP.join(
            sorted(set(relation_name + new_relation_name)
                   ))
