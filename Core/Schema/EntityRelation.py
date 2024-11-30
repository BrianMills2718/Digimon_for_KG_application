from dataclasses import dataclass, asdict


@dataclass
class Entity:
    def __init__(self, entity_name: str, source_id: str, **kwargs):
        """
        Initializes an Entity object with the given attributes.

        Args:
            entity_name (str): The name of the entity, serving as the primary key.
            source_id (str): The unique identifier of the chunk from which this entity is derived.
            entity_type (str, optional): The type of the entity. Defaults to an empty string.
            description (str, optional): A description of the entity. Defaults to an empty string.
        """
        self.entity_name = entity_name  # Primary key
        self.source_id = source_id  # Unique identifier of the chunk from which this entity is derived
        self.entity_type = kwargs.get("entity_type", "")  # Entity type
        self.description = kwargs.get("description", "")  # The description of this entity

    @property
    def as_dict(self):
        return asdict(self)

@dataclass
class Relationship:

    def __init__(self, src_id: str, tgt_id: str, source_id: str, **kwargs):
        """
        Initializes an Edge object with the given attributes.

        Args:
            src_id (str): The name of the entity on the left side of the edge.
            tgt_id (str): The name of the entity on the right side of the edge.
            source_id (str): The unique identifier of the source from which this edge is derived.
            **kwargs: Additional keyword arguments for optional attributes.
                - relation_name (str, optional): The name of the relation. Defaults to an empty string.
                - relation_type (str, optional): The type of the relation. Defaults to an empty string.
                - weight (float, optional): The weight of the edge, used in GraphRAG and LightRAG. Defaults to 0.0.
                - description (str, optional): A description of the edge, used in GraphRAG and LightRAG. Defaults to an empty string.
                - keywords (str, optional): Keywords associated with the edge, used in LightRAG. Defaults to an empty string.
                - rank (int, optional): The rank of the edge, used in LightRAG. Defaults to 0.
        """
        self.src_id = src_id  # Name of the entity on the left side of the edge
        self.tgt_id = tgt_id  # Name of the entity on the right side of the edge
        self.source_id = source_id  # Unique identifier of the source from which this edge is derived
        self.relation_name = kwargs.get('relation_name', "")  # Name of the relation
        self.relation_type = kwargs.get('relation_type', "")  # Type of the relation
        self.weight = kwargs.get('weight', 0.0)  # Weight of the edge, used in GraphRAG and LightRAG
        self.description = kwargs.get('description', "")  # Description of the edge, used in GraphRAG and LightRAG
        self.keywords = kwargs.get('keywords', "")  # Keywords associated with the edge, used in LightRAG
        self.rank = kwargs.get('rank', 0)  # Rank of the edge, used in LightRAG

    @property
    def as_dict(self):
        return asdict(self)
