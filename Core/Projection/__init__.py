"""Canonical projection inputs and projection helpers."""

from .Identity import ProjectionIdentityManifest, build_identity_manifest
from .PropertyGraph import (
    PROPERTY_GRAPH_PROJECTION_VERSION,
    BinaryEntityGraphProjection,
    project_foundation_ir_to_assertion_graph,
    project_foundation_ir_to_binary_entity_graph,
)
from .Relational import (
    RELATIONAL_PROJECTION_VERSION,
    RelationalProjection,
    project_foundation_ir_to_sqlite,
    relational_schema_manifest,
)
from .FoundationIR import (
    FOUNDATION_IR_FORMAT_VERSION,
    FOUNDATION_IR_PRODUCER,
    FOUNDATION_PASSAGE_FORMAT_VERSION,
    FoundationAssertionRecord,
    FoundationEntityRecord,
    FoundationIR,
    FoundationIRContractError,
    FoundationPassageRecord,
    FoundationRoleFillerRecord,
    load_foundation_ir,
    parse_foundation_ir,
    parse_foundation_passages,
)

__all__ = [
    "FOUNDATION_IR_FORMAT_VERSION",
    "FOUNDATION_IR_PRODUCER",
    "FOUNDATION_PASSAGE_FORMAT_VERSION",
    "ProjectionIdentityManifest",
    "build_identity_manifest",
    "PROPERTY_GRAPH_PROJECTION_VERSION",
    "BinaryEntityGraphProjection",
    "project_foundation_ir_to_assertion_graph",
    "project_foundation_ir_to_binary_entity_graph",
    "RELATIONAL_PROJECTION_VERSION",
    "RelationalProjection",
    "project_foundation_ir_to_sqlite",
    "relational_schema_manifest",
    "FoundationAssertionRecord",
    "FoundationEntityRecord",
    "FoundationIR",
    "FoundationIRContractError",
    "FoundationPassageRecord",
    "FoundationRoleFillerRecord",
    "load_foundation_ir",
    "parse_foundation_ir",
    "parse_foundation_passages",
]
