"""Canonical projection inputs and projection helpers."""

from .Execution import ArtifactRef, ExecutionLog
from .Project import FoundationProject, build_foundation_project
from .Identity import ProjectionIdentityManifest, build_identity_manifest
from .PropertyGraph import (
    assertion_graph_to_foundation_payload,
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
    "ArtifactRef",
    "ExecutionLog",
    "FoundationProject",
    "build_foundation_project",
    "assertion_graph_to_foundation_payload",
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
