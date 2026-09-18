"""Canonical projection inputs and projection helpers."""

from .Identity import ProjectionIdentityManifest, build_identity_manifest
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
