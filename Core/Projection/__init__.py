"""Canonical projection inputs and projection helpers."""

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
