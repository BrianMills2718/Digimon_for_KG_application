"""Strict consumer model for onto-canon6 Foundation Assertion IR.

This module is representation-neutral. It validates the governed semantic
handoff and builds canonical indexes that later projections can reuse; it does
not decide how assertions become graph edges, SQL rows, embeddings, or wiki
pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


FOUNDATION_IR_FORMAT_VERSION = "1.3"
FOUNDATION_PASSAGE_FORMAT_VERSION = "1.0"
FOUNDATION_IR_PRODUCER = "onto-canon6"


class FoundationIRContractError(ValueError):
    """Raised when a Foundation IR handoff violates the supported contract."""


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FoundationIRContractError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, field_name)


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise FoundationIRContractError(f"{field_name} must be a list")
    result = []
    for index, item in enumerate(value):
        result.append(_require_nonempty_string(item, f"{field_name}[{index}]"))
    return tuple(result)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FoundationIRContractError(f"{field_name} must be an object")
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class FoundationRoleFillerRecord:
    kind: str
    entity_id: str | None = None
    name: str | None = None
    entity_type: str | None = None
    alias_ids: tuple[str, ...] = ()
    value_kind: str | None = None
    value: Any = None
    normalized: Any = None
    raw: str | None = None

    @classmethod
    def from_payload(
        cls, payload: Any, *, context: str
    ) -> "FoundationRoleFillerRecord":
        if not isinstance(payload, dict):
            raise FoundationIRContractError(f"{context} must be an object")

        kind = _require_nonempty_string(payload.get("kind"), f"{context}.kind")
        entity_id = _optional_string(payload.get("entity_id"), f"{context}.entity_id")
        name = _optional_string(payload.get("name"), f"{context}.name")
        entity_type = _optional_string(
            payload.get("entity_type"), f"{context}.entity_type"
        )
        alias_ids = _string_list(payload.get("alias_ids"), f"{context}.alias_ids")
        value_kind = _optional_string(
            payload.get("value_kind"), f"{context}.value_kind"
        )
        raw = _optional_string(payload.get("raw"), f"{context}.raw")

        if kind == "entity" and entity_id is None:
            raise FoundationIRContractError(
                f"{context}.entity_id is required for entity fillers"
            )

        return cls(
            kind=kind,
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            alias_ids=alias_ids,
            value_kind=value_kind,
            value=payload.get("value"),
            normalized=payload.get("normalized"),
            raw=raw,
        )


@dataclass(frozen=True)
class FoundationAssertionRecord:
    assertion_id: str
    predicate: str
    claim_text: str | None
    roles: Mapping[str, tuple[FoundationRoleFillerRecord, ...]]
    qualifiers: Mapping[str, Any]
    confidence: float | None
    provenance_refs: tuple[str, ...]
    source_urls: tuple[str, ...]
    namespace_id: str | None
    source_registry_id: str | None

    @classmethod
    def from_payload(
        cls, payload: Any, *, index: int
    ) -> "FoundationAssertionRecord":
        context = f"assertions[{index}]"
        if not isinstance(payload, dict):
            raise FoundationIRContractError(f"{context} must be an object")

        assertion_id = _require_nonempty_string(
            payload.get("assertion_id"), f"{context}.assertion_id"
        )
        predicate = _require_nonempty_string(
            payload.get("predicate"), f"{context}.predicate"
        )
        claim_text = _optional_string(payload.get("claim_text"), f"{context}.claim_text")

        raw_roles = payload.get("roles")
        if not isinstance(raw_roles, dict):
            raise FoundationIRContractError(f"{context}.roles must be an object")

        roles: dict[str, tuple[FoundationRoleFillerRecord, ...]] = {}
        for role_name, fillers in raw_roles.items():
            role = _require_nonempty_string(role_name, f"{context}.roles role name")
            if not isinstance(fillers, list):
                raise FoundationIRContractError(
                    f"{context}.roles[{role!r}] must be a list"
                )
            roles[role] = tuple(
                FoundationRoleFillerRecord.from_payload(
                    filler, context=f"{context}.roles[{role!r}][{filler_index}]"
                )
                for filler_index, filler in enumerate(fillers)
            )

        raw_confidence = payload.get("confidence")
        if raw_confidence is not None and (
            isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
        ):
            raise FoundationIRContractError(
                f"{context}.confidence must be numeric or null"
            )

        return cls(
            assertion_id=assertion_id,
            predicate=predicate,
            claim_text=claim_text,
            roles=MappingProxyType(roles),
            qualifiers=_mapping(payload.get("qualifiers", {}), f"{context}.qualifiers"),
            confidence=float(raw_confidence) if raw_confidence is not None else None,
            provenance_refs=_string_list(
                payload.get("provenance_refs", []), f"{context}.provenance_refs"
            ),
            source_urls=_string_list(
                payload.get("source_urls", []), f"{context}.source_urls"
            ),
            namespace_id=_optional_string(
                payload.get("namespace_id"), f"{context}.namespace_id"
            ),
            source_registry_id=_optional_string(
                payload.get("source_registry_id"), f"{context}.source_registry_id"
            ),
        )

    def entity_ids(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for fillers in self.roles.values():
            for filler in fillers:
                if filler.entity_id is not None and filler.entity_id not in seen:
                    seen.add(filler.entity_id)
                    ordered.append(filler.entity_id)
        return tuple(ordered)


@dataclass(frozen=True)
class FoundationPassageRecord:
    passage_id: str
    text: str
    source_ref: str
    source_urls: tuple[str, ...]
    source_registry_id: str | None
    namespace_id: str | None
    content_hash: str | None
    supporting_provenance_refs: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Any, *, index: int) -> "FoundationPassageRecord":
        context = f"passages[{index}]"
        if not isinstance(payload, dict):
            raise FoundationIRContractError(f"{context} must be an object")

        content_hash = _optional_string(
            payload.get("content_hash"), f"{context}.content_hash"
        )
        if content_hash is not None:
            lowered = content_hash.lower()
            if len(lowered) != 64 or any(
                char not in "0123456789abcdef" for char in lowered
            ):
                raise FoundationIRContractError(
                    f"{context}.content_hash must be a 64-character SHA-256 hex digest"
                )
            content_hash = lowered

        supporting = _string_list(
            payload.get("supporting_provenance_refs"),
            f"{context}.supporting_provenance_refs",
        )
        if not supporting:
            raise FoundationIRContractError(
                f"{context}.supporting_provenance_refs must not be empty"
            )

        return cls(
            passage_id=_require_nonempty_string(
                payload.get("passage_id"), f"{context}.passage_id"
            ),
            text=_require_nonempty_string(payload.get("text"), f"{context}.text"),
            source_ref=_require_nonempty_string(
                payload.get("source_ref"), f"{context}.source_ref"
            ),
            source_urls=_string_list(
                payload.get("source_urls", []), f"{context}.source_urls"
            ),
            source_registry_id=_optional_string(
                payload.get("source_registry_id"), f"{context}.source_registry_id"
            ),
            namespace_id=_optional_string(
                payload.get("namespace_id"), f"{context}.namespace_id"
            ),
            content_hash=content_hash,
            supporting_provenance_refs=supporting,
        )


@dataclass(frozen=True)
class FoundationEntityRecord:
    """Representation-neutral canonical entity summary from assertion roles."""

    entity_id: str
    names: tuple[str, ...]
    entity_types: tuple[str, ...]
    alias_ids: tuple[str, ...]
    assertion_ids: tuple[str, ...]


@dataclass(frozen=True)
class FoundationIR:
    """Validated governed semantic input plus reusable canonical indexes."""

    format_version: str
    producer: str
    assertions: tuple[FoundationAssertionRecord, ...]
    source_sha256: str | None = None
    passages: tuple[FoundationPassageRecord, ...] = ()

    assertions_by_id: Mapping[str, FoundationAssertionRecord] = field(init=False)
    entities_by_id: Mapping[str, FoundationEntityRecord] = field(init=False)
    assertions_by_entity_id: Mapping[str, tuple[str, ...]] = field(init=False)
    assertions_by_provenance_ref: Mapping[str, tuple[str, ...]] = field(init=False)
    passages_by_id: Mapping[str, FoundationPassageRecord] = field(init=False)
    passages_by_provenance_ref: Mapping[str, tuple[str, ...]] = field(init=False)

    def __post_init__(self) -> None:
        assertions_by_id = {item.assertion_id: item for item in self.assertions}
        if len(assertions_by_id) != len(self.assertions):
            raise FoundationIRContractError("duplicate assertion_id in Foundation IR")

        passages_by_id = {item.passage_id: item for item in self.passages}
        if len(passages_by_id) != len(self.passages):
            raise FoundationIRContractError(
                "duplicate passage_id in Foundation passage bundle"
            )

        entity_names: dict[str, set[str]] = {}
        entity_types: dict[str, set[str]] = {}
        entity_aliases: dict[str, set[str]] = {}
        entity_assertions: dict[str, list[str]] = {}
        assertions_by_provenance_ref: dict[str, list[str]] = {}

        for assertion in self.assertions:
            for provenance_ref in assertion.provenance_refs:
                assertions_by_provenance_ref.setdefault(provenance_ref, []).append(
                    assertion.assertion_id
                )
            for fillers in assertion.roles.values():
                for filler in fillers:
                    if filler.entity_id is None:
                        continue
                    entity_id = filler.entity_id
                    if filler.name:
                        entity_names.setdefault(entity_id, set()).add(filler.name)
                    if filler.entity_type:
                        entity_types.setdefault(entity_id, set()).add(filler.entity_type)
                    entity_aliases.setdefault(entity_id, set()).update(filler.alias_ids)
                    ids = entity_assertions.setdefault(entity_id, [])
                    if assertion.assertion_id not in ids:
                        ids.append(assertion.assertion_id)

        entities_by_id = {
            entity_id: FoundationEntityRecord(
                entity_id=entity_id,
                names=tuple(sorted(entity_names.get(entity_id, set()))),
                entity_types=tuple(sorted(entity_types.get(entity_id, set()))),
                alias_ids=tuple(sorted(entity_aliases.get(entity_id, set()))),
                assertion_ids=tuple(assertion_ids),
            )
            for entity_id, assertion_ids in entity_assertions.items()
        }

        passages_by_provenance_ref: dict[str, list[str]] = {}
        for passage in self.passages:
            for provenance_ref in passage.supporting_provenance_refs:
                passages_by_provenance_ref.setdefault(provenance_ref, []).append(
                    passage.passage_id
                )

        object.__setattr__(
            self, "assertions_by_id", MappingProxyType(assertions_by_id)
        )
        object.__setattr__(self, "entities_by_id", MappingProxyType(entities_by_id))
        object.__setattr__(
            self,
            "assertions_by_entity_id",
            MappingProxyType(
                {
                    entity_id: tuple(assertion_ids)
                    for entity_id, assertion_ids in entity_assertions.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "assertions_by_provenance_ref",
            MappingProxyType(
                {
                    ref: tuple(assertion_ids)
                    for ref, assertion_ids in assertions_by_provenance_ref.items()
                }
            ),
        )
        object.__setattr__(self, "passages_by_id", MappingProxyType(passages_by_id))
        object.__setattr__(
            self,
            "passages_by_provenance_ref",
            MappingProxyType(
                {
                    ref: tuple(passage_ids)
                    for ref, passage_ids in passages_by_provenance_ref.items()
                }
            ),
        )

    def passages_for_assertion(
        self, assertion_id: str
    ) -> tuple[FoundationPassageRecord, ...]:
        assertion = self.assertions_by_id.get(assertion_id)
        if assertion is None:
            raise KeyError(assertion_id)
        passage_ids: list[str] = []
        seen: set[str] = set()
        for provenance_ref in assertion.provenance_refs:
            for passage_id in self.passages_by_provenance_ref.get(provenance_ref, ()):
                if passage_id not in seen:
                    seen.add(passage_id)
                    passage_ids.append(passage_id)
        return tuple(self.passages_by_id[item] for item in passage_ids)


def _validate_envelope(payload: Any) -> tuple[str, str, list[Any]]:
    if not isinstance(payload, dict):
        raise FoundationIRContractError("Foundation IR envelope must be an object")

    format_version = _require_nonempty_string(
        payload.get("format_version"), "format_version"
    )
    producer = _require_nonempty_string(payload.get("producer"), "producer")
    if format_version != FOUNDATION_IR_FORMAT_VERSION:
        raise FoundationIRContractError(
            f"unsupported Foundation IR format_version {format_version!r}; "
            f"expected {FOUNDATION_IR_FORMAT_VERSION!r}"
        )
    if producer != FOUNDATION_IR_PRODUCER:
        raise FoundationIRContractError(
            f"unsupported Foundation IR producer {producer!r}; "
            f"expected {FOUNDATION_IR_PRODUCER!r}"
        )

    assertions = payload.get("assertions")
    if not isinstance(assertions, list):
        raise FoundationIRContractError("assertions must be a list")
    count = payload.get("assertion_count")
    if isinstance(count, bool) or not isinstance(count, int):
        raise FoundationIRContractError("assertion_count must be an integer")
    if count != len(assertions):
        raise FoundationIRContractError(
            f"assertion_count={count} does not match len(assertions)={len(assertions)}"
        )
    return format_version, producer, assertions


def parse_foundation_passages(payload: Any) -> tuple[FoundationPassageRecord, ...]:
    if not isinstance(payload, dict):
        raise FoundationIRContractError("Foundation passage envelope must be an object")
    version = _require_nonempty_string(
        payload.get("format_version"), "passage.format_version"
    )
    producer = _require_nonempty_string(payload.get("producer"), "passage.producer")
    if version != FOUNDATION_PASSAGE_FORMAT_VERSION:
        raise FoundationIRContractError(
            f"unsupported Foundation passage format_version {version!r}; "
            f"expected {FOUNDATION_PASSAGE_FORMAT_VERSION!r}"
        )
    if producer != FOUNDATION_IR_PRODUCER:
        raise FoundationIRContractError(
            f"unsupported Foundation passage producer {producer!r}; "
            f"expected {FOUNDATION_IR_PRODUCER!r}"
        )
    raw_passages = payload.get("passages")
    if not isinstance(raw_passages, list):
        raise FoundationIRContractError("passages must be a list")
    count = payload.get("passage_count")
    if isinstance(count, bool) or not isinstance(count, int):
        raise FoundationIRContractError("passage_count must be an integer")
    if count != len(raw_passages):
        raise FoundationIRContractError(
            f"passage_count={count} does not match len(passages)={len(raw_passages)}"
        )
    passages = tuple(
        FoundationPassageRecord.from_payload(item, index=index)
        for index, item in enumerate(raw_passages)
    )
    if len({item.passage_id for item in passages}) != len(passages):
        raise FoundationIRContractError(
            "duplicate passage_id in Foundation passage bundle"
        )
    return passages


def parse_foundation_ir(
    payload: Any,
    *,
    passage_payload: Any | None = None,
    source_sha256: str | None = None,
) -> FoundationIR:
    format_version, producer, raw_assertions = _validate_envelope(payload)
    assertions = tuple(
        FoundationAssertionRecord.from_payload(item, index=index)
        for index, item in enumerate(raw_assertions)
    )
    if len({item.assertion_id for item in assertions}) != len(assertions):
        raise FoundationIRContractError("duplicate assertion_id in Foundation IR")

    passages = (
        parse_foundation_passages(passage_payload)
        if passage_payload is not None
        else ()
    )
    ir = FoundationIR(
        format_version=format_version,
        producer=producer,
        assertions=assertions,
        source_sha256=source_sha256,
        passages=passages,
    )

    if passage_payload is not None:
        assertion_refs = set(ir.assertions_by_provenance_ref)
        passage_refs = set(ir.passages_by_provenance_ref)
        missing = sorted(assertion_refs - passage_refs)
        orphaned = sorted(passage_refs - assertion_refs)
        if missing:
            raise FoundationIRContractError(
                "Foundation passage companion is missing provenance refs for "
                + ", ".join(missing)
            )
        if orphaned:
            raise FoundationIRContractError(
                "Foundation passage companion contains provenance refs outside "
                "the assertion selection: "
                + ", ".join(orphaned)
            )

    return ir


def _load_json_file(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        return json.loads(raw.decode("utf-8")), digest
    except UnicodeDecodeError as exc:
        raise FoundationIRContractError(f"{path} is not UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise FoundationIRContractError(f"{path} is not valid JSON: {exc}") from exc


def _validate_sha256_sidecar(path: Path, digest: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        return
    text = sidecar.read_text(encoding="utf-8").strip()
    if not text:
        raise FoundationIRContractError(f"{sidecar} is empty")
    expected = text.split()[0].lower()
    if len(expected) != 64 or any(
        char not in "0123456789abcdef" for char in expected
    ):
        raise FoundationIRContractError(
            f"{sidecar} does not contain a valid SHA-256"
        )
    if expected != digest:
        raise FoundationIRContractError(
            f"Foundation IR SHA-256 mismatch: sidecar={expected}, actual={digest}"
        )


def load_foundation_ir(
    path: str | Path,
    *,
    passage_path: str | Path | None = None,
    validate_sidecar: bool = True,
) -> FoundationIR:
    """Load an onto-canon6 Foundation snapshot and optional passage companion."""

    source_path = Path(path)
    payload, digest = _load_json_file(source_path)
    if validate_sidecar:
        _validate_sha256_sidecar(source_path, digest)

    passage_payload = None
    if passage_path is not None:
        passage_payload, _ = _load_json_file(Path(passage_path))

    return parse_foundation_ir(
        payload,
        passage_payload=passage_payload,
        source_sha256=digest,
    )
