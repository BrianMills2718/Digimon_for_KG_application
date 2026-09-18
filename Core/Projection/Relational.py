"""Relational projection of governed Foundation IR.

SQLite is the first backend because it adds no runtime dependency and is enough
to prove exact structured retrieval and cross-representation identity.  The
schema is normalized around Foundation identities so a future DuckDB/Postgres
backend can preserve the same logical contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import tempfile
from pathlib import Path
import sqlite3
from typing import Any

from .FoundationIR import FoundationIR
from .Identity import build_identity_manifest


RELATIONAL_PROJECTION_VERSION = "1.1"


@dataclass(frozen=True)
class RelationalProjection:
    path: Path
    projection_version: str
    source_sha256: str | None
    entity_count: int
    assertion_count: int
    passage_count: int


def relational_schema_manifest() -> dict[str, dict[str, str]]:
    """Return agent/catalog-facing descriptions of the canonical SQL schema."""

    return {
        "projection_metadata": {
            "purpose": "Projection identity/version and upstream snapshot metadata",
            "primary_key": "key",
        },
        "entities": {
            "purpose": "Canonical entities keyed by Foundation entity_id",
            "primary_key": "entity_id",
        },
        "entity_names": {
            "purpose": "Observed canonical/display names for each entity",
            "primary_key": "entity_id + name",
        },
        "entity_types": {
            "purpose": "Observed semantic types for each entity",
            "primary_key": "entity_id + entity_type",
        },
        "entity_aliases": {
            "purpose": "Producer-governed alias entity IDs",
            "primary_key": "entity_id + alias_id",
        },
        "assertions": {
            "purpose": "Governed assertions keyed by Foundation assertion_id",
            "primary_key": "assertion_id",
        },
        "assertion_roles": {
            "purpose": "Ordered n-ary role fillers without flattening role semantics",
            "primary_key": "assertion_id + role_name + filler_ordinal",
        },
        "assertion_qualifiers": {
            "purpose": "Additive Foundation qualifier values encoded as JSON",
            "primary_key": "assertion_id + qualifier_key",
        },
        "assertion_provenance": {
            "purpose": "Assertion to producer provenance/candidate references",
            "primary_key": "assertion_id + provenance_ref",
        },
        "assertion_source_urls": {
            "purpose": "Producer-observed source URLs attached to assertions",
            "primary_key": "assertion_id + source_url",
        },
        "passages": {
            "purpose": "Exact source passages keyed by Foundation passage_id",
            "primary_key": "passage_id",
        },
        "passage_support": {
            "purpose": "Passage to producer provenance references",
            "primary_key": "passage_id + provenance_ref",
        },
        "passage_source_urls": {
            "purpose": "Producer-observed source URLs attached to passages",
            "primary_key": "passage_id + source_url",
        },
    }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE projection_metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY
);

CREATE TABLE entity_names (
    entity_id TEXT NOT NULL,
    name TEXT NOT NULL,
    PRIMARY KEY (entity_id, name),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);

CREATE TABLE entity_types (
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    PRIMARY KEY (entity_id, entity_type),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);

CREATE TABLE entity_aliases (
    entity_id TEXT NOT NULL,
    alias_id TEXT NOT NULL,
    PRIMARY KEY (entity_id, alias_id),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);

CREATE TABLE assertions (
    assertion_id TEXT PRIMARY KEY,
    predicate TEXT NOT NULL,
    claim_text TEXT,
    confidence REAL,
    namespace_id TEXT,
    source_registry_id TEXT,
    payload_json TEXT NOT NULL
);

CREATE TABLE assertion_roles (
    assertion_id TEXT NOT NULL,
    role_name TEXT NOT NULL,
    filler_ordinal INTEGER NOT NULL,
    kind TEXT NOT NULL,
    entity_id TEXT,
    name TEXT,
    entity_type TEXT,
    value_kind TEXT,
    value_json TEXT,
    normalized_json TEXT,
    raw TEXT,
    PRIMARY KEY (assertion_id, role_name, filler_ordinal),
    FOREIGN KEY (assertion_id) REFERENCES assertions(assertion_id),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);

CREATE TABLE assertion_qualifiers (
    assertion_id TEXT NOT NULL,
    qualifier_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY (assertion_id, qualifier_key),
    FOREIGN KEY (assertion_id) REFERENCES assertions(assertion_id)
);

CREATE TABLE assertion_provenance (
    assertion_id TEXT NOT NULL,
    provenance_ref TEXT NOT NULL,
    PRIMARY KEY (assertion_id, provenance_ref),
    FOREIGN KEY (assertion_id) REFERENCES assertions(assertion_id)
);

CREATE TABLE assertion_source_urls (
    assertion_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    PRIMARY KEY (assertion_id, source_url),
    FOREIGN KEY (assertion_id) REFERENCES assertions(assertion_id)
);

CREATE TABLE passages (
    passage_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    source_registry_id TEXT,
    namespace_id TEXT,
    content_hash TEXT
);

CREATE TABLE passage_support (
    passage_id TEXT NOT NULL,
    provenance_ref TEXT NOT NULL,
    PRIMARY KEY (passage_id, provenance_ref),
    FOREIGN KEY (passage_id) REFERENCES passages(passage_id)
);

CREATE TABLE passage_source_urls (
    passage_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    PRIMARY KEY (passage_id, source_url),
    FOREIGN KEY (passage_id) REFERENCES passages(passage_id)
);

CREATE INDEX idx_assertions_predicate ON assertions(predicate);
CREATE INDEX idx_roles_entity ON assertion_roles(entity_id);
CREATE INDEX idx_roles_role_name ON assertion_roles(role_name);
CREATE INDEX idx_assertion_provenance_ref ON assertion_provenance(provenance_ref);
CREATE INDEX idx_passages_source_ref ON passages(source_ref);
CREATE INDEX idx_passage_support_ref ON passage_support(provenance_ref);
"""


def project_foundation_ir_to_sqlite(
    ir: FoundationIR,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> RelationalProjection:
    """Materialize a deterministic relational view of one Foundation IR snapshot."""

    output = Path(path)
    if output.is_symlink():
        raise ValueError("refusing to replace a symlink projection path")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    identity = build_identity_manifest(ir)
    raw_assertions = {item["assertion_id"]: item for item in ir.to_payload()["assertions"]}
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    conn = None
    try:
        conn = sqlite3.connect(str(temporary))
        conn.executescript(_SCHEMA)
        with conn:
            metadata = {
                "projection_kind": "foundation_relational",
                "projection_version": RELATIONAL_PROJECTION_VERSION,
                "foundation_format_version": ir.format_version,
                "foundation_producer": ir.producer,
                "source_sha256": ir.source_sha256 or "",
                "passage_sha256": ir.passage_sha256 or "",
            }
            conn.executemany(
                "INSERT INTO projection_metadata(key, value) VALUES (?, ?)",
                sorted(metadata.items()),
            )

            conn.executemany(
                "INSERT INTO entities(entity_id) VALUES (?)",
                ((entity_id,) for entity_id in identity.entity_ids),
            )

            for entity in ir.entities_by_id.values():
                conn.executemany(
                    "INSERT INTO entity_names(entity_id, name) VALUES (?, ?)",
                    ((entity.entity_id, value) for value in entity.names),
                )
                conn.executemany(
                    "INSERT INTO entity_types(entity_id, entity_type) VALUES (?, ?)",
                    ((entity.entity_id, value) for value in entity.entity_types),
                )
                conn.executemany(
                    "INSERT INTO entity_aliases(entity_id, alias_id) VALUES (?, ?)",
                    ((entity.entity_id, value) for value in entity.alias_ids),
                )

            for assertion in ir.assertions:
                conn.execute(
                    """
                    INSERT INTO assertions(
                        assertion_id, predicate, claim_text, confidence,
                        namespace_id, source_registry_id, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        assertion.assertion_id,
                        assertion.predicate,
                        assertion.claim_text,
                        assertion.confidence,
                        assertion.namespace_id,
                        assertion.source_registry_id,
                        _json(raw_assertions[assertion.assertion_id]),
                    ),
                )
                for role_name, fillers in assertion.roles.items():
                    for ordinal, filler in enumerate(fillers):
                        conn.execute(
                            """
                            INSERT INTO assertion_roles(
                                assertion_id, role_name, filler_ordinal, kind,
                                entity_id, name, entity_type, value_kind,
                                value_json, normalized_json, raw
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                assertion.assertion_id,
                                role_name,
                                ordinal,
                                filler.kind,
                                filler.entity_id,
                                filler.name,
                                filler.entity_type,
                                filler.value_kind,
                                _json(filler.value) if filler.value is not None else None,
                                _json(filler.normalized)
                                if filler.normalized is not None
                                else None,
                                filler.raw,
                            ),
                        )
                conn.executemany(
                    """
                    INSERT INTO assertion_qualifiers(
                        assertion_id, qualifier_key, value_json
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        (assertion.assertion_id, key, _json(value))
                        for key, value in sorted(assertion.qualifiers.items())
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO assertion_provenance(assertion_id, provenance_ref)
                    VALUES (?, ?)
                    """,
                    (
                        (assertion.assertion_id, value)
                        for value in dict.fromkeys(assertion.provenance_refs)
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO assertion_source_urls(assertion_id, source_url)
                    VALUES (?, ?)
                    """,
                    (
                        (assertion.assertion_id, value)
                        for value in dict.fromkeys(assertion.source_urls)
                    ),
                )

            for passage in ir.passages:
                conn.execute(
                    """
                    INSERT INTO passages(
                        passage_id, text, source_ref, source_registry_id,
                        namespace_id, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        passage.passage_id,
                        passage.text,
                        passage.source_ref,
                        passage.source_registry_id,
                        passage.namespace_id,
                        passage.content_hash,
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO passage_support(passage_id, provenance_ref)
                    VALUES (?, ?)
                    """,
                    (
                        (passage.passage_id, value)
                        for value in dict.fromkeys(passage.supporting_provenance_refs)
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO passage_source_urls(passage_id, source_url)
                    VALUES (?, ?)
                    """,
                    (
                        (passage.passage_id, value)
                        for value in dict.fromkeys(passage.source_urls)
                    ),
                )
        if conn.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("relational projection has unresolved foreign keys")
        if conn.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("relational projection integrity check failed")
        conn.close()
        conn = None
        # Publishing is the only mutation of the final path. Failed builds leave
        # the previous database intact; no-overwrite publication is race-safe.
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        if conn is not None:
            conn.close()
        temporary.unlink(missing_ok=True)

    return RelationalProjection(
        path=output,
        projection_version=RELATIONAL_PROJECTION_VERSION,
        source_sha256=ir.source_sha256,
        entity_count=len(identity.entity_ids),
        assertion_count=len(identity.assertion_ids),
        passage_count=len(identity.passage_ids),
    )
