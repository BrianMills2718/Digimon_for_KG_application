"""Tiny persisted manifest tying ER/RK graph artifacts to corpus chunk identities.

This intentionally solves one concrete correctness problem rather than creating a
resource-versioning framework: a persisted graph must be rebuilt when the set of
source chunks changes, including when documents are added (which cannot be
inferred from graph node source_ids alone).
"""

from __future__ import annotations

import json
from pathlib import Path

from Core.Common.Logger import logger


_MANIFEST_NAME = "source_chunk_manifest.json"


def chunk_ids_from_pairs(chunks) -> list[str]:
    return sorted(
        {
            str(chunk_id)
            for chunk_id, _chunk in chunks or []
            if chunk_id is not None and str(chunk_id)
        }
    )


def manifest_path(graph) -> Path | None:
    storage = getattr(graph, "_graph", None)
    namespace = getattr(storage, "namespace", None)
    if namespace is None or not hasattr(namespace, "get_save_path"):
        return None
    try:
        return Path(namespace.get_save_path(_MANIFEST_NAME))
    except Exception:
        return None


def read_manifest(graph) -> list[str] | None:
    path = manifest_path(graph)
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("chunk_ids") if isinstance(payload, dict) else None
        if not isinstance(values, list):
            return None
        return sorted({str(value) for value in values if value})
    except Exception as exc:
        logger.warning(f"Could not read graph chunk manifest '{path}': {exc}")
        return None


def manifest_matches(graph, chunks) -> bool | None:
    """Return True/False when a manifest exists, otherwise None."""
    persisted = read_manifest(graph)
    if persisted is None:
        return None
    return persisted == chunk_ids_from_pairs(chunks)


def write_manifest(graph, chunks) -> bool:
    path = manifest_path(graph)
    if path is None:
        logger.warning("Graph namespace unavailable; cannot persist chunk manifest")
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "chunk_ids": chunk_ids_from_pairs(chunks),
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return True
    except Exception as exc:
        logger.warning(f"Could not persist graph chunk manifest '{path}': {exc}")
        return False
