"""Small cleanup helpers for graph-derived canonical artifacts.

This is deliberately not a resource lifecycle framework. The maintained MCP
reference methods use predictable dataset-scoped VDB IDs, community storage,
and ER sparse-matrix paths. A successful forced graph rebuild should remove
those known derived artifacts so they cannot be reused against a different
graph generation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from Core.Common.Logger import logger


_CANONICAL_VDB_SUFFIXES = ("entities", "relations")
_SPARSE_FILENAMES = ("sparse_e2r.pkl", "sparse_r2c.pkl")
_COMMUNITY_FILENAMES = ("community_report.json", "community_node_map.json")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _working_dir(main_config) -> Path:
    working_dir = Path(getattr(main_config, "working_dir", "./results"))
    if not working_dir.is_absolute():
        working_dir = _project_root() / working_dir
    return working_dir


def canonical_vdb_paths(dataset_name: str) -> list[Path]:
    base = _project_root() / "storage" / "vdb"
    return [base / f"{dataset_name}_{suffix}" for suffix in _CANONICAL_VDB_SUFFIXES]


def canonical_community_paths(main_config, dataset_name: str) -> list[Path]:
    community_dir = _working_dir(main_config) / dataset_name / "community_storage"
    return [community_dir / filename for filename in _COMMUNITY_FILENAMES]


def canonical_sparse_paths(main_config, dataset_name: str) -> list[Path]:
    er_dir = _working_dir(main_config) / dataset_name / "er_graph"
    return [er_dir / filename for filename in _SPARSE_FILENAMES]


def _remove_paths(paths: Iterable[Path]) -> list[str]:
    removed = []
    for path in paths:
        try:
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path))
            elif path.exists():
                path.unlink()
                removed.append(str(path))
        except Exception as exc:
            logger.warning(f"Failed to invalidate derived artifact '{path}': {exc}")
    return removed


def invalidate_after_forced_graph_rebuild(
    main_config,
    dataset_name: str,
    *,
    invalidate_sparse_matrices: bool,
) -> list[str]:
    """Remove canonical persisted resources derived from a rebuilt graph.

    Canonical VDBs and community reports are dataset-scoped but not
    graph-generation-scoped, so rebuilding any graph variant invalidates them.
    Sparse matrices are currently persisted under the ER graph namespace and are
    invalidated only for ER rebuilds.
    """
    paths = list(canonical_vdb_paths(dataset_name))
    paths.extend(canonical_community_paths(main_config, dataset_name))
    if invalidate_sparse_matrices:
        paths.extend(canonical_sparse_paths(main_config, dataset_name))

    removed = _remove_paths(paths)
    if removed:
        logger.info(
            f"Invalidated graph-derived artifacts for '{dataset_name}': {removed}"
        )
    return removed
