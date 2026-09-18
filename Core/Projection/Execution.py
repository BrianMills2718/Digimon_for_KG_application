"""Small local execution/lineage recorder; no provider calls or orchestration."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any, Iterator
from uuid import uuid4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_digest() -> str:
    """Fingerprint the projection modules actually executed, not a guessed Git HEAD."""
    root = Path(__file__).parent
    entries = {p.name: file_sha256(p) for p in sorted(root.glob("*.py"))}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    path: str
    sha256: str
    input_digests: dict[str, str | None]

    @classmethod
    def from_file(cls, root: Path, path: Path, kind: str, inputs: dict[str, str | None]) -> "ArtifactRef":
        digest = file_sha256(path)
        return cls(f"{kind}:sha256:{digest}", kind, path.relative_to(root).as_posix(), digest, dict(inputs))


@dataclass
class Execution:
    execution_id: str
    operation: str
    inputs: list[dict[str, Any]]
    parameters: dict[str, Any]
    outputs: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ExecutionLog:
    """Append start/terminal events. Only completed operations advertise outputs.

    Single writer by design. Source-bearing artifacts remain local; the log
    contains references rather than full passages or credentials.
    """
    def __init__(self, path: Path):
        self.path = Path(path)

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()

    @contextmanager
    def operation(self, name: str, *, inputs: list[dict[str, Any]], parameters: dict[str, Any], parent_execution_id: str | None = None) -> Iterator[Execution]:
        execution = Execution(str(uuid4()), name, inputs, parameters)
        base = {
            "record_type": "execution_event", "schema_version": "1.0",
            "execution_id": execution.execution_id, "operation": name,
            "implementation_sha256": implementation_digest(),
            "inputs": inputs, "parameters": parameters,
            "parent_execution_id": parent_execution_id,
        }
        start = time.monotonic()
        self._append({**base, "status": "started", "at": datetime.now(timezone.utc).isoformat()})
        try:
            yield execution
        except BaseException as exc:
            frames = traceback.extract_tb(exc.__traceback__)
            self._append({
                **base, "status": "failed", "outputs": [],
                "at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - start,
                "error": {"type": type(exc).__name__, "message": str(exc)[:600],
                          "frames": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name} for f in frames[-8:]]},
                "diagnostics": execution.diagnostics,
            })
            raise
        else:
            self._append({
                **base, "status": "succeeded", "outputs": execution.outputs,
                "at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - start,
                "diagnostics": execution.diagnostics,
            })
