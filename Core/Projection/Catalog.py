"""Progressive-disclosure Markdown catalog for a saved Foundation project.

The catalog is an agent-readable map of semantic content and the actual
representations available in one project generation. It deliberately exposes
native locations/schemas instead of wrapping ordinary file navigation or SQL.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .Execution import file_sha256
from .FoundationIR import FoundationIR, load_foundation_ir
from .Relational import relational_schema_manifest

CATALOG_FORMAT_VERSION = "1.0"


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return text[:48] or "item"


def _page_name(canonical_id: str, label: str | None = None) -> str:
    digest = hashlib.sha256(canonical_id.encode("utf-8")).hexdigest()[:12]
    return f"{_slug(label or canonical_id)}-{digest}.md"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _rel(from_path: Path, target: Path) -> str:
    import os
    return Path(os.path.relpath(target, from_path.parent)).as_posix()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tree_entries(root: Path, *, exclude: Iterable[Path] = ()) -> list[dict[str, str]]:
    excluded = {path.resolve() for path in exclude}
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.resolve() in excluded:
            continue
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
        })
    return entries


def _preferred_name(ir: FoundationIR, entity_id: str) -> str:
    entity = ir.entities_by_id[entity_id]
    return entity.names[0] if entity.names else entity_id


def _assertion_passages(ir: FoundationIR, assertion_id: str) -> tuple[str, ...]:
    return tuple(p.passage_id for p in ir.passages_for_assertion(assertion_id))


def _representation_rows(project_manifest: dict[str, Any]) -> list[tuple[str, str, str]]:
    artifacts = project_manifest["artifacts"]
    return [
        ("Relational / SQLite", "available", artifacts["relational"]["path"]),
        ("Assertion graph", "available", artifacts["assertion_graph"]["path"]),
        ("Binary entity graph", "available", artifacts["binary_graph"]["path"]),
        ("Graph neighborhood retrieval", "available", "DIGIMON graph runtime over binary entity graph"),
        ("Vector index", "unavailable", "Batch 3 requires an observed semantic embedding/index route"),
        ("Progressive-disclosure catalog", "available", "this catalog"),
        ("Analytic suite", "available / partial", "degree, betweenness, PageRank, closeness, eigenvector centrality; components/density/clustering/coreness/bridges/articulation/assortativity summaries; Leiden unavailable until its real dependency path is present"),
    ]


@dataclass(frozen=True)
class CatalogProjection:
    root: Path
    manifest_path: Path
    index_path: Path
    entity_count: int
    assertion_count: int
    passage_count: int


def generate_foundation_catalog(
    ir: FoundationIR,
    project_manifest: dict[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> CatalogProjection:
    """Generate deterministic Markdown + a hash manifest for one project snapshot."""
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)

    entities_dir = root / "entities"
    assertions_dir = root / "assertions"
    sources_dir = root / "sources"
    schemas_dir = root / "schemas"
    representations_dir = root / "representations"
    for directory in (entities_dir, assertions_dir, sources_dir, schemas_dir, representations_dir):
        directory.mkdir(parents=True, exist_ok=True)

    entity_pages: dict[str, str] = {}
    assertion_pages: dict[str, str] = {}
    passage_pages: dict[str, str] = {}

    for entity_id in sorted(ir.entities_by_id):
        entity_pages[entity_id] = f"entities/{_page_name(entity_id, _preferred_name(ir, entity_id))}"
    for assertion in ir.assertions:
        assertion_pages[assertion.assertion_id] = f"assertions/{_page_name(assertion.assertion_id, assertion.predicate)}"
    for passage in ir.passages:
        passage_pages[passage.passage_id] = f"sources/{_page_name(passage.passage_id, passage.source_ref)}"

    # Source/evidence pages are intentionally metadata-first. Exact text remains
    # in the governed passage artifact and query surfaces, avoiding accidental
    # context explosion while keeping a direct evidence address.
    for passage in ir.passages:
        path = root / passage_pages[passage.passage_id]
        supporting = [
            aid
            for ref in passage.supporting_provenance_refs
            for aid in ir.assertions_by_provenance_ref.get(ref, ())
        ]
        lines = [
            f"# Source passage: {_md(passage.source_ref)}",
            "",
            f"- **Passage ID:** `{passage.passage_id}`",
            f"- **Source ref:** `{passage.source_ref}`",
            f"- **Namespace:** `{passage.namespace_id or ''}`",
            f"- **Source registry:** `{passage.source_registry_id or ''}`",
            f"- **Content hash (producer metadata):** `{passage.content_hash or ''}`",
            "",
            "## Supporting assertions",
        ]
        if supporting:
            for aid in sorted(set(supporting)):
                target = root / assertion_pages[aid]
                lines.append(f"- [{aid}]({_rel(path, target)})")
        else:
            lines.append("- None in this governed selection.")
        lines += [
            "",
            "## Evidence access",
            "",
            "Use the canonical `passage_id` / `source_ref` through the saved project or SQLite evidence tables to reopen the exact source text.",
        ]
        _write(path, "\n".join(lines))

    for assertion in ir.assertions:
        path = root / assertion_pages[assertion.assertion_id]
        lines = [
            f"# Assertion: {_md(assertion.predicate)}",
            "",
            f"- **Assertion ID:** `{assertion.assertion_id}`",
            f"- **Predicate:** `{assertion.predicate}`",
            f"- **Claim text:** {_md(assertion.claim_text or '')}",
            f"- **Confidence:** `{assertion.confidence if assertion.confidence is not None else ''}`",
            "",
            "## Roles",
            "",
        ]
        for role_name, fillers in assertion.roles.items():
            lines.append(f"### {_md(role_name)}")
            if not fillers:
                lines.append("- Empty role.")
            for filler in fillers:
                if filler.entity_id is not None:
                    target = root / entity_pages[filler.entity_id]
                    lines.append(
                        f"- entity [{_md(filler.name or filler.entity_id)}]({_rel(path, target)}) "
                        f"(`{filler.entity_id}`; type `{filler.entity_type or ''}`)"
                    )
                else:
                    rendered = filler.raw if filler.raw is not None else filler.value
                    lines.append(f"- value `{_md(rendered)}` (kind `{filler.value_kind or filler.kind}`)")
            lines.append("")
        lines += ["## Evidence", ""]
        passage_ids = _assertion_passages(ir, assertion.assertion_id) if ir.passages else ()
        if passage_ids:
            for passage_id in passage_ids:
                target = root / passage_pages[passage_id]
                lines.append(f"- [{passage_id}]({_rel(path, target)})")
        else:
            lines.append("- No passage companion evidence is available for this assertion in this project.")
        lines += ["", "## Qualifiers", ""]
        if assertion.qualifiers:
            for key, value in sorted(assertion.qualifiers.items()):
                lines.append(f"- `{key}`: `{_md(json.dumps(value, ensure_ascii=False, sort_keys=True))}`")
        else:
            lines.append("- None.")
        _write(path, "\n".join(lines))

    for entity_id, entity in sorted(ir.entities_by_id.items()):
        path = root / entity_pages[entity_id]
        lines = [
            f"# {_md(_preferred_name(ir, entity_id))}",
            "",
            f"- **Canonical entity ID:** `{entity_id}`",
            f"- **Types:** {', '.join(f'`{_md(x)}`' for x in entity.entity_types) or 'None'}",
            f"- **Aliases:** {', '.join(f'`{_md(x)}`' for x in entity.alias_ids) or 'None'}",
            "",
            "## Where this entity exists",
            "",
            f"- **SQL:** `entities.entity_id = {entity_id}`; join through `assertion_roles` for governed claims.",
            f"- **Binary graph:** node ID `{entity_id}` when the entity participates in at least one eligible binary assertion.",
            f"- **Assertion graph:** entity node ID `{entity_id}` connected from role-aware assertion nodes.",
            "- **Vector:** unavailable in this project generation until an observed embedding/index build succeeds.",
            "",
            "## Governed assertions",
            "",
        ]
        for assertion_id in entity.assertion_ids:
            assertion = ir.assertions_by_id[assertion_id]
            target = root / assertion_pages[assertion_id]
            lines.append(f"- [{_md(assertion.predicate)} — {assertion_id}]({_rel(path, target)})")
        if not entity.assertion_ids:
            lines.append("- None.")
        lines += [
            "",
            "## Retrieval / analysis guidance",
            "",
            "This page is descriptive, not a workflow policy. Use exact SQL for structured attributes/joins, the graph runtime for neighborhood/subgraph structure, and source pages for evidence. The external harness chooses the sequence.",
        ]
        _write(path, "\n".join(lines))

    relational = relational_schema_manifest()
    relational_path = schemas_dir / "relational.md"
    lines = ["# Relational schema", "", "Artifact: `" + project_manifest["artifacts"]["relational"]["path"] + "`", "", "| Table | Purpose | Primary key |", "|---|---|---|"]
    for table, info in relational.items():
        lines.append(f"| `{table}` | {_md(info['purpose'])} | `{info['primary_key']}` |")
    _write(relational_path, "\n".join(lines))

    graph_path = schemas_dir / "graph.md"
    _write(graph_path, "\n".join([
        "# Graph schemas",
        "",
        "## Assertion graph",
        "",
        "Lossless role-aware structural projection for the supported Foundation IR shape. Entity and assertion IDs are canonical; value-node IDs are projection-local. Assertion → filler edges preserve role name and ordinal.",
        "",
        "## Binary entity graph",
        "",
        "Undirected association view for exactly-two-entity assertions. N-ary assertions are not clique-expanded. Parallel governed assertions remain separately identified in the source MultiGraph and are retained as assertion records in the runtime retrieval view.",
        "",
        f"- Assertion graph artifact: `{project_manifest['artifacts']['assertion_graph']['path']}`",
        f"- Binary graph artifact: `{project_manifest['artifacts']['binary_graph']['path']}`",
    ]))

    vector_path = schemas_dir / "vectors.md"
    _write(vector_path, "\n".join([
        "# Vector collections",
        "",
        "**Status: unavailable in this project generation.**",
        "",
        "The target projection will index entity, assertion, and passage documents while retaining canonical IDs and embedding/index metadata. This catalog does not fabricate a vector address before a real semantic index has been built and queried.",
    ]))

    reps_path = root / "representations.md"
    lines = ["# Representations and capabilities", "", "| Representation / capability | Status | Address / note |", "|---|---|---|"]
    for name, status, address in _representation_rows(project_manifest):
        lines.append(f"| {_md(name)} | **{status}** | `{_md(address)}` |")
    _write(reps_path, "\n".join(lines))

    index_path = root / "index.md"
    lines = [
        "# DIGIMON knowledge and retrieval catalog",
        "",
        f"Foundation snapshot: `{project_manifest['input_digests']['foundation']}`",
        f"Passage companion: `{project_manifest['input_digests'].get('passages') or 'none'}`",
        "",
        "This is a progressive-disclosure map of both governed semantic content and the retrieval environment. It describes available representations and join identities; it does not prescribe an agent's reasoning sequence.",
        "",
        "## Environment",
        "",
        "- [Representations and capabilities](representations.md)",
        "- [Relational schema](schemas/relational.md)",
        "- [Graph schemas](schemas/graph.md)",
        "- [Vector collections](schemas/vectors.md)",
        "",
        "## Entities",
        "",
    ]
    for entity_id in sorted(entity_pages, key=lambda eid: (_preferred_name(ir, eid).casefold(), eid)):
        lines.append(f"- [{_md(_preferred_name(ir, entity_id))}]({entity_pages[entity_id]}) — `{entity_id}`")
    lines += ["", "## Assertions", ""]
    for assertion in ir.assertions:
        lines.append(f"- [{_md(assertion.predicate)}]({assertion_pages[assertion.assertion_id]}) — `{assertion.assertion_id}`")
    if ir.passages:
        lines += ["", "## Sources / evidence", ""]
        for passage in ir.passages:
            lines.append(f"- [{_md(passage.source_ref)}]({passage_pages[passage.passage_id]}) — `{passage.passage_id}`")
    _write(index_path, "\n".join(lines))

    catalog_manifest_path = root / "catalog.json"
    manifest = {
        "format_version": CATALOG_FORMAT_VERSION,
        "project_generation_id": project_manifest["generation_id"],
        "input_digests": project_manifest["input_digests"],
        "entry_point": "index.md",
        "entity_pages": entity_pages,
        "assertion_pages": assertion_pages,
        "passage_pages": passage_pages,
        "representation_status": [
            {"name": name, "status": status, "address": address}
            for name, status, address in _representation_rows(project_manifest)
        ],
    }
    # Write once, then bind the immutable page set into the manifest.
    _write_json(catalog_manifest_path, manifest)
    manifest["files"] = _tree_entries(root, exclude=(catalog_manifest_path,))
    _write_json(catalog_manifest_path, manifest)

    validate_foundation_catalog(root, expected_project_manifest=project_manifest)
    return CatalogProjection(
        root=root,
        manifest_path=catalog_manifest_path,
        index_path=index_path,
        entity_count=len(entity_pages),
        assertion_count=len(assertion_pages),
        passage_count=len(passage_pages),
    )


def validate_foundation_catalog(
    root: str | Path,
    *,
    expected_project_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "catalog.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != CATALOG_FORMAT_VERSION:
        raise ValueError("unsupported catalog manifest version")
    if expected_project_manifest is not None:
        if manifest.get("project_generation_id") != expected_project_manifest.get("generation_id"):
            raise ValueError("catalog/project generation mismatch")
        if manifest.get("input_digests") != expected_project_manifest.get("input_digests"):
            raise ValueError("catalog/project input mismatch")
    for item in manifest.get("files", []):
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing/escaped catalog file: {item['path']}")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"changed catalog file: {item['path']}")
    entry = (root / manifest["entry_point"]).resolve()
    if not entry.is_relative_to(root) or not entry.is_file():
        raise ValueError("catalog entry point is missing")
    return manifest
