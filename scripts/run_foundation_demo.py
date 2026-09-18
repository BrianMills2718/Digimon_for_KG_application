#!/usr/bin/env python3
"""Run the growing Foundation project demo (projections + SQL/graph evidence)."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Core.Projection.Execution import file_sha256
from Core.Projection.Project import FoundationProject, build_foundation_project


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--passages", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entity-id", required=True)
    parser.add_argument("--graph-hops", type=int, help="Also run maintained graph retrieval")
    parser.add_argument("--predicate", action="append", help="Filter assertion predicates before graph traversal")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reuse", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.reuse:
        project = FoundationProject.open(args.output, expected_inputs={
            "foundation": file_sha256(args.ir),
            "passages": file_sha256(args.passages) if args.passages else None,
        })
    else:
        project = build_foundation_project(args.ir, args.output, passage_path=args.passages, overwrite=args.overwrite)
    result = project.evidence_for_entity(args.entity_id)
    output = {"project": str(project.root), "result": result,
              "not_integrated": list(project.manifest["not_integrated"])}
    if args.graph_hops is not None:
        try:
            output["graph_result"] = asyncio.run(project.graph_neighborhood(
                [args.entity_id], k=args.graph_hops, predicates=args.predicate))
            output["not_integrated"] = [name for name in output["not_integrated"] if name != "graph_runtime"]
        except (KeyError, ValueError, RuntimeError) as exc:
            output["graph_result"] = {"status": "error", "error": str(exc)}
    print(json.dumps(output, indent=2, ensure_ascii=False))
    success = result["status"] == "ok" and output.get("graph_result", {"status": "ok"})["status"] == "ok"
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
