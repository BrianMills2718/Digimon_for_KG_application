#!/usr/bin/env python3
"""Run the growing Foundation project demo (currently projections + SQL evidence)."""
from __future__ import annotations

import argparse
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
    print(json.dumps({"project": str(project.root), "result": result, "not_integrated": project.manifest["not_integrated"]}, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
