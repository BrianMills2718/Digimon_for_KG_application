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
    parser.add_argument("--catalog", action="store_true", help="Generate/revalidate the progressive-disclosure catalog")
    parser.add_argument("--centrality", choices=["degree", "betweenness", "pagerank"], help="Analyze the retrieved graph working set")
    parser.add_argument("--top-n", type=int, default=5, help="Number of analytical entities to retain")
    parser.add_argument("--aggregate-predicates", action="store_true", help="Run exact SQL assertion counts by predicate")
    parser.add_argument("--finding", action="store_true", help="Persist a bounded finding from --centrality output")
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
    if args.catalog:
        try:
            output["catalog_result"] = project.generate_catalog(overwrite=args.overwrite)
            output["not_integrated"] = [name for name in output["not_integrated"] if name != "catalog"]
        except (OSError, ValueError, RuntimeError) as exc:
            output["catalog_result"] = {"status": "error", "error": str(exc)}
    if args.aggregate_predicates:
        try:
            output["sql_analytics"] = project.aggregate_predicates()
        except (OSError, ValueError, RuntimeError) as exc:
            output["sql_analytics"] = {"status": "error", "error": str(exc)}
    if args.centrality is not None:
        try:
            hops = args.graph_hops if args.graph_hops is not None else 2
            output["analytic_result"] = asyncio.run(project.analyze_graph(
                [args.entity_id], k=hops, method=args.centrality, top_n=args.top_n, predicates=args.predicate))
            output["not_integrated"] = [name for name in output["not_integrated"] if name != "analytics"]
            if args.finding:
                output["finding_result"] = project.create_finding(output["analytic_result"])
        except (KeyError, ValueError, RuntimeError) as exc:
            output["analytic_result"] = {"status": "error", "error": str(exc)}
    if args.graph_hops is not None:
        try:
            output["graph_result"] = asyncio.run(project.graph_neighborhood(
                [args.entity_id], k=args.graph_hops, predicates=args.predicate))
            output["not_integrated"] = [name for name in output["not_integrated"] if name != "graph_runtime"]
        except (KeyError, ValueError, RuntimeError) as exc:
            output["graph_result"] = {"status": "error", "error": str(exc)}
    print(json.dumps(output, indent=2, ensure_ascii=False))
    success = (
        result["status"] == "ok"
        and output.get("graph_result", {"status": "ok"})["status"] == "ok"
        and output.get("catalog_result", {"status": "ok"})["status"] == "ok"
        and output.get("sql_analytics", {"status": "ok"})["status"] == "ok"
        and output.get("analytic_result", {"status": "ok"})["status"] == "ok"
        and output.get("finding_result", {"status": "derived_finding"})["status"] == "derived_finding"
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
