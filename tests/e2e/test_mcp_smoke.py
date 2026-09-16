#!/usr/bin/env python3
"""End-to-end canary for the canonical DIGIMON MCP path.

Default mode reuses ``Fictional_Test`` artifacts. Set
``DIGIMON_CANARY_REBUILD=1`` to prepare a separate corpus namespace and force a
fresh ER graph + entity VDB build. Model-backed steps require provider credentials.
"""

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

SOURCE_DATASET = "Fictional_Test"
SOURCE_DIRECTORY = PROJECT_ROOT / "Data" / SOURCE_DATASET
REBUILD = os.getenv("DIGIMON_CANARY_REBUILD", "").strip().lower() in {
    "1", "true", "yes", "on"
}
DATASET = os.getenv(
    "DIGIMON_CANARY_DATASET",
    "Fictional_Canary_Rebuild" if REBUILD else SOURCE_DATASET,
)
GRAPH_ID = f"{DATASET}_ERGraph"
VDB_ID = f"{DATASET}_entities"
INSUFFICIENT = "Insufficient retrieved evidence to answer the question."

results: list[tuple[str, bool, str]] = []


def record(step: str, passed: bool, detail: str = ""):
    results.append((step, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {step}" + (f" -- {detail}" if detail else ""))


def summary() -> bool:
    failed = [(step, detail) for step, ok, detail in results if not ok]
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} passed")
    for step, detail in failed:
        print(f"  [FAIL] {step}: {detail}")
    return not failed


def valid_answer(answer: str) -> bool:
    value = str(answer).strip()
    return (
        len(value) > 10
        and value != "Failed to generate answer."
        and value != INSUFFICIENT
    )


async def main():
    import digimon_mcp_stdio_server as mcp

    print(f"DIGIMON MCP Canary -- {DATASET} ({'rebuild' if REBUILD else 'reuse'})")

    try:
        await mcp._ensure_initialized()
        record("initialize", "initialized" in mcp._state)
    except Exception as exc:
        record("initialize", False, str(exc))
        traceback.print_exc()
        return summary()

    if REBUILD:
        try:
            prepared = json.loads(
                await mcp.corpus_prepare(str(SOURCE_DIRECTORY), DATASET)
            )
            ok = prepared.get("status") == "success" and prepared.get("document_count", 0) > 0
            record("corpus_prepare", ok, str(prepared.get("message", "")))
            if not ok:
                return summary()
        except Exception as exc:
            record("corpus_prepare", False, str(exc))
            traceback.print_exc()
            return summary()

    try:
        built = json.loads(await mcp.graph_build_er(DATASET, force_rebuild=REBUILD))
        graph_ok = (
            built.get("status") == "success"
            and GRAPH_ID in mcp._state["context"].list_graphs()
        )
        record(
            "graph_build_er",
            graph_ok,
            f"nodes={built.get('node_count')}, edges={built.get('edge_count')}",
        )
        if not graph_ok:
            return summary()
    except Exception as exc:
        record("graph_build_er", False, str(exc))
        traceback.print_exc()
        return summary()

    try:
        built = json.loads(
            await mcp.entity_vdb_build(GRAPH_ID, VDB_ID, force_rebuild=REBUILD)
        )
        vdb_ok = (
            built.get("num_entities_indexed", 0) > 0
            and not str(built.get("status", "")).lower().startswith("error")
        )
        record("entity_vdb_build", vdb_ok, str(built.get("status", "")))
        if not vdb_ok:
            return summary()
    except Exception as exc:
        record("entity_vdb_build", False, str(exc))
        traceback.print_exc()
        return summary()

    resources = json.loads(await mcp.list_available_resources())
    record(
        "resource_discovery",
        GRAPH_ID in resources.get("graphs", []) and VDB_ID in resources.get("vdbs", []),
        f"graphs={resources.get('graphs')}, vdbs={resources.get('vdbs')}",
    )

    methods = json.loads(await mcp.list_methods())
    record("method_discovery", len(methods) == 10, f"count={len(methods)}")

    graph_types = json.loads(await mcp.list_graph_types())
    record("graph_type_discovery", len(graph_types) == 5, f"count={len(graph_types)}")

    search = json.loads(await mcp.entity_vdb_search(VDB_ID, "Zorathian Empire", top_k=5))
    entities = [item["entity_name"] for item in search.get("similar_entities", [])]
    record("entity_vdb_search", bool(entities), f"entities={entities[:3]}")

    chunks = []
    if entities:
        neighbors = json.loads(await mcp.entity_onehop(entities[:3], GRAPH_ID))
        record(
            "entity_onehop",
            neighbors.get("total_neighbors_found", 0) > 0,
            f"neighbors={neighbors.get('total_neighbors_found', 0)}",
        )

        relationships = json.loads(
            await mcp.relationship_onehop(entities[:3], GRAPH_ID)
        )
        record(
            "relationship_onehop",
            bool(relationships.get("one_hop_relationships", [])),
            f"relationships={len(relationships.get('one_hop_relationships', []))}",
        )

        chunk_result = json.loads(await mcp.chunk_get_text(GRAPH_ID, entities[:3]))
        chunks = chunk_result.get("retrieved_chunks", [])
        record("chunk_get_text", bool(chunks), f"chunks={len(chunks)}")
    else:
        record("entity_onehop", False, "no entity search results")
        record("relationship_onehop", False, "no entity search results")
        record("chunk_get_text", False, "no entity search results")

    extracted = json.loads(
        await mcp.meta_extract_entities(
            "What is the connection between crystal technology and the Zorathian Empire?"
        )
    ).get("entities", [])
    extracted_names = [item["entity_name"] for item in extracted]
    record("meta_extract_entities", bool(extracted_names), str(extracted_names))

    linked_ids = []
    if extracted_names:
        linked = json.loads(
            await mcp.entity_link(extracted_names, VDB_ID, similarity_threshold=0.1)
        )
        linked_ids = [
            item["linked_entity_id"]
            for item in linked.get("linked_entities_results", [])
            if item.get("link_status") == "linked" and item.get("linked_entity_id")
        ]
    record("entity_link", bool(linked_ids), f"linked={linked_ids}")

    ppr_seeds = linked_ids[:3] or entities[:3]
    if ppr_seeds:
        ranked = json.loads(await mcp.entity_ppr(GRAPH_ID, ppr_seeds, top_k=10)).get(
            "ranked_entities", []
        )
        record("entity_ppr", bool(ranked), f"ranked={len(ranked)}")
    else:
        record("entity_ppr", False, "no seed entities")

    evidence_text = [
        chunk.get("text_content", chunk.get("text", ""))[:500]
        for chunk in chunks[:3]
        if chunk.get("text_content", chunk.get("text", ""))
    ]
    if evidence_text:
        answer = await mcp.meta_generate_answer(
            "What is crystal technology?", evidence_text
        )
        record("meta_generate_answer", valid_answer(answer), answer[:100])
    else:
        record("meta_generate_answer", False, "no retrieved source evidence")

    try:
        method_result = json.loads(
            await mcp.execute_method(
                "basic_local",
                "What is crystal technology?",
                DATASET,
            )
        )
        method_answer = str(method_result.get("final_output", {}).get("answer", ""))
        record(
            "execute_method.basic_local",
            "error" not in method_result and valid_answer(method_answer),
            f"answer={method_answer[:100]}, steps={list(method_result.get('all_step_outputs', {}))}",
        )
    except Exception as exc:
        record("execute_method.basic_local", False, str(exc))
        traceback.print_exc()

    return summary()


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)
