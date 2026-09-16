#!/usr/bin/env python3
"""End-to-end canary for the canonical DIGIMON MCP path.

The test calls the current MCP tool functions directly via Python imports rather
than speaking the stdio protocol. By default it reuses the small
``Fictional_Test`` dataset/artifacts for a fast smoke run.

Set ``DIGIMON_CANARY_REBUILD=1`` to exercise corpus preparation plus a forced
ER-graph/VDB rebuild under a separate canary dataset namespace. That mode needs
working provider credentials because graph extraction/embeddings/answer steps
are model-backed.

Usage:
    python tests/e2e/test_mcp_smoke.py
    DIGIMON_CANARY_REBUILD=1 python tests/e2e/test_mcp_smoke.py
"""

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

# Ensure project root on path and cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

SOURCE_DATASET = "Fictional_Test"
SOURCE_DIRECTORY = PROJECT_ROOT / "Data" / SOURCE_DATASET
REBUILD = os.getenv("DIGIMON_CANARY_REBUILD", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DATASET = os.getenv(
    "DIGIMON_CANARY_DATASET",
    "Fictional_Canary_Rebuild" if REBUILD else SOURCE_DATASET,
)
GRAPH_ID = f"{DATASET}_ERGraph"
VDB_ID = f"{DATASET}_entities"

# Track results.
results: list[tuple[str, bool, str]] = []


def record(step: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append((step, passed, detail))
    print(f"  [{status}] {step}" + (f" -- {detail}" if detail else ""))


def print_summary() -> bool:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)

    for step, ok, _ in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {step}")

    print(f"\n{passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("\nAll smoke tests PASSED!")
        return True

    print(f"\n{failed} smoke test(s) FAILED")
    return False


async def main():
    mode = "clean rebuild" if REBUILD else "cached/reuse"
    print("=" * 60)
    print(f"DIGIMON MCP Canary -- {DATASET} ({mode})")
    print("=" * 60)

    # ================================================================
    # A. SETUP -- initialize DIGIMON and make graph/VDB available
    # ================================================================
    print("\n--- A. Setup ---")

    import digimon_mcp_stdio_server as mcp_srv

    await mcp_srv._ensure_initialized()
    record(
        "A1: _ensure_initialized()",
        "initialized" in mcp_srv._state,
        f"keys: {list(mcp_srv._state.keys())}",
    )

    if REBUILD:
        if not SOURCE_DIRECTORY.exists():
            record(
                "A2: source corpus exists",
                False,
                f"missing directory: {SOURCE_DIRECTORY}",
            )
            return print_summary()

        try:
            prepare_json = await mcp_srv.corpus_prepare(
                str(SOURCE_DIRECTORY), DATASET
            )
            prepare_result = json.loads(prepare_json)
            prepare_ok = (
                prepare_result.get("status") == "success"
                and prepare_result.get("document_count", 0) > 0
            )
            record(
                "A2: corpus_prepare()",
                prepare_ok,
                (
                    f"documents={prepare_result.get('document_count', 0)}, "
                    f"path={prepare_result.get('corpus_json_path')}"
                ),
            )
            if not prepare_ok:
                return print_summary()
        except Exception as exc:
            record("A2: corpus_prepare()", False, f"ERROR: {exc}")
            traceback.print_exc()
            return print_summary()

    from Core.AgentTools.graph_construction_tools import build_er_graph
    from Core.AgentSchema.graph_construction_tool_contracts import BuildERGraphInputs

    config = mcp_srv._state["config"]
    llm = mcp_srv._state["llm"]
    encoder = mcp_srv._state["encoder"]
    chunk_factory = mcp_srv._state["chunk_factory"]
    context = mcp_srv._state["context"]

    build_inputs = BuildERGraphInputs(
        target_dataset_name=DATASET,
        force_rebuild=REBUILD,
    )
    build_result = await build_er_graph(
        build_inputs, config, llm, encoder, chunk_factory
    )
    graph_instance = getattr(build_result, "graph_instance", None)
    if graph_instance:
        if hasattr(graph_instance, "_graph") and hasattr(
            graph_instance._graph, "namespace"
        ):
            graph_instance._graph.namespace = chunk_factory.get_namespace(DATASET)
        context.add_graph_instance(GRAPH_ID, graph_instance)

    graph_ok = build_result.status == "success" and GRAPH_ID in context.list_graphs()
    record(
        "A3: build/load ER graph",
        graph_ok,
        (
            f"graph_id={GRAPH_ID}, status={build_result.status}, "
            f"nodes={getattr(build_result, 'node_count', None)}, "
            f"edges={getattr(build_result, 'edge_count', None)}"
        ),
    )
    if not graph_ok:
        return print_summary()

    # Build/load entity VDB.
    from Core.AgentTools.entity_vdb_tools import entity_vdb_build_tool
    from Core.AgentSchema.tool_contracts import EntityVDBBuildInputs

    vdb_inputs = EntityVDBBuildInputs(
        graph_reference_id=GRAPH_ID,
        vdb_collection_name=VDB_ID,
        force_rebuild=REBUILD,
    )
    vdb_result = await entity_vdb_build_tool(vdb_inputs, context)
    vdb_ok = vdb_result.num_entities_indexed > 0
    record(
        "A4: build/load entity VDB",
        vdb_ok,
        f"entities_indexed={vdb_result.num_entities_indexed}",
    )
    if not vdb_ok:
        return print_summary()

    # ================================================================
    # B. DISCOVERY TOOLS
    # ================================================================
    print("\n--- B. Discovery Tools ---")

    resources_json = await mcp_srv.list_available_resources()
    resources = json.loads(resources_json)
    has_graphs = GRAPH_ID in resources.get("graphs", [])
    has_vdbs = VDB_ID in resources.get("vdbs", [])
    record(
        "B1: list_available_resources()",
        has_graphs and has_vdbs,
        f"graphs={resources.get('graphs')}, vdbs={resources.get('vdbs')}",
    )

    methods_json = await mcp_srv.list_methods()
    methods = json.loads(methods_json)
    record(
        "B2: list_methods()",
        len(methods) == 10,
        f"count={len(methods)}, names={[m['name'] for m in methods]}",
    )

    types_json = await mcp_srv.list_graph_types()
    types_list = json.loads(types_json)
    record(
        "B3: list_graph_types()",
        len(types_list) == 5,
        f"count={len(types_list)}, names={[t['name'] for t in types_list]}",
    )

    # ================================================================
    # C. OPERATOR COMPOSITION CHAIN 1 -- basic local pattern
    # ================================================================
    print("\n--- C. Operator Composition Chain 1 (basic local) ---")

    search_json = await mcp_srv.entity_vdb_search(
        VDB_ID, "Zorathian Empire", top_k=5
    )
    search_result = json.loads(search_json)
    similar = search_result.get("similar_entities", [])
    entity_names = [e["entity_name"] for e in similar]
    record(
        "C1: entity_vdb_search('Zorathian Empire')",
        len(entity_names) > 0,
        f"found {len(entity_names)} entities: {entity_names[:3]}",
    )

    chunks = []
    if entity_names:
        onehop_json = await mcp_srv.entity_onehop(entity_names[:3], GRAPH_ID)
        onehop_result = json.loads(onehop_json)
        total_neighbors = onehop_result.get("total_neighbors_found", 0)
        record(
            "C2: entity_onehop()",
            total_neighbors > 0,
            f"total_neighbors={total_neighbors}",
        )

        rel_json = await mcp_srv.relationship_onehop(entity_names[:3], GRAPH_ID)
        rel_result = json.loads(rel_json)
        rels = rel_result.get("one_hop_relationships", [])
        record(
            "C3: relationship_onehop()",
            len(rels) > 0,
            f"relationships={len(rels)}",
        )

        chunk_json = await mcp_srv.chunk_get_text(GRAPH_ID, entity_names[:3])
        chunk_result = json.loads(chunk_json)
        chunks = chunk_result.get("retrieved_chunks", [])
        record(
            "C4: chunk_get_text()",
            len(chunks) > 0,
            f"chunks={len(chunks)}",
        )
    else:
        for step in [
            "C2: entity_onehop()",
            "C3: relationship_onehop()",
            "C4: chunk_get_text()",
        ]:
            record(step, False, "SKIPPED -- no entities from C1")

    # ================================================================
    # D. MODEL-ASSISTED OPERATOR PATH
    # ================================================================
    print("\n--- D. Model-assisted operator path ---")

    extract_json = await mcp_srv.meta_extract_entities(
        "What is the connection between crystal technology and the Zorathian Empire?"
    )
    extract_result = json.loads(extract_json)
    extracted = extract_result.get("entities", [])
    extracted_names = [e["entity_name"] for e in extracted] if extracted else []
    record(
        "D1: meta_extract_entities()",
        len(extracted_names) > 0,
        f"extracted={extracted_names}",
    )

    if extracted_names:
        link_json = await mcp_srv.entity_link(
            extracted_names, VDB_ID, similarity_threshold=0.1
        )
        link_result = json.loads(link_json)
        linked_pairs = link_result.get("linked_entities_results", [])
        linked_ids = [
            pair["linked_entity_id"]
            for pair in linked_pairs
            if pair.get("linked_entity_id") and pair.get("link_status") == "linked"
        ]
        record(
            "D2: entity_link()",
            len(linked_ids) > 0,
            f"linked={len(linked_ids)} of {len(extracted_names)}",
        )

        ppr_seeds = linked_ids[:3] if linked_ids else entity_names[:3]
        if ppr_seeds:
            ppr_json = await mcp_srv.entity_ppr(GRAPH_ID, ppr_seeds, top_k=10)
            ppr_result = json.loads(ppr_json)
            ranked = ppr_result.get("ranked_entities", [])
            record(
                "D3: entity_ppr()",
                len(ranked) > 0,
                f"ranked={len(ranked)} entities",
            )
        else:
            record("D3: entity_ppr()", False, "SKIPPED -- no seed entities")
    else:
        record("D2: entity_link()", False, "SKIPPED -- no entities extracted")
        record("D3: entity_ppr()", False, "SKIPPED -- no entities extracted")

    # ================================================================
    # E. ANSWER GENERATION
    # ================================================================
    print("\n--- E. Answer Generation ---")

    test_chunks = []
    if entity_names and chunks:
        for chunk in chunks[:3]:
            text = chunk.get("text_content", chunk.get("text", ""))
            if text:
                test_chunks.append(text[:500])

    if not test_chunks:
        test_chunks = [
            "The Zorathian Empire was known for its crystal technology, "
            "which powered their floating cities and advanced weapons."
        ]

    answer = await mcp_srv.meta_generate_answer(
        "What is crystal technology?", test_chunks
    )
    answer_ok = len(answer) > 10 and answer != "Failed to generate answer."
    record(
        "E1: meta_generate_answer()",
        answer_ok,
        f"answer_len={len(answer)}, preview={answer[:100]}...",
    )

    # ================================================================
    # F. NAMED METHOD (convenience shortcut)
    # ================================================================
    print("\n--- F. Named Method ---")

    try:
        method_json = await mcp_srv.execute_method(
            "basic_local", "What is crystal technology?", DATASET
        )
        method_result = json.loads(method_json)
        method_ok = (
            "final_output" in method_result or "all_step_outputs" in method_result
        )
        record(
            "F1: execute_method('basic_local')",
            method_ok,
            f"keys={list(method_result.keys())}",
        )
    except Exception as exc:
        record(
            "F1: execute_method('basic_local')",
            False,
            f"ERROR: {exc}",
        )
        traceback.print_exc()

    return print_summary()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
