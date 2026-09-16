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
            prepare_json = await mcp_srv.corpus_prepare(str(SOURCE_DIRECTORY), DATASET)
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

    try:
        graph_json = await mcp_srv.graph_build_er(DATASET, force_rebuild=REBUILD)
        graph_result = json.loads(graph_json)
        context = mcp_srv._state["context"]
        graph_ok = (
            graph_result.get("status") == "success"
            and GRAPH_ID in context.list_graphs()
        )
        record(
            "A3: graph_build_er()",
            graph_ok,
            (
                f"graph_id={GRAPH_ID}, status={graph_result.get('status')}, "
                f"nodes={graph_result.get('node_count')}, "
                f"edges={graph_result.get('edge_count')}"
            ),
        )
        if not graph_ok:
            return print_summary()
    except Exception as exc:
        record("A3: graph_build_er()", False, f"ERROR: {exc}")
        traceback.print_exc()
        return print_summary()

    try:
        vdb_json = await mcp_srv.entity_vdb_build(
            GRAPH_ID,
            VDB_ID,
            force_rebuild=REBUILD,
        )
        vdb_result = json.loads(vdb_json)
        vdb_ok = (
            vdb_result.get("num_entities_indexed", 0) > 0
            and not str(vdb_result.get("status", "")).lower().startswith("error")
        )
        record(
            "A4: entity_vdb_build()",
            vdb_ok,
            (
                f"status={vdb_result.get('status')}, "
                f"entities_indexed={vdb_result.get('num_entities_indexed', 0)}"
            ),
        )
        if not vdb_ok:
            return print_summary()
    except Exception as exc:
        record("A4: entity_vdb_build()", False, f"ERROR: {exc}")
        traceback.print_exc()
        return print_summary()

    # ================================================================
    # B. DISCOVERY TOOLS
    # ================================================================
    print("\n--- B. Discovery Tools ---")

    resources = json.loads(await mcp_srv.list_available_resources())
    has_graphs = GRAPH_ID in resources.get("graphs", [])
    has_vdbs = VDB_ID in resources.get("vdbs", [])
    record(
        "B1: list_available_resources()",
        has_graphs and has_vdbs,
        f"graphs={resources.get('graphs')}, vdbs={resources.get('vdbs')}",
    )

    methods = json.loads(await mcp_srv.list_methods())
    record(
        "B2: list_methods()",
        len(methods) == 10,
        f"count={len(methods)}, names={[m['name'] for m in methods]}",
    )

    types_list = json.loads(await mcp_srv.list_graph_types())
    record(
        "B3: list_graph_types()",
        len(types_list) == 5,
        f"count={len(types_list)}, names={[t['name'] for t in types_list]}",
    )

    # ================================================================
    # C. INDIVIDUAL MCP CAPABILITY PATH
    # ================================================================
    print("\n--- C. Individual MCP capability path ---")

    search_result = json.loads(
        await mcp_srv.entity_vdb_search(VDB_ID, "Zorathian Empire", top_k=5)
    )
    similar = search_result.get("similar_entities", [])
    entity_names = [entity["entity_name"] for entity in similar]
    record(
        "C1: entity_vdb_search('Zorathian Empire')",
        len(entity_names) > 0,
        f"found {len(entity_names)} entities: {entity_names[:3]}",
    )

    chunks = []
    if entity_names:
        onehop_result = json.loads(
            await mcp_srv.entity_onehop(entity_names[:3], GRAPH_ID)
        )
        total_neighbors = onehop_result.get("total_neighbors_found", 0)
        record(
            "C2: entity_onehop()",
            total_neighbors > 0,
            f"total_neighbors={total_neighbors}",
        )

        rel_result = json.loads(
            await mcp_srv.relationship_onehop(entity_names[:3], GRAPH_ID)
        )
        rels = rel_result.get("one_hop_relationships", [])
        record(
            "C3: relationship_onehop()",
            len(rels) > 0,
            f"relationships={len(rels)}",
        )

        chunk_result = json.loads(
            await mcp_srv.chunk_get_text(GRAPH_ID, entity_names[:3])
        )
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

    extract_result = json.loads(
        await mcp_srv.meta_extract_entities(
            "What is the connection between crystal technology and the Zorathian Empire?"
        )
    )
    extracted = extract_result.get("entities", [])
    extracted_names = [entity["entity_name"] for entity in extracted] if extracted else []
    record(
        "D1: meta_extract_entities()",
        len(extracted_names) > 0,
        f"extracted={extracted_names}",
    )

    if extracted_names:
        link_result = json.loads(
            await mcp_srv.entity_link(
                extracted_names,
                VDB_ID,
                similarity_threshold=0.1,
            )
        )
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
            ppr_result = json.loads(
                await mcp_srv.entity_ppr(GRAPH_ID, ppr_seeds, top_k=10)
            )
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
    # E. GROUNDED ANSWER GENERATION
    # ================================================================
    print("\n--- E. Grounded answer generation ---")

    test_chunks = []
    for chunk in chunks[:3]:
        text = chunk.get("text_content", chunk.get("text", ""))
        if text:
            test_chunks.append(text[:500])

    if not test_chunks:
        record(
            "E1: meta_generate_answer()",
            False,
            "SKIPPED -- no retrieved source evidence; no fabricated fallback allowed",
        )
    else:
        answer = await mcp_srv.meta_generate_answer(
            "What is crystal technology?",
            test_chunks,
        )
        answer_ok = len(answer) > 10 and answer != "Failed to generate answer."
        record(
            "E1: meta_generate_answer()",
            answer_ok,
            f"answer_len={len(answer)}, preview={answer[:100]}...",
        )

    # ================================================================
    # F. NAMED METHOD
    # ================================================================
    print("\n--- F. Named Method ---")

    try:
        method_result = json.loads(
            await mcp_srv.execute_method(
                "basic_local",
                "What is crystal technology?",
                DATASET,
            )
        )
        if "error" in method_result:
            method_ok = False
            detail = method_result["error"]
        else:
            final_output = method_result.get("final_output", {})
            method_chunks = final_output.get("chunks", [])
            method_ok = len(method_chunks) > 0
            detail = (
                f"final_chunks={len(method_chunks)}, "
                f"steps={list(method_result.get('all_step_outputs', {}).keys())}"
            )
        record(
            "F1: execute_method('basic_local') returns evidence",
            method_ok,
            detail,
        )
    except Exception as exc:
        record(
            "F1: execute_method('basic_local') returns evidence",
            False,
            f"ERROR: {exc}",
        )
        traceback.print_exc()

    return print_summary()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
