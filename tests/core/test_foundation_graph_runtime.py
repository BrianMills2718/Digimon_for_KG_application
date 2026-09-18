"""Real GraphML/NetworkXStorage/typed-operator/evidence integration; synthetic input."""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import networkx as nx
import pytest

from Core.Operators._context import OperatorContext
from Core.Projection.Execution import file_sha256
from Core.Projection.GraphRuntime import (
    open_foundation_graph_context, retrieve_foundation_subgraph, simple_entity_view,
)
from Core.Projection.Project import FoundationProject, build_foundation_project
from Core.Schema.SlotTypes import SlotKind
from Core.Storage.NetworkXStorage import NetworkXStorage


def fixture_payloads():
    # B has the same display label as A; IDs must remain distinct.
    pairs = [("a1", "knows", ["a", "b"]), ("a2", "advises", ["a", "b"]),
             ("a3", "knows", ["b", "c"]), ("a4", "knows", ["c", "d"]),
             ("nary", "transfer", ["a", "b", "outside"]),
             ("loop", "self_report", ["isolated", "isolated"])]
    assertions, passages = [], []
    for aid, predicate, nodes in pairs:
        assertions.append({"assertion_id": aid, "predicate": predicate,
                           "roles": {f"role{i}": [{"kind": "entity", "entity_id": f"entity:{node}",
                                                    "name": "Same Label" if node in ("a", "b") else node,
                                                    "entity_type": "Person"}] for i, node in enumerate(nodes)},
                           "qualifiers": {"sys:polarity": "negative"} if aid == "a2" else {},
                           "provenance_refs": [f"cand:{aid}"], "namespace_id": "demo", "source_registry_id": "test"})
        passages.append({"passage_id": f"passage:{aid}", "source_ref": f"source:{aid}",
                         "text": f"  Exact source for {aid}.\n", "namespace_id": "demo", "source_registry_id": "test",
                         "supporting_provenance_refs": [f"cand:{aid}"]})
    return {"format_version": "1.3", "producer": "onto-canon6", "assertion_count": len(assertions), "assertions": assertions}, \
           {"format_version": "1.0", "producer": "onto-canon6", "passage_count": len(passages), "passages": passages}


@pytest.fixture
def project(tmp_path):
    source, windows = fixture_payloads()
    ir = tmp_path / "foundation.json"; passages = tmp_path / "passages.json"
    ir.write_text(json.dumps(source)); passages.write_text(json.dumps(windows))
    return build_foundation_project(ir, tmp_path / "project", passage_path=passages)


def events(project):
    return [json.loads(line) for line in (project.root / "executions.jsonl").read_text().splitlines()]


def test_binds_real_storage_and_public_operator_context(project):
    ctx, artifact = asyncio.run(open_foundation_graph_context(project))
    assert isinstance(ctx, OperatorContext)
    assert type(ctx.graph) is NetworkXStorage  # not a mock or demo graph engine
    assert ctx.graph.get_edge_num() == 4
    assert ctx.graph.graph.nodes["entity:a"]["entity_name"] == "entity:a"
    assert ctx.graph.graph.nodes["entity:b"]["entity_name"] == "entity:b"
    edge = asyncio.run(ctx.graph.get_edge("entity:a", "entity:b"))
    assert json.loads(edge["assertion_ids_json"]) == ["a1", "a2"]
    assert edge["assertion_count"] == 2 and edge["weight"] == 1
    originals = json.loads(edge["assertions_json"])
    assert json.loads(originals[1]["qualifiers_json"])["sys:polarity"] == "negative"
    assert file_sha256(project.root / artifact["path"]) == artifact["sha256"]
    # Native binary artifact still has both separate edges and the n-ary isolate.
    original = nx.read_graphml(project.root / project.manifest["artifacts"]["binary_graph"]["path"],force_multigraph=True)
    assert original.number_of_edges("entity:a", "entity:b") == 2


@pytest.mark.parametrize("k,nodes,aids", [(1,{"a","b"},{"a1","a2"}), (2,{"a","b","c"},{"a1","a2","a3"}), (3,{"a","b","c","d"},{"a1","a2","a3","a4"})])
def test_real_khop_matches_independent_expected_working_set(project,k,nodes,aids):
    slots=asyncio.run(retrieve_foundation_subgraph(project,["entity:a"],k=k))
    assert slots["subgraph"].kind is SlotKind.SUBGRAPH
    assert slots["chunks"].kind is SlotKind.CHUNK_SET
    assert slots["entities"].kind is SlotKind.ENTITY_SET
    subgraph=slots["subgraph"].data
    assert subgraph.nodes=={f"entity:{n}" for n in nodes}
    assert set(slots["subgraph"].metadata["assertion_ids"])==aids
    assert {c.chunk_id for c in slots["chunks"].data}=={f"passage:{aid}" for aid in aids}
    assert subgraph.nx_graph.number_of_nodes()==len(nodes)
    for chunk in slots["chunks"].data:
        aid=chunk.chunk_id.split(":")[1]
        assert chunk.text==f"  Exact source for {aid}.\n"
        assert chunk.extra["assertion_ids"]==[aid]
        assert chunk.extra["namespace_id"]=="demo"
        assert chunk.extra["source_ref"]==f"source:{aid}"
    # Evidence from B-C, C-D or a ternary involving A must not leak into 1 hop.
    assert "passage:nary" not in {c.chunk_id for c in slots["chunks"].data}


def test_filter_before_parallel_aggregation_does_not_leak_other_claims(project):
    result=asyncio.run(project.graph_neighborhood(["entity:a"],k=3,predicates=["advises"]))
    assert result["nodes"]==["entity:a","entity:b"]
    assert result["assertion_ids"]==["a2"]
    assert [c["chunk_id"] for c in result["evidence"]]==["passage:a2"]
    assert result["skipped_nonbinary_assertions"]==["nary"]
    assert result["graph_scope"]["polarity_policy"].startswith("retained")


def test_empty_predicate_filter_is_not_treated_as_all(project):
    result=asyncio.run(project.graph_neighborhood(["entity:a"],predicates=[]))
    assert result["status"]=="insufficient_evidence"
    assert result["nodes"]==["entity:a"] and result["edges"]==[]


def test_isolate_is_not_fabricated_as_connected(project):
    result=asyncio.run(project.graph_neighborhood(["entity:outside"]))
    assert result["status"]=="insufficient_evidence"
    assert result["evidence"]==[] and result["nodes"]==["entity:outside"]
    # Still accessible in the separate entity-wide SQL evidence path.
    assert project.evidence_for_entity("entity:outside")["status"]=="ok"


def test_self_loop_policy_is_explicit(project):
    result=asyncio.run(project.graph_neighborhood(["entity:isolated"]))
    assert result["assertion_ids"]==["loop"]
    assert result["edges"]==[("entity:isolated","entity:isolated")]
    assert result["graph_scope"]["self_loop_policy"]=="retain"


def test_unknown_seed_is_a_failed_execution_not_a_plausible_empty_success(project):
    with pytest.raises(KeyError,match="unknown canonical"):
        asyncio.run(project.graph_neighborhood(["entity:missing"]))
    failure=events(project)[-1]
    assert failure["operation"]=="subgraph.khop_paths" and failure["status"]=="failed"
    assert failure["outputs"]==[] and failure["diagnostics"]["unknown_entity_ids"]==["entity:missing"]


@pytest.mark.parametrize("bad",[0,-1,True,1.5,"2"])
def test_invalid_hop_count_is_not_silently_coerced(project,bad):
    with pytest.raises(ValueError,match="positive integer"):
        asyncio.run(project.graph_neighborhood(["entity:a"],k=bad))


def test_stale_binary_rejected_on_held_project(project):
    p=project.root / project.manifest["artifacts"]["binary_graph"]["path"]
    p.write_text(p.read_text()+"\n")
    with pytest.raises(ValueError,match="stale graph input"):
        asyncio.run(project.graph_neighborhood(["entity:a"]))
    assert events(project)[-1]["status"]=="failed"


def test_operator_failure_does_not_become_empty_success(project,monkeypatch):
    async def fail(*args,**kwargs):
        raise RuntimeError("injected traversal failure")
    monkeypatch.setattr(NetworkXStorage,"find_k_hop_neighbors_batch",fail)
    with pytest.raises(RuntimeError,match="injected traversal failure"):
        asyncio.run(project.graph_neighborhood(["entity:a"]))
    assert events(project)[-1]["outputs"]==[]


def test_lineage_links_exact_runtime_subgraph_and_evidence_artifacts(project):
    result=asyncio.run(project.graph_neighborhood(["entity:a"],k=1))
    terminal={e["execution_id"]:e for e in events(project) if e["status"]=="succeeded"}
    for key in ("artifact","subgraph_artifact","runtime_graph_artifact"):
        ref=result[key]
        assert file_sha256(project.root/ref["path"])==ref["sha256"]
        assert ref["artifact_id"] in [a["artifact_id"] for a in terminal[ref["producing_execution"]]["outputs"]]
    evidence=terminal[result["execution_id"]]
    assert evidence["inputs"][0]["artifact_id"]==result["subgraph_artifact"]["artifact_id"]
    retrieval=terminal[result["subgraph_artifact"]["producing_execution"]]
    assert retrieval["inputs"][0]["artifact_id"]==result["runtime_graph_artifact"]["artifact_id"]
    assert "Core.Operators.subgraph.khop_paths" in retrieval["diagnostics"]["consumer_implementations"]


def test_no_passage_companion_means_no_fabricated_source(project,tmp_path):
    source=project.root/project.manifest["artifacts"]["foundation"]["path"]
    no_evidence=build_foundation_project(source,tmp_path/"no-evidence")
    result=asyncio.run(no_evidence.graph_neighborhood(["entity:a"]))
    assert result["status"]=="insufficient_evidence" and result["evidence"]==[]
    assert result["assertion_ids"]==["a1","a2","a3"]


def test_saved_project_cli_build_and_reuse_include_real_graph_path(tmp_path):
    repo=Path(__file__).resolve().parents[2]
    fixture=repo/"tests/fixtures/foundation_demo"
    cmd=[sys.executable,str(repo/"scripts/run_foundation_demo.py"),"--ir",str(fixture/"foundation.json"),"--passages",str(fixture/"passages.json"),"--output",str(tmp_path/"cli-project"),"--entity-id","entity:alice","--graph-hops","2"]
    first=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
    assert first.returncode==0,first.stderr
    second=subprocess.run(cmd+["--reuse"],capture_output=True,text=True,timeout=20)
    assert second.returncode==0,second.stderr
    a,b=json.loads(first.stdout),json.loads(second.stdout)
    assert a["graph_result"]["evidence"]==b["graph_result"]["evidence"]
    assert a["graph_result"]["assertion_ids"]==["gassert2_binary"]
    assert "graph_runtime" not in a["not_integrated"]


def test_public_lazy_exports_keep_same_actual_functions():
    import Core.Operators as public
    from Core.Operators.subgraph.khop_paths import subgraph_khop_paths
    assert public.subgraph_khop_paths is subgraph_khop_paths
    assert public.OperatorContext is OperatorContext
    with pytest.raises(AttributeError):
        getattr(public,"not_a_real_operator")


def test_partial_evidence_is_explicit_not_reported_as_complete(tmp_path):
    source, windows = fixture_payloads()
    source["assertions"][1]["provenance_refs"] = []
    windows["passages"] = [p for p in windows["passages"] if p["passage_id"] != "passage:a2"]
    windows["passage_count"] = len(windows["passages"])
    ir, companion = tmp_path / "foundation.json", tmp_path / "passages.json"
    ir.write_text(json.dumps(source)); companion.write_text(json.dumps(windows))
    project = build_foundation_project(ir, tmp_path / "project", passage_path=companion)
    result = asyncio.run(project.graph_neighborhood(["entity:a"], k=1))
    assert result["status"] == "partial_evidence"
    assert result["assertions_without_passages"] == ["a2"]
    assert result["assertion_ids"] == ["a1", "a2"]
    assert [chunk["chunk_id"] for chunk in result["evidence"]] == ["passage:a1"]


def test_nontext_payload_is_not_stringified_into_evidence():
    from Core.Operators.subgraph.materialize import subgraph_materialize
    from Core.Schema.SlotTypes import SubgraphRecord, SlotValue
    class BadStore:
        async def get_data_by_key(self, _):
            return {"content": 12345}
    async def run():
        storage = NetworkXStorage()
        await storage.upsert_node("entity:a", {"source_id": "passage:bad"})
        return await subgraph_materialize({"subgraph": SlotValue(SlotKind.SUBGRAPH, SubgraphRecord({"entity:a"}, []))},
                                          OperatorContext(storage, doc_chunks=BadStore()))
    assert asyncio.run(run())["chunks"].data == []


def test_explicit_passage_ids_are_not_split_on_legacy_separator(tmp_path):
    source, windows = fixture_payloads()
    windows["passages"][0]["passage_id"] = "passage:opaque<SEP>identity"
    ir, companion = tmp_path / "foundation.json", tmp_path / "passages.json"
    ir.write_text(json.dumps(source)); companion.write_text(json.dumps(windows))
    project = build_foundation_project(ir, tmp_path / "project", passage_path=companion)
    result = asyncio.run(project.graph_neighborhood(["entity:a"], k=1))
    assert {chunk["chunk_id"] for chunk in result["evidence"]} == {"passage:opaque<SEP>identity", "passage:a2"}
