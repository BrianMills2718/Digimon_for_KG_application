#!/usr/bin/env python3
"""End-to-end QA evaluation on HotPotQA using the NEW operator pipeline.

Uses PipelineExecutor + basic_local method plan to compare with old system.
"""
import asyncio
import sys
import os
import json
import time

sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATASET = 'HotpotQAsmallest'


async def setup():
    """Set up graph, VDB, and context — same as old eval."""
    from Option.Config2 import Config
    from Core.Provider.LiteLLMProvider import LiteLLMProvider
    from Core.Index.EmbeddingFactory import get_rag_embedding
    from Core.Chunk.ChunkFactory import ChunkFactory
    from Core.AgentSchema.context import GraphRAGContext
    from Core.AgentTools.graph_construction_tools import build_er_graph
    from Core.AgentSchema.graph_construction_tool_contracts import BuildERGraphInputs

    config = Config.from_yaml_file('Option/Config2.yaml')
    llm = LiteLLMProvider(config.llm)
    encoder = get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    context = GraphRAGContext(
        target_dataset_name=DATASET, main_config=config,
        llm_provider=llm, embedding_provider=encoder, chunk_storage_manager=chunk_factory,
    )

    # Build ER graph (cached)
    build_inputs = BuildERGraphInputs(target_dataset_name=DATASET, force_rebuild=False)
    build_result = await build_er_graph(build_inputs, config, llm, encoder, chunk_factory)
    gi = getattr(build_result, 'graph_instance', None)
    if gi:
        if hasattr(gi, '_graph') and hasattr(gi._graph, 'namespace'):
            gi._graph.namespace = chunk_factory.get_namespace(DATASET)
        context.add_graph_instance(build_result.graph_id, gi)
    GRAPH_ID = build_result.graph_id

    # Build VDB (or use cached)
    from Core.AgentTools.entity_vdb_tools import entity_vdb_build_tool
    from Core.AgentSchema.tool_contracts import EntityVDBBuildInputs
    vdb_id = f'{DATASET}_entities'
    vdb_params = EntityVDBBuildInputs(
        graph_reference_id=GRAPH_ID,
        vdb_collection_name=vdb_id,
        force_rebuild=False
    )
    vdb_result = await entity_vdb_build_tool(vdb_params, context)
    print(f"Graph: {GRAPH_ID} | VDB: {vdb_result.vdb_reference_id} ({vdb_result.num_entities_indexed} entities)")

    return config, llm, context, GRAPH_ID, vdb_id


class ChunkLookup:
    """Wrapper providing get_data_by_key for operators from ChunkFactory data."""
    def __init__(self, chunks_dict):
        self._chunks = chunks_dict

    async def get_data_by_key(self, chunk_id):
        return self._chunks.get(chunk_id)


class GraphStorageProxy:
    """Proxy that adds entity_metakey to NetworkXStorage (which is Pydantic and frozen)."""
    def __init__(self, storage, entity_metakey="entity_name"):
        self._storage = storage
        self.entity_metakey = entity_metakey

    def __getattr__(self, name):
        if name in ('_storage', 'entity_metakey'):
            raise AttributeError(name)
        return getattr(self._storage, name)


async def build_operator_context(context, graph_id, vdb_id, llm, config):
    """Build an OperatorContext from the existing GraphRAGContext."""
    from Core.Operators._context import OperatorContext

    graph_instance = context.get_graph_instance(graph_id)
    graph_storage = GraphStorageProxy(graph_instance._graph, graph_instance.entity_metakey)

    # Get VDB
    entities_vdb = context.get_vdb_instance(vdb_id) if hasattr(context, 'get_vdb_instance') else None
    if entities_vdb is None and hasattr(context, 'vdbs'):
        entities_vdb = context.vdbs.get(vdb_id)

    # Build chunk lookup from ChunkFactory
    chunk_storage = context.chunk_storage_manager
    chunks_dict = {}
    if chunk_storage:
        dataset_name = graph_id
        for suffix in ["_ERGraph", "_RKGraph", "_TreeGraph", "_PassageGraph"]:
            if dataset_name.endswith(suffix):
                dataset_name = dataset_name[:-len(suffix)]
                break
        chunks_list = await chunk_storage.get_chunks_for_dataset(dataset_name)
        for chunk_id, chunk_obj in chunks_list:
            chunks_dict[chunk_id] = chunk_obj.content if hasattr(chunk_obj, 'content') else str(chunk_obj)
    doc_chunks = ChunkLookup(chunks_dict)
    print(f"Loaded {len(chunks_dict)} chunks for lookup")

    retriever_config = config.retriever if hasattr(config, 'retriever') else None

    return OperatorContext(
        graph=graph_storage,
        entities_vdb=entities_vdb,
        doc_chunks=doc_chunks,
        llm=llm,
        config=retriever_config,
    )


async def retrieve_and_answer(question, op_ctx, llm):
    """Run basic_local pipeline via direct operator calls."""
    from Core.Schema.SlotTypes import SlotKind, SlotValue
    from Core.Operators.entity.vdb import entity_vdb
    from Core.Operators.relationship.onehop import relationship_onehop
    from Core.Operators.chunk.occurrence import chunk_occurrence
    from Core.Operators.meta.generate_answer import meta_generate_answer

    # Step 1: Entity VDB search
    query_slot = SlotValue(kind=SlotKind.QUERY_TEXT, data=question, producer="input")
    try:
        entity_result = await entity_vdb(
            {"query": query_slot}, op_ctx, {"top_k": 10}
        )
    except Exception as e:
        return f"Error in entity.vdb: {e}", ""

    entities = entity_result.get("entities")
    if not entities or not entities.data:
        return "insufficient information", ""

    # Step 2: Relationship one-hop
    try:
        rel_result = await relationship_onehop(
            {"entities": entities}, op_ctx
        )
    except Exception as e:
        rel_result = {"relationships": SlotValue(kind=SlotKind.RELATIONSHIP_SET, data=[], producer="error")}

    # Step 3: Chunk occurrence
    try:
        chunk_result = await chunk_occurrence(
            {"entities": entities}, op_ctx
        )
    except Exception as e:
        chunk_result = {"chunks": SlotValue(kind=SlotKind.CHUNK_SET, data=[], producer="error")}

    chunks = chunk_result.get("chunks")
    if not chunks or not chunks.data:
        # Fallback: build context from entity descriptions
        ent_descs = []
        for e in entities.data[:10]:
            if e.description:
                ent_descs.append(f"{e.entity_name}: {e.description}")
        if ent_descs:
            from Core.Schema.SlotTypes import ChunkRecord
            fallback_chunks = [ChunkRecord(chunk_id="fallback", text="\n".join(ent_descs))]
            chunks = SlotValue(kind=SlotKind.CHUNK_SET, data=fallback_chunks, producer="fallback")

    if not chunks or not chunks.data:
        return "insufficient information", ""

    # Step 4: Generate answer
    try:
        answer_result = await meta_generate_answer(
            {"query": query_slot, "chunks": chunks}, op_ctx
        )
        answer = answer_result["answer"].data
    except Exception as e:
        answer = f"Error in generate_answer: {e}"

    # Build debug context string
    ctx_parts = []
    if entities and entities.data:
        ctx_parts.append(f"Entities: {[e.entity_name for e in entities.data[:10]]}")
    rels = rel_result.get("relationships")
    if rels and rels.data:
        ctx_parts.append(f"Relationships: {len(rels.data)}")
    if chunks and chunks.data:
        ctx_parts.append(f"Chunks: {len(chunks.data)}")
    debug_ctx = " | ".join(ctx_parts)

    return answer, debug_ctx


async def judge_answer(llm, question: str, predicted: str, gold: str) -> dict:
    """Use LLM to judge if the predicted answer matches the gold answer."""
    prompt = f"""You are a strict QA evaluation judge. Your ONLY job is to determine if the predicted answer matches the gold answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Rules:
- If the predicted answer contains the gold answer (or vice versa), mark it CORRECT
- Semantic equivalence counts (e.g., "yes" = "Yes, they are both American")
- Partial matches count if the key information is present (e.g., "Greenwich Village" matches "Greenwich Village, New York City")
- "insufficient information" or "I don't know" is ALWAYS WRONG unless the gold answer is also unknown
- Do NOT evaluate whether the gold answer is factually correct — just whether the prediction matches it

Respond with ONLY a JSON object: {{"correct": true/false, "reason": "brief explanation"}}"""

    result = await llm.aask(prompt, format="json")
    try:
        from Core.Common.Utils import prase_json_from_response
        parsed = prase_json_from_response(result)
        return parsed if isinstance(parsed, dict) else {"correct": False, "reason": "parse error"}
    except:
        return {"correct": False, "reason": f"parse error: {result[:100]}"}


async def main():
    # Load questions
    with open(f'Data/{DATASET}/Question.json') as f:
        questions = [json.loads(line) for line in f]
    print(f"Loaded {len(questions)} questions from {DATASET}")

    # Setup
    config, llm, context, graph_id, vdb_id = await setup()

    # Build operator context
    op_ctx = await build_operator_context(context, graph_id, vdb_id, llm, config)
    print(f"OperatorContext built: graph={op_ctx.graph is not None}, vdb={op_ctx.entities_vdb is not None}, chunks={op_ctx.doc_chunks is not None}")

    # Evaluate each question
    results = []
    total_time = 0

    print(f"\n{'='*70}")
    print(f"PIPELINE QA EVALUATION — {len(questions)} QUESTIONS")
    print(f"{'='*70}\n")

    for i, q in enumerate(questions):
        t0 = time.time()
        question = q['question']
        gold = q['answer']
        q_type = q.get('type', '?')

        print(f"Q{i+1}/{len(questions)} [{q_type}]: {question}")
        print(f"  Gold: {gold}")

        # Run pipeline
        predicted, debug_ctx = await retrieve_and_answer(question, op_ctx, llm)
        print(f"  Predicted: {predicted}")
        if debug_ctx:
            print(f"  Context: {debug_ctx}")

        # Judge
        judgment = await judge_answer(llm, question, predicted, gold)
        correct = judgment.get('correct', False)
        reason = judgment.get('reason', '')
        elapsed = time.time() - t0
        total_time += elapsed

        status = "CORRECT" if correct else "WRONG"
        print(f"  [{status}] {reason} ({elapsed:.1f}s)")
        print()

        results.append({
            'question': question,
            'gold': gold,
            'predicted': predicted,
            'correct': correct,
            'reason': reason,
            'type': q_type,
            'time': elapsed
        })

    # Summary
    n_correct = sum(1 for r in results if r['correct'])
    n_total = len(results)
    accuracy = n_correct / n_total if n_total > 0 else 0

    print(f"\n{'='*70}")
    print(f"PIPELINE QA EVALUATION RESULTS — {DATASET}")
    print(f"{'='*70}")
    print(f"  Accuracy: {n_correct}/{n_total} ({accuracy:.0%})")
    print(f"  Total time: {total_time:.1f}s ({total_time/n_total:.1f}s avg)")

    # By type
    from collections import Counter
    type_correct = Counter()
    type_total = Counter()
    for r in results:
        type_total[r['type']] += 1
        if r['correct']:
            type_correct[r['type']] += 1
    print(f"\n  By type:")
    for t in sorted(type_total.keys()):
        print(f"    {t}: {type_correct[t]}/{type_total[t]} ({type_correct[t]/type_total[t]:.0%})")

    # Failures
    failures = [r for r in results if not r['correct']]
    if failures:
        print(f"\n  Failures:")
        for r in failures:
            print(f"    Q: {r['question']}")
            print(f"      Gold: {r['gold']}")
            print(f"      Predicted: {r['predicted']}")
            print()

    # Save
    with open(f'results/{DATASET}_pipeline_qa_eval.json', 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'n_correct': n_correct,
            'n_total': n_total,
            'avg_time': total_time / n_total,
            'results': results
        }, f, indent=2)
    print(f"  Results saved to results/{DATASET}_pipeline_qa_eval.json")

    # Compare with old system
    old_results_path = f'results/{DATASET}_qa_eval.json'
    if os.path.exists(old_results_path):
        with open(old_results_path) as f:
            old = json.load(f)
        old_acc = old.get('accuracy', 0)
        print(f"\n  COMPARISON:")
        print(f"    Old system: {old_acc:.0%}")
        print(f"    New pipeline: {accuracy:.0%}")
        if accuracy >= old_acc:
            print(f"    Status: PASS (new >= old)")
        else:
            print(f"    Status: FAIL (new < old)")


asyncio.run(main())
