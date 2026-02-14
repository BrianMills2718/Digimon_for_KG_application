#!/usr/bin/env python3
"""End-to-end QA evaluation on HotPotQA using the graph retrieval pipeline.

Pipeline per question:
  1. Entity.VDBSearch (find relevant entities for the question)
  2. Relationship.OneHop (get relationships around those entities)
  3. Chunk.GetText (get source text for relevant entities)
  4. LLM answer generation (answer the question given retrieved context)
  5. LLM judge (compare predicted answer to gold answer)
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
    """Set up graph, VDB, and context."""
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


async def retrieve_context_for_question(question: str, context, graph_id: str, vdb_id: str):
    """Retrieve relevant context from the graph for a question.

    Multi-strategy retrieval:
    1. VDB search for relevant entities
    2. PPR from seed entities to find structurally related entities
    3. Get relationships around all relevant entities
    4. Get source text chunks
    """
    from Core.AgentTools.entity_tools import entity_vdb_search_tool, entity_ppr_tool
    from Core.AgentSchema.tool_contracts import EntityVDBSearchInputs, EntityPPRInputs
    from Core.AgentTools.chunk_tools import chunk_get_text_for_entities_tool

    graph_instance = context.get_graph_instance(graph_id)
    nx_g = graph_instance._graph.graph
    all_graph_nodes = set(nx_g.nodes())

    # Step 0: Direct entity name matching from the question
    # Extract potential entity names by matching graph nodes against question text
    question_lower = question.lower()
    direct_matches = []
    for node in all_graph_nodes:
        # Match multi-word entity names that appear in the question
        if len(node) > 2 and node.lower() in question_lower:
            direct_matches.append(node)
    # Sort by length (longer = more specific matches first)
    direct_matches.sort(key=len, reverse=True)

    # Step 1: VDB search for relevant entities
    search_params = EntityVDBSearchInputs(
        vdb_reference_id=vdb_id,
        query_text=question,
        top_k_results=10
    )
    search_result = await entity_vdb_search_tool(search_params, context)
    vdb_entities = [e.entity_name for e in search_result.similar_entities]

    # Step 2: PPR from seed entities (direct matches + VDB top hits)
    seed_entities = list(dict.fromkeys(direct_matches[:5] + [e for e in vdb_entities[:5] if e in all_graph_nodes]))
    ppr_entities = []
    if seed_entities:
        try:
            ppr_params = EntityPPRInputs(
                graph_reference_id=graph_id,
                seed_entity_ids=seed_entities[:5],
                top_k_results=10
            )
            ppr_result = await entity_ppr_tool(ppr_params, context)
            ppr_entities = [name for name, score in ppr_result.ranked_entities]
        except Exception:
            pass

    # Combine: direct matches first, then PPR, then VDB
    all_entities = list(dict.fromkeys(direct_matches + ppr_entities + vdb_entities))

    if not all_entities:
        return "", [], []

    # Step 3: Get relationships around top entities
    relationships = []
    for entity in all_entities[:8]:
        if entity in nx_g.nodes():
            for neighbor in nx_g.neighbors(entity):
                edge_data = nx_g.edges[entity, neighbor]
                rel_name = edge_data.get('relation_name', '')
                desc = edge_data.get('description', '')
                rel_str = f"{entity} --[{rel_name}]--> {neighbor}"
                if desc:
                    rel_str += f" ({desc})"
                relationships.append(rel_str)

    # Step 4: Get source text chunks for more entities
    existing_entities = [e for e in all_entities if e in nx_g.nodes()]
    chunks_text = []
    if existing_entities:
        chunk_result = await chunk_get_text_for_entities_tool(
            {'graph_reference_id': graph_id, 'entity_ids': existing_entities[:8], 'max_chunks_per_entity': 3},
            context
        )
        if isinstance(chunk_result, dict):
            retrieved = chunk_result.get('retrieved_chunks', [])
            if isinstance(retrieved, list):
                seen = set()
                for c in retrieved:
                    text = c.get('text_content', '') if isinstance(c, dict) else str(c)
                    if text and text not in seen:
                        seen.add(text)
                        chunks_text.append(text)

    # Build context string — source text first (most important)
    parts = []
    if chunks_text:
        parts.append("Source text:\n" + "\n---\n".join(chunks_text[:8]))
    if relationships:
        parts.append("Graph relationships:\n" + "\n".join(relationships[:20]))
    if all_entities:
        parts.append("Relevant entities: " + ", ".join(all_entities[:15]))

    retrieved_context = "\n\n".join(parts)
    return retrieved_context, all_entities, relationships


async def answer_question(llm, question: str, retrieved_context: str) -> str:
    """Use LLM to answer the question given retrieved context."""
    prompt = f"""Answer the following question based on the provided context.
Give a concise, direct answer (a few words or one sentence max).
Use reasoning to combine information from multiple parts of the context if needed.
Only say "insufficient information" if the context truly contains nothing relevant.

Context:
{retrieved_context}

Question: {question}

Answer:"""
    answer = await llm.aask(prompt)
    return answer.strip()


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

    # Evaluate each question
    results = []
    total_time = 0

    print(f"\n{'='*70}")
    print(f"EVALUATING {len(questions)} QUESTIONS")
    print(f"{'='*70}\n")

    for i, q in enumerate(questions):
        t0 = time.time()
        question = q['question']
        gold = q['answer']
        q_type = q.get('type', '?')

        print(f"Q{i+1}/{len(questions)} [{q_type}]: {question}")
        print(f"  Gold: {gold}")

        # Retrieve context
        retrieved_context, entities, rels = await retrieve_context_for_question(
            question, context, graph_id, vdb_id
        )

        # Generate answer
        predicted = await answer_question(llm, question, retrieved_context)
        print(f"  Predicted: {predicted}")

        # Judge
        judgment = await judge_answer(llm, question, predicted, gold)
        correct = judgment.get('correct', False)
        reason = judgment.get('reason', '')
        elapsed = time.time() - t0
        total_time += elapsed

        status = "CORRECT" if correct else "WRONG"
        print(f"  [{status}] {reason} ({elapsed:.1f}s)")
        print(f"  Entities found: {len(entities)}, Relationships: {len(rels)}, Context length: {len(retrieved_context)}")
        print()

        results.append({
            'question': question,
            'gold': gold,
            'predicted': predicted,
            'correct': correct,
            'reason': reason,
            'type': q_type,
            'num_entities': len(entities),
            'num_relationships': len(rels),
            'context_length': len(retrieved_context),
            'time': elapsed
        })

    # Summary
    n_correct = sum(1 for r in results if r['correct'])
    n_total = len(results)
    accuracy = n_correct / n_total if n_total > 0 else 0

    print(f"\n{'='*70}")
    print(f"QA EVALUATION RESULTS — {DATASET}")
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
            print(f"      Reason: {r['reason']}")
            print()

    # Save results
    with open(f'results/{DATASET}_qa_eval.json', 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'n_correct': n_correct,
            'n_total': n_total,
            'avg_time': total_time / n_total,
            'results': results
        }, f, indent=2)
    print(f"  Results saved to results/{DATASET}_qa_eval.json")


asyncio.run(main())
