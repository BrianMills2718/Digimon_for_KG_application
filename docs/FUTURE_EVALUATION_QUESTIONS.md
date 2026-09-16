# Future Evaluation Questions

## Purpose

These questions are intentionally **deferred** while the current priority is finishing and clarifying the DIGIMON architecture. They should guide later validation, benchmarking, ablation work, and research positioning without distorting the near-term implementation around a benchmark.

The goal of this document is to preserve the important questions now so the architecture can be evaluated rigorously later.

## Core System Questions

1. **When does an explicit knowledge graph materially help over lexical/vector retrieval?**
   - Which question classes benefit from graph structure?
   - Which classes are better handled by simpler retrieval?
   - When should the harness avoid graph reasoning entirely?

2. **What is the value of adaptive composition?**
   - Does selecting among local, global, path-based, community, vector, table, and cross-modal workflows improve outcomes compared with always using one fixed method?
   - Does an intelligent harness discover useful compositions that static pipelines miss?

3. **How reliable is graph construction?**
   - How often are important entities or relationships omitted?
   - How often are entities incorrectly merged or duplicated?
   - How do extraction errors propagate into downstream reasoning?

4. **How well is provenance preserved?**
   - Can every important final claim be traced to source chunks?
   - Can graph edges be traced to the text that justified them?
   - Does synthesis preserve those evidence boundaries?

5. **How should conflicting or temporal evidence be represented?**
   - What happens when two documents disagree?
   - How should relationships with different valid time ranges coexist?
   - Can the harness surface disagreement without collapsing it into one answer?

6. **How does the system control graph expansion?**
   - When does K-hop/path exploration become noisy or expensive?
   - Which pruning or ranking signals best preserve useful paths?
   - How should the harness decide when to stop exploring?

7. **How useful is automatic retrieval-method selection?**
   - Does the selected method match the best available method for the question?
   - How often does the harness recover when its first method choice is poor?
   - Are self-reported confidence values meaningful, or should routing confidence be calibrated separately?

8. **What are the quality/cost/latency tradeoffs?**
   - How much answer quality is gained per additional graph operation, LLM call, or token?
   - Which workflows are appropriate for interactive versus batch use?

9. **What happens when the graph is incomplete but the source text contains the answer?**
   - Can the harness fall back from graph reasoning to text/vector retrieval?
   - Can it recognize that a missing edge may be an extraction failure rather than evidence of absence?

10. **How do incremental updates behave?**
    - When new documents arrive, how are entity identity, relationships, communities, and indexes updated?
    - Which artifacts can be updated incrementally versus rebuilt?

## Question Classes for a Later Evaluation Suite

A useful evaluation set should include more than conventional QA accuracy.

### 1. Single-hop controls
Questions that ordinary text/vector retrieval should answer easily. These test whether the harness avoids unnecessary graph work.

### 2. Bridge / multi-hop questions
Questions where an intermediate entity or fact must be discovered before the final fact can be retrieved.

Example dependency shape:

```text
q1: identify an intermediate entity
q2: retrieve a fact about {{q1.entity}}
q3: verify the final answer against source evidence
```

### 3. Comparison questions
Questions that require retrieving facts about two or more entities and comparing them.

### 4. Explicit relational/path questions
Questions such as "How is X connected to Y?" where the path itself is part of the answer.

### 5. Ambiguous entity-resolution questions
Questions involving aliases, same-name entities, abbreviations, or overlapping descriptions.

### 6. Negative / insufficient-evidence questions
Questions where the correct behavior is to report that the corpus does not establish the requested claim.

### 7. Conflicting / temporal evidence questions
Questions where documents disagree or where claims are only valid during particular time periods.

### 8. Structural / global questions
Questions about communities, bridges, central actors, clusters, or other graph-level structure that is not naturally represented by a single retrieved chunk.

## Candidate Baselines for Later

When benchmarking becomes a priority, useful comparisons include:

- lexical/BM25 retrieval;
- vector retrieval;
- lexical + vector hybrid retrieval;
- fixed KG retrieval;
- KG + text/vector hybrid retrieval;
- fixed retrieval method versus harness-selected composition.

The existing evaluation framework already records exact match, token-level F1/precision/recall, latency, LLM calls, and token usage. Later evaluation can build on that rather than redesigning the architecture around a new benchmark harness.

## Important Research/Review Questions

These are also useful interview or design-review prompts:

- Why construct a graph instead of only using BM25 + embeddings + an LLM?
- What part of DIGIMON is infrastructure, and what part is adaptive reasoning policy?
- How are entity-resolution and extraction errors detected?
- Can a final claim be traced through graph evidence back to original source text?
- What happens when two sources disagree?
- How does the system avoid path explosion?
- Why should a harness select among retrieval strategies rather than use one universal pipeline?
- What happens when useful information exists in text but was not represented in the graph?
- How are newly ingested documents propagated through graph and index artifacts?

## Current Decision

Do **not** optimize the architecture around these benchmarks yet. Finish the capability layer, typed contracts, resource model, provenance boundaries, and harness interaction model first. Then use these questions to test the finished architecture rather than allowing a benchmark to become the architecture.
