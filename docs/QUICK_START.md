# DIGIMON Quick Start

**Snapshot:** 2026-09-17

This guide is intentionally narrow: configure the repository and choose a current entry point without relying on older experimental setup instructions.

For context before running the project, see:

- `VISION.md` — full project north star;
- `CURRENT_STATE.md` — what is materially implemented now;
- `IMPLEMENTATION_MAP.md` — module classification and code caveats;
- `ARCHITECTURE.md` — target technical design;
- `GAP_ANALYSIS.md` — current→target gaps;
- `ROADMAP.md` — ordered work.

## 1. Install dependencies

For the smallest dependency set:

```bash
pip install -r requirements-minimal.txt
```

For the broader research environment:

```bash
pip install -r requirements.txt
```

The broader environment includes optional research/retrieval components and may require additional system or GPU-specific setup.

## 2. Create the runtime configuration

Copy the example configuration:

```bash
cp Option/Config2.example.yaml Option/Config2.yaml
```

Then edit `Option/Config2.yaml` with the LLM and embedding providers you want to use. Do not commit real API keys.

## 3. Current agent-facing surface: MCP

The strongest current modern agent-facing protocol surface is:

```text
digimon_mcp_stdio_server.py
```

It exposes DIGIMON through `FastMCP` over stdio. Configure an MCP-capable harness/client to launch that server using the client's stdio-server configuration mechanism.

A capable harness can:

1. call specialized DIGIMON capabilities/operators and compose them itself;
2. execute a named reference method when a known composition is convenient;
3. optionally use DIGIMON's method-selection convenience behavior.

The external harness owns adaptive reasoning and sequencing. MCP is **an interface to DIGIMON, not the project thesis**. The target public surfaces are CLI + Python runtime + MCP over the same maintained representation/retrieval/analytics core.

The MCP surface also exposes corpus/graph construction, resource/config inspection, community/graph helpers, analysis and cross-modal tools.

Current implementation facts worth knowing:

- required typed plan inputs must be explicitly wired by name;
- invalid plans fail closed by default; best-effort execution must be requested explicitly;
- active graph and canonical entity/relation VDB selection are tracked pragmatically;
- rebuilt graphs invalidate known stale VDB/community/matrix artifacts;
- current head still lacks a fresh runtime certification in the available development environment.

See `CURRENT_STATE.md` for exact current behavior.

## 4. Standalone raw corpus mode

For CLI/raw-corpus experimentation, use an existing directory under `Data/` or another document directory.

Example:

```text
Data/MySampleTexts/
```

The maintained standalone path now applies configured chunking. In the broader ecosystem architecture, however, governed semantic IR from onto-canon6 is the intended canonical upstream seam; raw-document mode remains useful for independent operation, benchmarks and compatibility.

## 5. Transitional CLI

`digimon_cli.py` is implemented, but it still instantiates the older internal `PlanningAgent` / `AgentOrchestrator`. Treat it as a **transitional human-facing entry point**, not the definition of the target runtime architecture.

Interactive mode:

```bash
python digimon_cli.py -c Data/MySampleTexts -i
```

Single question:

```bash
python digimon_cli.py -c Data/MySampleTexts -q "What are the main entities and how are they connected?"
```

Experimental ReAct-style mode:

```bash
python digimon_cli.py -c Data/MySampleTexts -q "How are the major entities connected?" --react
```

Batch questions:

```bash
python digimon_cli.py -c Data/MySampleTexts -b queries.txt -o results.json
```

Custom configuration:

```bash
python digimon_cli.py -c Data/MySampleTexts -i --config path/to/config.yaml
```

## 6. Secondary HTTP/API surface

The repository also contains `api.py` as a secondary HTTP/API entry point:

```bash
python api.py
```

See `API_REFERENCE.md` for API-oriented documentation. API/UI surfaces are not currently the architectural center.

## 7. Deterministic verification path

When a real runner is available, the maintained minimal verification sequence is:

```bash
pip install -r requirements-minimal.txt
pytest tests/core -q
python tests/e2e/test_mcp_smoke.py
DIGIMON_CANARY_REBUILD=1 python tests/e2e/test_mcp_smoke.py
```

Do not treat source-reviewed tests or historical Actions runs as proof that current head is green until this is actually executed.

## Configuration notes

`Option/Config2.example.yaml` includes LLM/embedding configuration plus data/results locations and optional behavior flags. Provider availability and credentials depend on the runtime environment.

Custom ontology configuration exists, but the maintained selected-path → loaded ontology → extraction flow is still an explicit current gap; see `CURRENT_STATE.md` / `ROADMAP.md` rather than assuming every override is already proven.

## Troubleshooting

### Missing API key or provider credentials

Check `Option/Config2.yaml` and the provider configuration used by the selected LLM/embedding backend.

### Optional dependency conflicts

Start with `requirements-minimal.txt` and enable broader research components only when needed.

### CLI corpus-path errors

The current CLI declares `--corpus/-c` as required. Verify that the directory exists before launching it.

### Missing/stale graph-derived resources

Current resource handling is pragmatic rather than a generalized resource-governance subsystem: graph source manifests trigger rebuilds when source chunks change, and successful rebuilds invalidate known canonical downstream VDB/community/matrix artifacts. See `CURRENT_STATE.md` for remaining edge cases such as graph-scoped sparse identity.

## Next reading

1. `../README.md` — concise project overview.
2. `VISION.md` — full project thesis.
3. `CURRENT_STATE.md` — code-truth status map.
4. `ARCHITECTURE.md` — target technical design.
5. `IMPLEMENTATION_MAP.md` — module/capability detail.
6. `GAP_ANALYSIS.md` — what remains incomplete.
7. `ROADMAP.md` — ordered implementation sequence.
8. `DOCUMENTATION_COVERAGE.md` — documentation reconciliation checklist.
9. `AGENT_INTELLIGENCE_ENHANCEMENTS.md` — harness-control/AoT-GoT policy detail.
10. `FUTURE_EVALUATION_QUESTIONS.md` — deferred evaluation/research questions.

Historical root MCP planning/tracker files remain lineage pointers; they are not current execution plans.
