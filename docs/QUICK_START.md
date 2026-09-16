# DIGIMON Quick Start

**Snapshot:** 2026-09-16

This guide is intentionally narrow: configure the public snapshot and choose an entry point without relying on older experimental setup instructions.

For architecture/status before running the project, see:

- `CURRENT_STATE.md`
- `ARCHITECTURE.md`
- `GAP_ANALYSIS.md`
- `ROADMAP.md`

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

Then edit `Option/Config2.yaml` with the LLM and embedding providers you want to use. The included example contains OpenAI-style `llm` and `embedding` fields plus data/results locations.

Do not commit real API keys.

## 3. Preferred architecture: MCP / external intelligent harness

The preferred architectural surface is:

```text
digimon_mcp_stdio_server.py
```

It exposes DIGIMON through `FastMCP` over stdio. Configure your MCP-capable harness/client to launch that server using the client-specific stdio-server configuration mechanism.

The harness can then work at three levels:

1. call individual DIGIMON capabilities/operators and compose them itself;
2. execute a named reference method when a known composition is convenient;
3. optionally use DIGIMON's auto-selection helper to choose a reference method.

For a capable harness, **individual capability composition is the conceptual default**. Reference/auto modes are conveniences, not mandatory orchestration.

The same MCP surface also provides corpus/graph construction, resource/config inspection, graph/community helpers, analysis, and cross-modal tools.

See `../FUNCTIONALITY.md` for the current capability inventory.

## 4. Choose a corpus directory

For CLI/API experimentation, use an existing directory under `Data/` or a directory containing the documents you want DIGIMON to work with.

Example:

```text
Data/MySampleTexts/
```

## 5. Transitional CLI

`digimon_cli.py` is implemented, but it still instantiates the older internal `PlanningAgent` / `AgentOrchestrator`. Treat it as a **transitional/compatibility entry point**, not the target orchestration boundary.

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

See `API_REFERENCE.md` for API-oriented documentation in this snapshot. The API/UI surfaces are not currently the architectural center of the reconciliation.

## Configuration notes

`Option/Config2.example.yaml` includes:

- LLM provider/model configuration;
- embedding provider/model configuration;
- `data_root`;
- `working_dir`;
- optional `disable_colbert` behavior.

Provider support is mediated through the repository's provider layer; model availability and provider-specific credentials depend on the runtime environment.

## Troubleshooting

### Missing API key or provider credentials

Check `Option/Config2.yaml` and the provider configuration used by the selected LLM/embedding backend.

### Optional dependency conflicts

Start with `requirements-minimal.txt` and enable additional research components only when needed.

### CLI corpus-path errors

The current CLI declares `--corpus/-c` as required. Verify that the directory exists before launching it.

### Missing MCP prerequisite/resource

The MCP layer exposes resource inspection and several prerequisite-building helpers. The current architecture still classifies resource lifecycle/prerequisite handling as **Partial** because those semantics are not yet unified under one typed resource catalog. See `GAP_ANALYSIS.md` rather than assuming every missing resource is auto-built uniformly.

## Next reading

1. `../README.md` — concise project/architecture overview.
2. `CURRENT_STATE.md` — code-truth status map.
3. `ARCHITECTURE.md` — target design.
4. `GAP_ANALYSIS.md` — what remains incomplete.
5. `ROADMAP.md` — architecture-completion sequence.
6. `AGENT_INTELLIGENCE_ENHANCEMENTS.md` — harness-first reasoning and AoT/GoT heuristic policy.
7. `FUTURE_EVALUATION_QUESTIONS.md` — deliberately deferred benchmarking/research questions.

`../MCP_IMPLEMENTATION_TRACKER.md`, `../MCP_INTEGRATION_DETAILED_PLAN.md`, and `../MCP_QUICK_REFERENCE.md` are retained only as historical pointers; their original checkpoint plans are available in Git history.