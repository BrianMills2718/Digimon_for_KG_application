# DIGIMON Quick Start

This guide is intentionally narrow: get the public snapshot configured and exercise the current CLI/API surfaces without relying on older experimental setup instructions.

## 1. Install dependencies

For the smallest supported dependency set:

```bash
pip install -r requirements-minimal.txt
```

For the broader research environment:

```bash
pip install -r requirements.txt
```

The full dependency set includes more optional research/retrieval components and may require additional system or GPU-specific setup.

## 2. Create the runtime configuration

Copy the example configuration:

```bash
cp Option/Config2.example.yaml Option/Config2.yaml
```

Then edit `Option/Config2.yaml` with the LLM and embedding providers you want to use. The included example contains OpenAI-style fields for `llm` and `embedding` plus the dataset/results locations.

Do not commit real API keys.

## 3. Choose a corpus directory

The CLI requires a corpus path. Use an existing directory under `Data/` or a directory containing the documents you want the harness to work with.

Example:

```text
Data/MySampleTexts/
```

## 4. Run the CLI

Interactive mode:

```bash
python digimon_cli.py -c Data/MySampleTexts -i
```

Single question:

```bash
python digimon_cli.py -c Data/MySampleTexts -q "What are the main entities and how are they connected?"
```

Experimental ReAct-style iterative planning:

```bash
python digimon_cli.py -c Data/MySampleTexts -q "How are the major entities connected?" --react
```

Batch questions:

```bash
python digimon_cli.py -c Data/MySampleTexts -b queries.txt -o results.json
```

A custom configuration can be supplied with:

```bash
python digimon_cli.py -c Data/MySampleTexts -i --config path/to/config.yaml
```

## 5. API surface

The repository also contains `api.py` as an HTTP/API entry point:

```bash
python api.py
```

See `docs/API_REFERENCE.md` for the API-oriented documentation in this snapshot.

## 6. MCP / intelligent-harness surface

`digimon_mcp_stdio_server.py` exposes DIGIMON capabilities for an MCP-capable harness. The intended architecture is harness-first: the harness decides how to compose the available graph, vector, text, community, and structured operations rather than relying on one fixed pipeline.

See:

- `../FUNCTIONALITY.md` for the capability overview;
- `AGENT_INTELLIGENCE_ENHANCEMENTS.md` for the current reasoning architecture;
- `../MCP_QUICK_REFERENCE.md` for MCP-specific material in this snapshot.

## Configuration notes

The example `Option/Config2.example.yaml` includes:

- `llm` provider/model configuration;
- `embedding` provider/model configuration;
- `data_root`;
- `working_dir`;
- an optional `disable_colbert` switch.

Provider support is mediated through the repository's provider layer; model availability and provider-specific credentials depend on the environment in which you run DIGIMON.

## Troubleshooting

### Missing API key or provider credentials

Check `Option/Config2.yaml` and the provider configuration used by your selected LLM/embedding backend.

### Optional retrieval dependency conflicts

Start with `requirements-minimal.txt` and enable additional research components only when needed.

### Corpus path errors

The current CLI declares `--corpus/-c` as required. Make sure the directory exists before launching the CLI.

## Next reading

1. `../README.md` — project and architecture overview.
2. `../FUNCTIONALITY.md` — what the tool layer exposes.
3. `AGENT_INTELLIGENCE_ENHANCEMENTS.md` — harness-first reasoning and AoT/GoT heuristic policy.
4. `FUTURE_EVALUATION_QUESTIONS.md` — deferred benchmarking and research questions.
