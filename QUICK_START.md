# DIGIMON Quick Start Guide

## Installation Options

### Option 1: Minimal Installation (Recommended)

For basic GraphRAG functionality with OpenAI:

```bash
pip install -r requirements-minimal.txt
```

This gives you:
- ✅ Core GraphRAG pipeline
- ✅ OpenAI LLM/embeddings
- ✅ FAISS vector search
- ✅ REST API
- ✅ Basic graph operations

### Option 2: Full Installation

For all features (may have dependency conflicts):

```bash
pip install -r requirements.txt
```

### Option 3: Docker (Easiest)

```bash
# Minimal version
docker-compose up api-minimal

# Full version
docker-compose up digimon-full

# Development environment
docker-compose up dev
```

## Configuration

1. Copy the example config:
```bash
cp Option/Config2.example.yaml Option/Config2.yaml
```

2. Edit `Option/Config2.yaml` and add your API keys:
```yaml
llm:
  api_key: "your-openai-api-key"
  model: "gpt-3.5-turbo"

embedding:
  api_key: "your-openai-api-key"  # Can be same as LLM key
  model: "text-embedding-3-small"

# If you have ColBERT issues:
disable_colbert: true
```

## Verify Installation

```bash
python test_minimal_setup.py
```

## Basic Usage

### 1. Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:5000`

### 2. Use the CLI

```bash
# Interactive mode
python digimon_cli.py -i

# Process a corpus
python digimon_cli.py -c /path/to/documents

# Ask a question
python digimon_cli.py -q "What is machine learning?"
```

### 3. Build a Knowledge Graph

```python
# Using the Python API
from Core.GraphRAG import GraphRAG

# Initialize
graphrag = GraphRAG()

# Build from documents
graphrag.build(dataset_name="my_docs", corpus_path="./Data/my_docs")

# Query
result = graphrag.query("Tell me about the main topics")
```

## Common Issues

### 1. ColBERT Dependency Conflict

If you see transformer/tokenizer errors:

```yaml
# In Option/Config2.yaml
disable_colbert: true
```

Or use minimal requirements which exclude ColBERT.

### 2. Missing API Key

```
ValueError: Please set your API key in Option/Config2.yaml
```

Make sure to add your OpenAI API key to the config file.

### 3. CUDA/GPU Errors

Use the CPU-only version:
```bash
pip install -r requirements-minimal.txt
```

## Next Steps

- Check out the [examples](./examples/) directory
- Read the [API documentation](./docs/API_REFERENCE.md)
- See [DEPENDENCIES.md](./DEPENDENCIES.md) for feature-specific requirements