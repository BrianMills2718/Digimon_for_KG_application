# DIGIMON Dependencies Guide

This guide explains the different dependency configurations for DIGIMON and which features require which dependencies.

## Quick Start

For basic functionality:
```bash
pip install -r requirements-minimal.txt
```

For all features:
```bash
pip install -r requirements.txt
```

## Dependency Tiers

### 1. Minimal Dependencies (`requirements-minimal.txt`)

These are the core dependencies required for basic DIGIMON functionality:

- **Core Framework**: pydantic, pyyaml, loguru
- **Data Processing**: numpy, pandas, networkx, scipy
- **LLM Integration**: litellm, openai, tiktoken
- **Vector Database**: faiss-cpu, llama-index-core
- **Web API**: flask

**Features Available**:
- ✅ Basic GraphRAG pipeline
- ✅ Entity and relationship extraction
- ✅ FAISS vector indexing
- ✅ OpenAI LLM/embedding support
- ✅ REST API
- ✅ Basic graph algorithms

**Features NOT Available**:
- ❌ ColBERT retrieval
- ❌ Alternative LLM providers (Anthropic, Ollama)
- ❌ Advanced graph visualizations
- ❌ PostgreSQL vector database
- ❌ Streamlit UI

### 2. Optional Dependencies (`requirements-optional.txt`)

Install these based on your specific needs:

#### Graph Visualization
```bash
pip install matplotlib seaborn graspologic igraph
```
- Required for: Graph visualization, advanced graph metrics, community detection

#### Alternative LLM Providers
```bash
pip install anthropic ollama instructor
```
- Required for: Using Claude, local models, or structured outputs

#### Advanced NLP
```bash
pip install nltk gensim sentence-transformers transformers
```
- Required for: Topic modeling, advanced embeddings, Hugging Face models

#### PostgreSQL Support
```bash
pip install pgvector psycopg2-binary sqlalchemy llama-index-vector-stores-postgres
```
- Required for: Using PostgreSQL as vector database instead of FAISS

#### Streamlit UI
```bash
pip install streamlit streamlit-agraph
```
- Required for: Web-based interactive UI

### 3. Full Dependencies (`requirements.txt`)

The complete requirements file includes all dependencies. Use this if you want all features available.

## Known Conflicts

### ColBERT Dependencies

ColBERT requires older versions of transformers that conflict with other components:

```bash
# ColBERT requires:
colbert-ai==0.2.21
transformers==4.21.0
tokenizers==0.12.1

# But other components need:
transformers>=4.49.0
```

**Solution**: ColBERT is now optional. Set `disable_colbert: true` in your Config2.yaml or use `vdb_type: faiss` instead of `colbert`.

### GPU Dependencies

For GPU support, you need to install PyTorch with CUDA:

```bash
# CPU only (default)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Docker Configurations

We provide several Docker configurations for different use cases:

### Basic CPU Configuration
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt
COPY . .
CMD ["python", "api.py"]
```

### Full Features with GPU
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# ... (see Dockerfile.gpu)
```

### Development Environment
```dockerfile
FROM python:3.10
# ... (see Dockerfile.dev)
```

## Feature-Dependency Matrix

| Feature | Required Dependencies | Config Changes |
|---------|----------------------|----------------|
| Basic GraphRAG | requirements-minimal.txt | None |
| ColBERT Retrieval | colbert-ai, transformers==4.21.0 | `vdb_type: colbert` |
| Claude LLM | anthropic | `llm.api_type: anthropic` |
| Local LLMs | ollama | `llm.api_type: ollama` |
| PostgreSQL VDB | pgvector, psycopg2 | `vdb_type: postgres` |
| Graph Visualization | matplotlib, graspologic | None |
| Streamlit UI | streamlit | Run `streamlit_agent_frontend.py` |
| GPU Acceleration | torch-cuda, faiss-gpu | `device: cuda` |

## Installation Troubleshooting

### 1. faiss-cpu vs faiss-gpu
- Use `faiss-cpu` for most cases (included in minimal)
- Only install `faiss-gpu` if you have CUDA and need GPU acceleration

### 2. Conflicting transformers versions
- If you see transformer/tokenizer errors, you likely have the ColBERT conflict
- Solution: Use FAISS instead of ColBERT

### 3. Missing system dependencies
Some packages require system libraries:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

# macOS
brew install python@3.10

# Windows
# Use WSL2 or install Visual Studio Build Tools
```

### 4. Memory issues during installation
For systems with limited memory:

```bash
pip install --no-cache-dir -r requirements-minimal.txt
```

## Recommended Configurations

### For Development
```bash
pip install -r requirements-minimal.txt
pip install pytest pytest-asyncio ipython black ruff
```

### For Production API Server
```bash
pip install -r requirements-minimal.txt
pip install gunicorn prometheus-client
```

### For Research/Experimentation
```bash
pip install -r requirements.txt  # Full dependencies
pip install jupyter notebook
```

## Environment Variables

Some dependencies require API keys:

```bash
# Required for OpenAI
export OPENAI_API_KEY="your-key"

# Optional for other providers
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"

# For PostgreSQL
export DATABASE_URL="postgresql://user:pass@localhost/dbname"
```

## Upgrading Dependencies

To upgrade dependencies safely:

1. **Test in isolation first**:
   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install -r requirements-minimal.txt
   python -m pytest tests/
   ```

2. **Check for breaking changes**:
   - LlamaIndex: Check [migration guides](https://docs.llamaindex.ai/en/stable/migrating/)
   - Pydantic: v1 to v2 has breaking changes
   - NetworkX: v2 to v3 changed some APIs

3. **Update gradually**:
   ```bash
   pip install --upgrade package_name
   python -m pytest tests/
   ```

## Contributing

When adding new dependencies:

1. Add to appropriate requirements file
2. Document in this file under the correct section
3. Update feature-dependency matrix
4. Test with minimal dependencies to ensure it's optional
5. Add to Docker configurations if needed