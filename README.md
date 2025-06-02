Okay, this is a great README that already captures a lot of the spirit and direction of your DIGIMON fork! To make it even more comprehensive and reflective of our recent progress and your larger vision (especially the agent autonomously handling raw text to insights), I'll integrate our recent accomplishments and future plans more explicitly.

Here's a revised, complete README.md. I've focused on:

Clearly stating the end-to-end agent vision.
Updating "Current Features & Capabilities" to reflect the new agent tools.
Refining "Future Plans" to be more specific about the agent's intelligence and the ReACT paradigm.
Ensuring a logical flow and incorporating the strong sections you already have.
Markdown

# 👾 DIGIMON: Deep Analysis of Graph-Based Retrieval-Augmented Generation (RAG) Systems

<div style="text-align: center;">
  <a href="https://github.com/JayLZhou/GraphRAG"><img src="https://img.shields.io/badge/Original_Graph_RAG-red"/></a>
  <a href="https://github.com/JayLZhou/GraphRAG"><img src="https://img.shields.io/github/stars/JayLZhou/GraphRAG"/></a>
  <a href="https://github.com/JayLZhou/GraphRAG"><img src="https://img.shields.io/github/forks/JayLZhou/GraphRAG"/></a>
</div>

> **GraphRAG** is a popular 🔥🔥🔥 and powerful 💪💪💪 RAG system! 🚀💡 Inspired by systems like Microsoft's, graph-based RAG is unlocking endless possibilities in AI.

> The DIGIMON project focuses on **modularizing and decoupling** these methods 🧩 to **unveil the mystery** 🕵️‍♂️🔍✨ behind them, enable advanced analysis, and develop intelligent agent-driven RAG capable of handling complex, multi-step information processing tasks from raw data to summarized insights. Our project🔨 is included in [Awesome Graph-based RAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG).

![Workflow of GraphRAG](./Doc/workflow.png)

---
[![](https://img.shields.io/badge/cs.Zhou-2025.04338-B31C1C?logo=arxiv&logoColor=B31C1C)](https://www.arxiv.org/abs/2503.04338)
[![](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/agentscope/)
[![](https://img.shields.io/badge/Contribute-Welcome-green)](https://modelscope.github.io/agentscope/tutorial/contribute.html) 
- If you find the original work or this fork helpful, please kindly cite [the original GraphRAG paper](https://www.arxiv.org/abs/2503.04338).
- Download the datasets referenced by the original paper: [GraphRAG-dataset](https://drive.google.com/file/d/14nYYw-3FutumQnSRwKavIbG3LRSmIzDX/view?usp=sharing)

---

## About this Fork: DIGIMON

This fork, **DIGIMON (Deep Analysis of Graph-Based Retrieval-Augmented Generation Systems)**, significantly refactors and extends the original [JayLZhou/GraphRAG](https://github.com/JayLZhou/GraphRAG) project. The primary motivation is to transform the initial monolithic testing pipeline into a highly modular and flexible "usage suite." This allows for deeper analysis, more adaptable experimentation, and provides a robust foundation for advanced research into graph-based RAG, particularly through the development of an intelligent agent.

**Key Goals for DIGIMON:**

* Provide a robust and modular framework for building, querying, and evaluating diverse graph-based RAG systems.
* Enable detailed analysis of individual components and operator chains within graph RAG pipelines.
* Develop an intelligent LLM-based agent capable of:
    * **End-to-End Data Processing:** Handling tasks from raw text files in a directory (e.g., preparing a corpus, chunking) through to KG construction and insight extraction.
    * **Autonomous KG Structuring:** Intelligently selecting appropriate graph types (e.g., ERGraph, TreeGraph, PassageGraph), applying ontologies, and configuring build parameters based on the data and task.
    * **Dynamic Retrieval Strategies:** Determining and executing optimal retrieval strategies by composing sequences of the system's granular operators in a ReACT-style (Reason, Act, Observe) iterative workflow.
* Serve as a flexible platform for PhD research (e.g., on social media discourse analysis) and broader exploration of advanced RAG techniques and agentic information retrieval.

---

## Current Features & Capabilities

### Modular Architecture & Operational Modes

The system features a modular design with distinct operational modes, manageable via `main.py`, and increasingly, through agent-callable tools:

1.  **Build Mode (via `main.py` or Agent Tools):** Constructs knowledge graphs and generates all necessary artifacts (e.g., graph structure, vector databases).
    * Agent Tools available for building all 5 core graph types: `ERGraph`, `RKGraph`, `TreeGraph`, `TreeGraphBalanced`, `PassageGraph`.
    * Agent Tool available for `PrepareCorpusFromDirectory` (processes `.txt` files into `Corpus.json`).
    ```bash
    # Example CLI usage
    python main.py build -opt Option/Method/RAPTOR.yaml -dataset_name your_dataset
    ```
2.  **Query Mode (via `main.py` or Agent Tools):** Loads pre-built artifacts to answer questions.
    ```bash
    # Example CLI usage
    python main.py query -opt Option/Method/RAPTOR.yaml -dataset_name your_dataset -question "Your question here?"
    ```
3.  **Evaluate Mode (via `main.py`):** Assesses performance against benchmark datasets.
    ```bash
    # Example CLI usage
    python main.py evaluate -opt Option/Method/RAPTOR.yaml -dataset_name your_dataset
    ```

### Web Interface (API & UI)

A Flask API server (`api.py`) and an initial React UI provide user-friendly interaction, though current development is heavily focused on backend agent capabilities.
* **API Endpoints:** `/api/query`, `/api/build`, `/api/evaluate`.
* **UI:** Allows selection of datasets, methods, and initiation of operations.

### Available RAG Methods & Graph Types

* **Pre-defined Configurations:** `Dalk`, `GR`, `LGraphRAG`, `GGraphRAG`, `HippoRAG`, `KGP`, `LightRAG`, `RAPTOR`, `ToG`. These methods are compositions of underlying granular operators.
* **Supported Graph Types (for agent construction and analysis):**
    * **ChunkTree:** Hierarchical summary trees (`TreeGraph`, `TreeGraphBalanced`).
    * **PassageGraph:** Nodes are passages, edges link passages with shared (WAT-linked) entities.
    * **KG/TKG/RKG:** Graphs with explicit entities and relationships (`ERGraph`, `RKGraph`).

### Intelligent Agent Framework (Core Development Focus)

The central aim is an intelligent agent that dynamically plans and executes RAG strategies:
* **Granular Operator Tools:** The agent leverages ~16 conceptual retrieval and graph manipulation operators as its building blocks.
* **Structured Agent Tools:**
    * **Graph Construction Tools:** Defined with Pydantic contracts (`Core/AgentSchema/graph_construction_tool_contracts.py`) and implemented (`Core/AgentTools/graph_construction_tools.py`) for all five graph types. The `build_er_graph` tool has been successfully tested with live LLM calls.
    * **Corpus Preparation Tool:** `PrepareCorpusFromDirectoryTool` implemented and tested, allowing the agent to process raw `.txt` files.
* **Pydantic-based Execution Plans:** The agent's reasoning (planned or ReACT-driven) aims to produce structured sequences of tool calls.
* **Agent Orchestrator:** (`Core/AgentOrchestrator/orchestrator.py`) Executes tool calls based on the agent's decisions.
* **Agent Brain:** (`Core/AgentBrain/agent_brain.py`) Houses the planning logic (currently using an LLM to generate plans). Future work will enhance this for ReACT-style reasoning.

---

## Quick Start 🚀

### From Source
```bash
Clone this repository

cd Digimon_KG
Install Dependencies
The primary Conda environment is defined in experiment.yml.

Bash

conda env create -f experiment.yml -n digimon
conda activate digimon
(Note: environment.yml may also exist; experiment.yml is often referenced for the core setup).

API Keys and Configuration
Copy Option/Config2.example.yaml to Option/Config2.yaml.
Edit Option/Config2.yaml to include your API keys (e.g., OpenAI api_key for llm and embedding sections) and set desired default models (e.g., llm.model: "openai/o4-mini-2025-04-16").
Method-specific configurations (used by main.py) are in Option/Method/.
Custom ontology can be defined in Config/custom_ontology.json and referenced in GraphConfig or overridden by agent tools.
Supported LLM Backends
DIGIMON uses LiteLLMProvider for broad LLM compatibility, configured via Option/Config2.yaml:

Cloud-based models: OpenAI (e.g., "openai/gpt-4o", "openai/o4-mini-2025-04-16"), Anthropic, Gemini, etc.
Locally deployed models: Any LiteLLM-supported endpoint (Ollama, LlamaFactory-compatible).
Set llm.model to the appropriate LiteLLM string (e.g., "ollama/llama3").
Set llm.base_url if needed (e.g., "http://localhost:11434" for Ollama, though often LiteLLM handles this).
llm.api_key can often be set to "None" or a placeholder for local models.
Representative Graph RAG Methods & Operators
(This section can largely retain the excellent tables from your current README, as they provide valuable context on the original GraphRAG methods and the derived operators. I'm keeping it concise here for the handoff structure but you should integrate your full tables back.)

Graph Types Overview
(Integrate your existing "Graph Types" table here, comparing Chunk Tree, Passage Graph, KG, TKG, RKG across attributes like Original Content, Entity Name, etc.)

Retrieval Operators (Agent's Building Blocks)
(Integrate your existing tables for Entity Operators, Relationship Operators, Chunk Operators, Subgraph Operators, and Community Operators, including their Name, Description, and Example Methods.)

The DIGIMON agent's intelligence will stem from its ability to dynamically select, configure, and chain these operators to address complex queries.

🎯 Future Plans for DIGIMON (Agent-Centric)
This section outlines the specific future development goals for the DIGIMON agent and framework:

Agent Planning & Execution Enhancement:
[ ] ReACT-style Agent Core: Evolve AgentBrain to support a ReACT (Reason, Act, Observe) paradigm for more robust and adaptive multi-step task execution.
[ ] Advanced Planning Prompts: Refine prompt engineering for the PlanningAgent to effectively utilize all available tools (corpus prep, graph build, retrieval, summarization) and manage data flow between tool calls for complex queries.
Tool Development & Refinement:
[ ] Retrieval Tools: Define Pydantic contracts and implement robust agent tools for the remaining ~13 granular retrieval operators (e.g., for relationship retrieval, community analysis, advanced subgraph extraction).
[ ] Summarization Tool: Implement a flexible SummarizeTextTool for final answer synthesis.
[ ] Integrated Testing: Create integrated tests for all graph construction tools (similar to the one for build_er_graph) using real components with strategic LLM mocking/use.
Agent Intelligence & Strategy:
[ ] KG Structuring Strategy: Develop logic/heuristics/LLM-prompts to enable the agent to intelligently select the optimal graph type(s) and construction parameters based on the input data and user query.
[ ] Dynamic Retrieval Strategy: Enable the agent to dynamically choose and sequence retrieval operators to best answer a query given a constructed graph.
[ ] Ontology Management: Enhance agent capabilities for more dynamic interaction with custom_ontology.json, potentially including ontology selection or refinement suggestions.
Framework & Usability:
[ ] Evaluation Framework for Agent Strategies: Develop methods to evaluate the end-to-end effectiveness of agent-composed RAG strategies.
[ ] Comprehensive Documentation: Document the agent framework, all tool contracts, and provide example workflows.
[ ] Dockerization: Create a Docker image for easier deployment and reproducibility.
🧭 Cite The Original GraphRAG Paper
If you find this work useful, please consider citing the original paper that inspired this project:

In-depth Analysis of Graph-based RAG in a Unified Framework
@article{zhou2025depth,
  title={In-depth Analysis of Graph-based RAG in a Unified Framework},
  author={Zhou, Yingli and Su, Yaodong and Sun, Youran and Wang, Shu and Wang, Taotao and He, Runyuan and Zhang, Yongwei and Liang, Sicong and Liu, Xilin and Ma, Yuchi and others},
  journal={arXiv preprint arXiv:2503.04338},
  year={2025}
}

**Key Changes Made:**

* **Elevated Agent Vision:** More clearly stated the goal of an agent handling tasks end-to-end, from raw text to insights, using ReACT principles.
* **Updated Current Features:** Specifically mentioned the new `PrepareCorpusFromDirectoryTool` and the status of graph construction tools and their testing.
* **Refined Future Plans:** Made the future plans more agent-centric, focusing on planning, ReACT, KG structuring strategy, dynamic retrieval, and the remaining tool implementations.
* **LLM Configuration Clarity:** Updated the "Supported LLM Backends" to reflect the use of `LiteLLMProvider` and how to configure for cloud vs. local models using LiteLLM conventions.
* **Structure and Flow:** Ensured a logical flow from project goals to current capabilities, quick start, and future plans.

ou'll also want to integrate your detailed tables for "Graph Types Overview" and "Retrieval Operators" back into the "Representative Graph RAG Methods & Operators" section.