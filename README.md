
## 👾 DIGIMON: Deep Analysis of Graph-Based Retrieval-Augmented Generation (RAG) Systems


<div style="text-align: center;">
  <a href="https://github.com/JayLZhou/GraphRAG"><img src="https://img.shields.io/badge/DIGIMON-red"/></a>
  <a href="https://github.com/JayLZhou/GraphRAG"><img src="https://img.shields.io/badge/Graph_RAG-red"/></a>
  <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/JayLZhou/GraphRAG"/></a>
  <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/forks/JayLZhou/GraphRAG"/></a>
  <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/last-commit/JayLZhou/GraphRAG?color=blue"/></a>
</div>



<!-- ![Static Badge](https://img.shields.io/badge/DIGIMON-red)
![Static Badge](https://img.shields.io/badge/LLM-red)
![Static Badge](https://img.shields.io/badge/Graph_RAG-red)
![Static Badge](https://img.shields.io/badge/Document_QA-green)
![Static Badge](https://img.shields.io/badge/Document_Summarization-green) -->


<!-- <img src="img.png" alt="Description of the image" width="450" height="350"> -->

> **GraphRAG** is a popular 🔥🔥🔥 and powerful 💪💪💪 RAG system! 🚀💡 Inspired by systems like Microsoft's, graph-based RAG is unlocking endless possibilities in AI.

> Our project focuses on **modularizing and decoupling** these methods 🧩 to **unveil the mystery** 🕵️‍♂️🔍✨ behind them and share fun and valuable insights! 🤩💫  Our project🔨 is included in [Awesome Graph-based RAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG).

![Workflow of GraphRAG](Doc\workflow.png)

---

## Representative Methods

We select the following Graph RAG methods:

| Method | Description| Link | Graph Type|
| --- |--- |--- | :---: | 
| RAPTOR | ICLR 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2401.18059-b31b1b.svg)](https://arxiv.org/abs/2401.18059)  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/parthsarthi03/raptor)| Tree |
| KGP | AAAI 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2308.11730-b31b1b.svg)](https://arxiv.org/abs/2308.11730)  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YuWVandy/KG-LLM-MDQA)| Passage Graph |
| DALK | EMNLP 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.04819-b31b1b.svg)](https://arxiv.org/abs/2405.04819) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/David-Li0406/DALK)| KG |
| HippoRAG | NIPS 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.14831-b31b1b.svg)](https://arxiv.org/abs/2405.14831) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/OSU-NLP-Group/HippoRAG) | KG |
| G-retriever | NIPS 2024  | [![arXiv](https://img.shields.io/badge/arXiv-2402.07630-b31b1b.svg)](https://arxiv.org/abs/2402.07630) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/XiaoxinHe/G-Retriever)| KG |
| ToG | ICLR 2024  | [![arXiv](https://img.shields.io/badge/arXiv-2307.07697-b31b1b.svg)](https://arxiv.org/abs/2307.07697) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/IDEA-FinAI/ToG)| KG |
| MS GraphRAG | Microsoft Project |  [![arXiv](https://img.shields.io/badge/arXiv-2404.16130-b31b1b.svg)](https://arxiv.org/abs/2404.16130) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/microsoft/graphrag)| TKG |
| FastGraphRAG | CircleMind Project  | [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/circlemind-ai/fast-graphrag)| TKG |
| LightRAG | High Star Project  | [![arXiv](https://img.shields.io/badge/arXiv-2410.05779-b31b1b.svg)](https://arxiv.org/abs/2410.05779) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/HKUDS/LightRAG)| RKG |


##  Graph Types
Based on the entity and relation, we categorize the graph into the following types:

+ **Chunk Tree**: A tree structure formed by document content and summary.
+ **Passage Graph**: A relational network composed of passages, tables, and other elements within documents.
+ **KG**: knowledge graph (KG) is constructed by extracting entities and relationships from each chunk, which contains only entities and relations, is commonly represented as triples.
+ **TKG**: A textual knowledge graph (TKG) is a specialized KG (following the same construction step as KG), which enriches entities with detailed descriptions and type information.
+ **RKG**: A rich knowledge graph (RKG), which further incorporates keywords associated with relations.

The criteria for the classification of graph types are as follows:

|Graph Attributes | Chunk Tree |Passage Graph | KG  | TKG | RKG |
| --- |--- |--- |--- | --- | --- |
|Original Content| ✅|✅| ❌|❌|❌| 
|Entity Name| ❌|❌|✅|✅|✅|
|Entity Type| ❌| ❌| ❌|✅|✅|
|Entity Description|❌| ❌| ❌|✅|✅|
|Relation Name| ❌|❌|✅|❌|✅|
|Relation keyword|❌| ❌| ❌|❌|✅|
|Relation Description|❌| ❌| ❌|✅|✅|
|Edge Weight| ❌|❌|✅|✅|✅|

##  Operators in the Retrieve Stage 
> The retrieval stage lies the **key role** ‼️ in the entire GraphRAG process. ✨ The goal is to identify query-relevant content that supports the generation phase, enabling the LLM to provide more **accurate** responses.


💡💡💡 After thoroughly reviewing all implementations, we've distilled them into a set of **16** operators 🧩🧩. Each method then constructs its retrieval module by combining one or more of these operators 🧩.

### Five Types of Operators

> We classify the operators into five categories, each offering a different way to retrieve and structure relevant information from graph-based data.


#### ⭕️ Entity Operators
> Retrieve entities (e.g., people, places, organizations) that are most relevant to the given query.

| Name | Description | Example Methods  |
|---|---|---|
| **VDB**  | Select top-k nodes from the vector database  | G-retriever, RAPTOR, KGP |
| **RelNode** | Extract nodes from given relationships | LightRAG  |
| **PPR** | Run PPR on the graph, return top-k nodes with PPR scores | FastGraphRAG  | 
| **Agent** | Utilizes LLM to find the useful entities| ToG |
| **Onehop** | Selects the one-hop neighbor entities of the given entities| LightRAG |
| **Link** | Return top-1 similar entity for each given entity| HippoRAG |
| **TF-IDF** | Rank entities based on the TF-IFG matrix| KGP |



#### ➡️ Relationship Operators
> Extracting useful relationships for the given query.

| Name | Description | Example Methods |
|---|---|---|
| **VDB** | Retrieve relationships by vector-database| LightRAG、G-retriever |
| **Onehop** | Selects relationships linked by one-hop neighbors of the given selected entities | Local Search for MS GraphRAG |
| **Aggregator** | Compute relationship scores from entity PPR matrix, return top-k | FastGraphRAG |
| **Agent**| Utilizes LLM to find the useful entities| ToG|


#### 📄 Chunk Operators
> Retrieve the most relevant text segments (chunks) related to the query.

                                      
| Name               | Description                                                    | Example Methods              |
|---|---|---|
| **Aggregator** | Use the relationship scores and the relationship-chunk interactions to select the top-k chunks | HippoRAG |
| **FromRel** | Return chunks containing given relationships | LightRAG |
| **Occurrence** | Rank top-k chunks based on occurrence of both entities in relationships | Local Search for MS GraphRAG |



#### 📈 Subgraph Operators
> Extract a relevant subgraph for the given query

| Name  | Description | Example Methods |
|---|---|---|
| **KhopPath** | Find k-hop paths with start and endpoints in the given entity set | DALK |
| **Steiner** | Compute Steiner tree based on given entities and relationships  | G-retriever     |
| **AgentPath** |  Identify the most relevant 𝑘-hop paths to a given question, by using LLM to filter out the irrelevant paths | TOG   |




#### 🔗 Community Operators
> Identify high-level information, which is only used for MS GraphRAG.


| **Name** | **Description**   | **Example Methods**  |
|---|---|---|
| **Entity**  | Detects communities containing specified entities   | Local Search for MS GraphRAG                   |
| **Layer**  | Returns all communities below a required layer | Global Search for MS GraphRAG                  | 


You can freely 🪽 combine those operators 🧩 to create more and more GraphRAG methods. 


#### 🌰 Examples 
> Below, we present some examples illustrating how existing algorithms leverage these operators.


| Name | Operators|
|---|---|
| **HippoRAG**     | Chunk (Aggregator) |
| **LightRAG**     | Chunk (FromRel) + Entity (RelNode) + Relationship (VDB)                          |
| **FastGraphRAG** | Chunk (Aggregator) + Entity (PPR) + Relationship (Aggregator)  |


## 📊 Dataset \& Data Format

Please view the following link: [GraphRAG-dataset](https://drive.google.com/file/d/14nYYw-3FutumQnSRwKavIbG3LRSmIzDX/view?usp=sharing)


## 📖 See Our Paper

For a detailed analysis, check out our paper:

📄 **[In-depth Analysis of Graph-based RAG in a Unified Framework](https://www.arxiv.org/abs/2503.04338) (arXiv preprint 2503.04338)**