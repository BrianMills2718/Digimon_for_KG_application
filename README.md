
## 👾 DIGIMON: Deep Analysis of Graph-Based Retrieval-Augmented Generation (RAG) Systems
<!-- <img src="img.png" alt="Description of the image" width="450" height="350"> -->

![Static Badge](https://img.shields.io/badge/DIGIMON-red)
![Static Badge](https://img.shields.io/badge/LLM-red)
![Static Badge](https://img.shields.io/badge/Graph_RAG-red)
![Static Badge](https://img.shields.io/badge/Document_QA-green)
![Static Badge](https://img.shields.io/badge/Document_Summarization-green)

> **GraphRAG** is a popular 🔥🔥🔥 and powerful 💪💪💪 RAG system! 🚀💡 Inspired by systems like Microsoft's, graph-based RAG is unlocking endless possibilities in AI.

> Our project focuses on **modularizing and decoupling** these methods 🧩 to **unveil the mystery** 🕵️‍♂️🔍✨ behind them and share fun and valuable insights! 🤩💫


## Representative Methods

We select the following Graph RAG methods:

| Method | Description| Link | Graph Type|
| --- |--- |--- | :---: | 
| RAPTOR | ICLR 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2401.18059-b31b1b.svg)](https://arxiv.org/abs/2401.18059)  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/parthsarthi03/raptor)| Tree |
| KGP | AAAI 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2308.11730-b31b1b.svg)](https://arxiv.org/abs/2308.11730)  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YuWVandy/KG-LLM-MDQA)| Passage Graph |
| DALK | EMNLP 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.04819-b31b1b.svg)](https://arxiv.org/abs/2405.04819) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/David-Li0406/DALK)| ER Graph |
| HippoRAG | NIPS 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.14831-b31b1b.svg)](https://arxiv.org/abs/2405.14831) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/OSU-NLP-Group/HippoRAG)| ER Graph |
| G-retriever | NIPS 2024  | [![arXiv](https://img.shields.io/badge/arXiv-2402.07630-b31b1b.svg)](https://arxiv.org/abs/2402.07630) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/XiaoxinHe/G-Retriever)| ER Graph |
| ToG | NIPS 2024  | [![arXiv](https://img.shields.io/badge/arXiv-2307.07697-b31b1b.svg)](https://arxiv.org/abs/2307.07697) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/IDEA-FinAI/ToG)| ER Graph |
| GraphCoT | ACL 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2404.07103-b31b1b.svg)](https://arxiv.org/abs/2404.07103) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PeterGriffinJin/Graph-CoT)| ER Graph |
| MS GraphRAG | Microsoft Project |  [![arXiv](https://img.shields.io/badge/arXiv-2404.16130-b31b1b.svg)](https://arxiv.org/abs/2404.16130) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/microsoft/graphrag)| KG |
| LightRAG | High Star Project  | [![arXiv](https://img.shields.io/badge/arXiv-2410.05779-b31b1b.svg)](https://arxiv.org/abs/2410.05779) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/HKUDS/LightRAG)| RKG |
| FastGraphRAG | CircleMind Project  | [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/circlemind-ai/fast-graphrag)| RKG |

##  Graph Types
Based on the entity and relation, we categorize the graph into the following types:

+ **Chunk Tree**: A tree structure formed by document content and summary.
+ **Passage Graph**: A relational network composed of passages, tables, and other elements within documents.
+ **ER Graph**: An Entity-Relation Graph, which contains only entities and relations, is commonly represented as triples.
+ **KG**: A Knowledge Graph, which enriches entities with detailed descriptions and type information.
+ **RKG**: A Rich Knowledge Graph, which further incorporates keywords associated with relations.

The criteria for the classification of graph types are as follows:

|Graph Attributes | Chunk Tree |Passage Graph | ER  | KG | RKG |
| --- |--- |--- |--- | --- | --- |
|Original Content| ✅|✅| ❌|❌|❌| 
|Entity Name| ❌|❌|✅|✅|✅|
|Entity Type| ❌| ❌| ❌|✅|✅|
|Entity Description|❌| ❌| ❌|✅|✅|
|Relation Name| ❌|❌|✅|✅|✅|
|Edge Weight| ❌|❌|✅|✅|✅|
|Relation keyword|❌| ❌| ❌|❌|✅|
|Relation Description|❌| ❌| ❌|✅|✅|

