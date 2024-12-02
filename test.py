from lazy_object_proxy.utils import await_

from Core.GraphRAG import GraphRAG

import asyncio
if __name__ == "__main__":
    
    with open("./book.txt") as f:
        doc = f.read()
    graph_rag = GraphRAG(working_dir = "./test_book")
    asyncio.run(graph_rag.insert([doc]))
    
    # asyncio.run(graph.query("who are you"))
   