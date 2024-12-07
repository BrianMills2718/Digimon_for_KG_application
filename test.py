
from Core.GraphRAG import GraphRAG

import asyncio
if __name__ == "__main__":
    
    with open("./book.txt") as f:
        doc = f.read()
    graph_rag = GraphRAG(working_dir = "./test_book")
    asyncio.run(graph_rag.insert([doc]))
    
    asyncio.run(graph_rag.query("What relationship does Fred Gehrke have to the 23rd overall pick in the 2010 Major League Baseball Draft?"))
   