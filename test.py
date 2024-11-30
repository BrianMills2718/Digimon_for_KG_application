from Core.Graph.RKGraph import ERGraph

import asyncio
if __name__ == "__main__":
    
    with open("./book.txt") as f:
        doc = f.read()
    graph = ERGraph()

    chunks = asyncio.run(graph.chunk_documents(doc))

    asyncio.run(graph._construct_graph(chunks))

    
    # asyncio.run(graph.query("who are you"))
   