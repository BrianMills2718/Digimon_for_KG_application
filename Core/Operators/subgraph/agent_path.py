"""Subgraph agent path operator.

Use LLM to filter and select relevant paths from a subgraph.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from Core.Common.Logger import logger
from Core.Schema.SlotTypes import SlotKind, SlotValue, SubgraphRecord


async def subgraph_agent_path(
    inputs: Dict[str, SlotValue],
    ctx: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, SlotValue]:
    """
    Inputs:  {"query": SlotValue(QUERY_TEXT), "subgraph": SlotValue(SUBGRAPH)}
    Outputs: {"subgraph": SlotValue(SUBGRAPH)}

    Filters paths in the subgraph using LLM relevance scoring.
    """
    query = inputs["query"].data
    sg = inputs["subgraph"].data  # SubgraphRecord

    if not sg or not sg.paths:
        return {"subgraph": SlotValue(kind=SlotKind.SUBGRAPH, data=sg or SubgraphRecord(nodes=set(), edges=[]), producer="subgraph.agent_path")}

    try:
        # Format paths for LLM
        path_strs = []
        for i, path in enumerate(sg.paths):
            path_strs.append(f"Path {i+1}: {' -> '.join(path)}")

        prompt = (
            f"Given the question: {query}\n\n"
            f"Here are reasoning paths in a knowledge graph:\n"
            + "\n".join(path_strs)
            + "\n\nWhich paths are most relevant to answering the question? "
            "Return the path numbers (comma-separated) that are relevant."
        )

        result = await ctx.llm.aask(msg=[{"role": "user", "content": prompt}])

        # Parse selected path numbers
        import re
        numbers = [int(n) - 1 for n in re.findall(r"\d+", result)]
        selected_paths = [sg.paths[n] for n in numbers if 0 <= n < len(sg.paths)]

        if not selected_paths:
            selected_paths = sg.paths  # fallback to all

        # Build filtered subgraph
        all_nodes = set()
        all_edges = []
        for path in selected_paths:
            for node in path:
                all_nodes.add(node)
            for i in range(len(path) - 1):
                all_edges.append((path[i], path[i + 1]))

        return {"subgraph": SlotValue(
            kind=SlotKind.SUBGRAPH,
            data=SubgraphRecord(nodes=all_nodes, edges=all_edges, paths=selected_paths),
            producer="subgraph.agent_path",
        )}

    except Exception as e:
        logger.exception(f"subgraph_agent_path failed: {e}")
        return {"subgraph": SlotValue(kind=SlotKind.SUBGRAPH, data=sg, producer="subgraph.agent_path")}
