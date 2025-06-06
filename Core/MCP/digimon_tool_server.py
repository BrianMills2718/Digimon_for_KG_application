"""
DIGIMON MCP Server with Tool Support

This server extends the base MCP server to include DIGIMON-specific tools.
"""

import asyncio
import logging
from typing import Dict, Any
from Core.MCP.base_mcp_server import MCPServer
from Core.MCP.tools import (
    entity_vdb_search_mcp_tool,
    entity_vdb_build_mcp_tool,
    entity_ppr_mcp_tool,
    build_er_graph_mcp_tool,
    build_rk_graph_mcp_tool,
    build_tree_graph_mcp_tool,
    build_tree_graph_balanced_mcp_tool,
    build_passage_graph_mcp_tool,
    corpus_prepare_mcp_tool
)

logger = logging.getLogger(__name__)


class DigimonMCPServer(MCPServer):
    """Enhanced MCP server with DIGIMON tools"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        super().__init__(host, port)
        self._register_digimon_tools()
        
    def _register_digimon_tools(self):
        """Register DIGIMON-specific tools"""
        # Register all tools
        tools_to_register = [
            # Entity tools
            entity_vdb_search_mcp_tool,
            entity_vdb_build_mcp_tool,
            entity_ppr_mcp_tool,
            # Graph building tools
            build_er_graph_mcp_tool,
            build_rk_graph_mcp_tool,
            build_tree_graph_mcp_tool,
            build_tree_graph_balanced_mcp_tool,
            build_passage_graph_mcp_tool,
            # Corpus tools
            corpus_prepare_mcp_tool
        ]
        
        for tool in tools_to_register:
            tool_def = tool.get_tool_definition()
            tool_def["handler"] = tool
            self.register_tool(tool_def)
            logger.info(f"Registered tool: {tool_def['name']}")
        
        logger.info(f"Registered {len(self.tools)} tools total")
        

async def start_digimon_server(host: str = "localhost", port: int = 8765):
    """Start the DIGIMON MCP server"""
    server = DigimonMCPServer(host, port)
    await server.start()
    
    # Keep server running
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        await server.stop()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DIGIMON MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Run server
    asyncio.run(start_digimon_server(args.host, args.port))


if __name__ == "__main__":
    main()