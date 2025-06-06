#!/usr/bin/env python
"""Start the MCP server for testing"""

import asyncio
import sys
sys.path.append('/home/brian/digimon_cc')

from Core.MCP.base_mcp_server import start_server

if __name__ == "__main__":
    print("Starting MCP test server on port 8765...")
    asyncio.run(start_server())