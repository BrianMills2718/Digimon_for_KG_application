#!/usr/bin/env python3
"""Check available tools in registry"""

from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Option.Config2 import Config

config = Config.default()
orch = AgentOrchestrator(config=config, llm=None, encoder=None, chunk_factory=None)

print("Available tools in registry:")
for tool_id in sorted(orch.registry.keys()):
    print(f"  - {tool_id}")
    
print(f"\nTotal: {len(orch.registry)} tools")