#!/bin/bash

# Run DIGIMON MCP Server

echo "Starting DIGIMON MCP Server..."

# Activate conda environment if it exists
if command -v conda &> /dev/null; then
    echo "Activating digimon conda environment..."
    conda activate digimon
fi

# Check if Config2.yaml exists
if [ ! -f "Option/Config2.yaml" ]; then
    echo "Error: Option/Config2.yaml not found!"
    echo "Please copy Option/Config2.example.yaml to Option/Config2.yaml and configure it."
    exit 1
fi

# Run the MCP server
python -m Core.MCP.digimon_mcp_server --port 8765

echo "MCP Server stopped."