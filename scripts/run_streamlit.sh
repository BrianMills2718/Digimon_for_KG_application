#!/bin/bash

# DIGIMON Streamlit Frontend Runner
# This script starts the Streamlit frontend for the DIGIMON agent control system

echo "🤖 Starting DIGIMON Agent Control System Frontend..."
echo "=================================================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed. Installing..."
    pip install streamlit plotly pandas requests
fi

# Check if the backend API is running
echo "🔍 Checking backend API connection..."
if curl -s http://localhost:5000/api/ontology > /dev/null 2>&1; then
    echo "✅ Backend API is running on port 5000"
else
    echo "⚠️  Backend API not detected. Make sure to start it with:"
    echo "   python api.py"
    echo ""
    echo "🚀 Starting frontend anyway..."
fi

# Start Streamlit
echo "🌐 Starting Streamlit frontend on http://localhost:8502"
echo "📝 Use Ctrl+C to stop the server"
echo "=================================================="

streamlit run streamlit_agent_frontend.py --server.port 8502 --server.headless false