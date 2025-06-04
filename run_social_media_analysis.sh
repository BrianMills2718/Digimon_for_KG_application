#!/bin/bash

echo "Starting DIGIMON Social Media Analysis System..."
echo "=============================================="
echo ""

# Check if the CSV file exists
if [ ! -f "COVID-19-conspiracy-theories-tweets.csv" ]; then
    echo "WARNING: COVID-19-conspiracy-theories-tweets.csv not found!"
    echo "The system will try to download from Hugging Face on first use."
    echo ""
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    exit 1
fi

# Kill any existing Flask servers on port 5000
echo "Checking for existing servers on port 5000..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true

# Start the API server
echo "Starting API server on http://localhost:5000..."
python social_media_api_simple.py &
API_PID=$!

# Wait for server to start
sleep 3

# Open the UI in browser
echo ""
echo "Opening UI in browser..."
echo "If browser doesn't open automatically, visit:"
echo "file://$(pwd)/social_media_analysis_ui.html"
echo ""

# Try to open in browser (works on most systems)
if command -v xdg-open &> /dev/null; then
    xdg-open "file://$(pwd)/social_media_analysis_ui.html" 2>/dev/null
elif command -v open &> /dev/null; then
    open "file://$(pwd)/social_media_analysis_ui.html" 2>/dev/null
elif command -v start &> /dev/null; then
    start "file://$(pwd)/social_media_analysis_ui.html" 2>/dev/null
fi

echo "=============================================="
echo "System is running!"
echo ""
echo "API Server PID: $API_PID"
echo "Press Ctrl+C to stop the server"
echo ""
echo "To use the system:"
echo "1. Click 'Ingest Dataset' to load the CSV file"
echo "2. Click 'Generate Analysis Plan' to create scenarios" 
echo "3. Click 'Execute All Scenarios' to run the analysis"
echo "=============================================="

# Wait for Ctrl+C
trap "echo 'Stopping server...'; kill $API_PID 2>/dev/null; exit" INT
wait $API_PID