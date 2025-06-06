#!/bin/bash
# Run DIGIMON Social Media Analysis with Execution Tracing

echo "===========================================" 
echo "DIGIMON Social Media Analysis with Tracing"
echo "==========================================="
echo ""

# Check if dataset exists
if [ ! -f "COVID-19-conspiracy-theories-tweets.csv" ]; then
    echo "⚠️  WARNING: COVID-19-conspiracy-theories-tweets.csv not found!"
    echo "Downloading dataset..."
    python download_covid_conspiracy_dataset.py
    if [ $? -ne 0 ]; then
        echo "Failed to download dataset. Please check your internet connection."
        echo "You can also manually download from:"
        echo "https://huggingface.co/datasets/webimmunization/COVID-19-conspiracy-theories-tweets"
        exit 1
    fi
else
    echo "✓ Dataset found: COVID-19-conspiracy-theories-tweets.csv"
    # Show dataset info
    echo "  $(wc -l < COVID-19-conspiracy-theories-tweets.csv) rows"
fi

# Check if config exists
if [ ! -f "Option/Config2.yaml" ]; then
    echo ""
    echo "⚠️  Warning: Config file not found!"
    echo "Please copy and configure the example config:"
    echo "  cp Option/Config2.example.yaml Option/Config2.yaml"
    echo "  Then edit it to add your API keys."
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Check Python
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        alias python=python3
    else
        echo "ERROR: Python is not installed or not in PATH"
        exit 1
    fi
fi

# Kill any existing servers on port 5000
echo ""
echo "Checking for existing servers..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true

# Start the traced API server
echo ""
echo "Starting DIGIMON API server with execution tracing..."
echo "API will be available at: http://localhost:5000"
echo ""

# Run the traced version
python social_media_api_traced.py &
API_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
for i in {1..10}; do
    if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
        echo "✓ API server is ready!"
        break
    fi
    sleep 1
done

# Check if server started successfully
if ! ps -p $API_PID > /dev/null; then
    echo "✗ Failed to start API server"
    echo "Please check if all dependencies are installed:"
    echo "  pip install flask flask-cors pandas datasets"
    exit 1
fi

# Try to open browser
echo ""
echo "===========================================" 
echo "System is running!"
echo ""
echo "Access the analysis UI at:"
echo ""
echo "  📊 Basic UI: file://$PWD/social_media_analysis_ui.html"
echo "  🔍 Traced UI: file://$PWD/social_media_traced_ui.html (Recommended)"
echo ""
echo "API Status: http://localhost:5000/api/health"
echo "Server PID: $API_PID"
echo ""
echo "Usage:"
echo "1. Open the Traced UI in your browser"
echo "2. Click 'Load COVID Dataset' to ingest the data"
echo "3. Click 'Generate Plan' to create analysis scenarios"
echo "4. Click 'Execute Analysis' to run with full tracing"
echo ""
echo "Press Ctrl+C to stop the server"
echo "==========================================="

# Try to open traced UI in browser
if command -v xdg-open &> /dev/null; then
    xdg-open "file://$(pwd)/social_media_traced_ui.html" 2>/dev/null
elif command -v open &> /dev/null; then
    open "file://$(pwd)/social_media_traced_ui.html" 2>/dev/null
fi

# Wait for interrupt
trap "echo ''; echo 'Stopping server...'; kill $API_PID 2>/dev/null; exit" INT TERM
wait $API_PID