# DIGIMON Social Media Analysis UI

A web-based interface for analyzing COVID-19 conspiracy theories on social media using DIGIMON's interrogative analysis framework.

## Features

- **Dataset Ingestion**: Load COVID-19 conspiracy theory tweets from Hugging Face
- **Automated Planning**: Generate analysis scenarios using interrogative views (Who/What/When/Where/Why/How)
- **Interactive Visualization**: Explore analysis scenarios with detailed breakdowns
- **Execution Framework**: Execute analysis plans using DIGIMON's graph-based tools

## Quick Start

### 1. Start the API Server

```bash
python social_media_api.py
```

The API server will start on `http://localhost:5000`

### 2. Open the UI

You have several options:

**Option A: Direct File Access**
```bash
# Open in browser
file:///home/brian/digimon_cc/social_media_analysis_ui.html
```

**Option B: Serve with Python**
```bash
python -m http.server 8080
# Then visit: http://localhost:8080/social_media_analysis_ui.html
```

**Option C: Use with Puppeteer MCP**
See `puppeteer_ui_demo.js` for automation examples

## UI Workflow

### Step 1: Dataset Ingestion
1. Enter the Hugging Face dataset name (default: `webimmunization/COVID-19-conspiracy-theories-tweets`)
2. Optionally set max rows for faster testing
3. Click "Ingest Dataset"
4. Wait for the dataset to load (shows row count and conspiracy types)

### Step 2: Configure Analysis
1. Set the number of scenarios to generate (1-10)
2. Select complexity levels:
   - **Simple**: Basic pattern identification
   - **Medium**: Relationship discovery and trend analysis
   - **Complex**: Causal analysis and predictions
3. Click "Generate Analysis Plan"

### Step 3: Explore Scenarios
1. Review generated scenarios in the main panel
2. Each scenario shows:
   - Title and complexity level
   - Research question
   - Interrogative views used
3. Click "View Details" to see:
   - Full interrogative view descriptions
   - Entity and relationship types
   - Analysis pipeline steps
   - Expected insights

### Step 4: Execute Analysis
1. Click "Execute All Scenarios" to run the analysis
2. The system will use DIGIMON's tools to:
   - Build knowledge graphs
   - Extract entities and relationships
   - Perform interrogative-based retrieval
   - Generate insights

## API Endpoints

- `POST /api/ingest-dataset`: Load dataset from Hugging Face
- `POST /api/generate-plan`: Generate analysis scenarios
- `POST /api/execute-analysis`: Execute analysis plan
- `GET /api/health`: Health check

## Technology Stack

- **Frontend**: HTML5, Tailwind CSS, Alpine.js
- **Backend**: Flask, Python async
- **Analysis**: DIGIMON GraphRAG tools
- **Automation**: Puppeteer MCP compatible

## Using with Puppeteer MCP

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
    }
  }
}
```

Then use commands like:
- `puppeteer_navigate` to open the UI
- `puppeteer_screenshot` to capture states
- `puppeteer_click` to interact with buttons
- `puppeteer_fill` to enter data

See `puppeteer_ui_demo.js` for complete automation examples.

## Extending the UI

### Adding New Analysis Types
1. Modify `automated_interrogative_planner.py` to add new scenario templates
2. Update the UI to display new scenario types
3. Implement execution logic in the API

### Custom Visualizations
1. Add new visualization components to the UI
2. Create API endpoints for specific data formats
3. Integrate with DIGIMON's graph visualization tools

## Troubleshooting

### Dataset Loading Issues
- Check internet connection for Hugging Face access
- Verify dataset name is correct
- Try with smaller max_rows value

### API Connection Errors
- Ensure Flask server is running on port 5000
- Check CORS is enabled
- Verify no firewall blocking

### UI Not Updating
- Check browser console for JavaScript errors
- Ensure Alpine.js is loading properly
- Try hard refresh (Ctrl+Shift+R)

## Next Steps

1. Implement full execution pipeline with graph building
2. Add real-time progress tracking
3. Create result visualization components
4. Export analysis reports
5. Add user authentication and session management