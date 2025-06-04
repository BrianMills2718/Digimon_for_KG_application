# DIGIMON Social Media Analysis System

A working implementation of the COVID-19 conspiracy theory analysis system using DIGIMON's interrogative analysis framework.

## Quick Start

### Option 1: Use the startup script (Recommended)
```bash
./run_social_media_analysis.sh
```

This will:
- Start the API server on port 5000
- Open the UI in your browser
- Provide instructions for using the system

### Option 2: Manual startup
```bash
# Terminal 1: Start the API server
python social_media_api_simple.py

# Terminal 2: Open the UI
# Open social_media_analysis_ui.html in your browser
```

## How It Works

### 1. Dataset Ingestion
- Loads COVID-19 conspiracy theory tweets from local CSV or Hugging Face
- Extracts features like hashtags, mentions, and conspiracy types
- Provides dataset statistics

### 2. Analysis Planning
- Uses interrogative framework (Who/What/When/Where/Why/How)
- Generates 5 analysis scenarios by default
- Each scenario includes multiple perspectives and expected insights

### 3. Analysis Execution
- Currently provides simulated results for demonstration
- Shows entities found, narratives discovered, and influence mechanisms
- Displays metrics like entity count, relationships, and processing time

## Current Implementation Status

✅ **Working Features:**
- Dataset ingestion from local CSV
- Interrogative-based scenario generation  
- Web UI with real-time updates
- Simulated analysis execution with realistic results
- Progress tracking and result display
- JSON export of results

🚧 **In Development:**
- Full DIGIMON graph building integration
- Real entity and relationship extraction
- Actual graph-based analysis
- Vector similarity search

## File Structure

```
/home/brian/digimon_cc/
├── social_media_analysis_ui.html      # Web interface
├── social_media_api_simple.py         # Working API server
├── social_media_execution.py          # Full execution engine (WIP)
├── Core/AgentTools/
│   ├── social_media_dataset_tools.py  # Dataset ingestion
│   └── automated_interrogative_planner.py # Scenario generation
└── run_social_media_analysis.sh       # Startup script
```

## Using the System

1. **Start the system:**
   ```bash
   ./run_social_media_analysis.sh
   ```

2. **Load your dataset:**
   - Default: Uses local `COVID-19-conspiracy-theories-tweets.csv`
   - Alternative: Can load from Hugging Face

3. **Generate analysis scenarios:**
   - Select number of scenarios (1-10)
   - Choose complexity levels
   - Click "Generate Analysis Plan"

4. **Review generated scenarios:**
   - Each scenario has a research question
   - Multiple interrogative perspectives
   - Expected insights listed

5. **Execute analysis:**
   - Click "Execute All Scenarios"
   - Watch progress bar
   - View results when complete

6. **Export results:**
   - Click "Download Results (JSON)"
   - Results include all findings and metrics

## Example Results

The system generates insights like:

**Who Analysis:**
- Key influencers spreading conspiracies
- Bot networks identified
- User engagement patterns

**What Analysis:**
- Dominant conspiracy narratives
- Sentiment distribution
- Topic frequency

**How Analysis:**
- Spread mechanisms (hashtags, mentions)
- Effectiveness of different strategies
- Amplification patterns

## Troubleshooting

### Port Already in Use
```bash
# Kill existing process on port 5000
lsof -ti:5000 | xargs kill -9
```

### CSV File Not Found
- Ensure `COVID-19-conspiracy-theories-tweets.csv` is in the main directory
- Or update the path in the UI

### API Connection Failed
- Check that the API server is running on port 5000
- Verify no firewall blocking localhost connections

## Next Steps

To enable full DIGIMON integration:

1. Configure `Option/Config2.yaml` with API keys
2. Run `python social_media_execution.py` for full graph building
3. Update `social_media_api.py` to use real execution engine

## Demo Mode

The current implementation uses simulated results to demonstrate the UI and workflow. This allows testing the interface without requiring full graph building, which can be resource-intensive.

To switch to real analysis, replace `social_media_api_simple.py` with the full `social_media_api.py` implementation.