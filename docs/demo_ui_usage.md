# DIGIMON Social Media Analysis - How to Use

## Quick Start

1. **Open the UI in your browser:**
   ```
   file:///home/brian/digimon_cc/social_media_traced_ui.html
   ```

2. **Load the Dataset:**
   - Click the blue "Load COVID Dataset" button
   - Wait for "Dataset Ready!" message showing 6590 tweets loaded

3. **Configure Analysis:**
   - Set number of scenarios (default: 3)
   - **For Standard Analysis:** Leave "Use Discourse Analysis Framework" unchecked
   - **For Advanced Analysis:** Check "Use Discourse Analysis Framework"
     - This enables the five interrogatives framework (Who/Says What/To Whom/In What Setting/With What Effect)
     - You can modify the research focus question if desired

4. **Generate Analysis Plan:**
   - Click the green "Generate Plan" button
   - Review the generated scenarios in the right panel

5. **Execute Analysis:**
   - Click the purple "Execute Analysis" button
   - Watch real-time execution progress in the center panel
   - Monitor the progress bar and current steps
   - See execution trace events as they happen

6. **View Results:**
   - When complete, a modal will show detailed results
   - Click "View Detailed Results" to see insights
   - Download the full execution trace for detailed analysis

## What's Happening Behind the Scenes

### Standard Mode:
- Uses traditional GraphRAG approach
- Generates analysis scenarios based on complexity levels
- Executes entity extraction, relationship analysis, and insight generation

### Discourse Analysis Mode:
- Uses sophisticated discourse analysis framework
- Generates mini-ontologies for each interrogative view
- Analyzes conspiracy theories from multiple perspectives:
  - **Who:** Key actors, influencers, communities
  - **Says What:** Narratives, claims, themes
  - **To Whom:** Target audiences, recipients
  - **In What Setting:** Platforms, contexts
  - **With What Effect:** Outcomes, impacts
- Identifies cross-interrogative patterns
- Provides richer, more nuanced insights

## API is Running
The API server is already running at http://localhost:5000

## Features:
- Real-time execution tracing
- Progress monitoring
- Detailed insights per scenario
- Entity and relationship discovery
- Cross-interrogative pattern analysis (discourse mode)
- Full execution trace download

The system will analyze COVID-19 conspiracy theory tweets to understand how misinformation spreads, who spreads it, and what effects it has on different audiences.