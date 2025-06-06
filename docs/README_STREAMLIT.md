# DIGIMON Streamlit Frontend

A modern web interface for controlling the DIGIMON agent system, built with Streamlit.

## Features

### 🤖 Agent Control
- **Standard Mode**: Generate complete execution plans upfront
- **ReAct Mode**: Iterative reasoning and action (experimental)
- Real-time query processing with plan visualization
- Support for all 10 GraphRAG methods (RAPTOR, HippoRAG, LightRAG, etc.)

### 📊 Dataset Management
- Browse available datasets (Fictional_Test, MySampleTexts, etc.)
- Upload new text files to create custom datasets
- View dataset statistics and file counts
- Automatic corpus preparation

### 🔍 Query Interface
- Natural language query input
- Real-time status updates during processing
- Comprehensive result display with retrieved context
- Support for both entity search and relationship analysis

### 📋 Plan Visualization
- Step-by-step execution tracking
- Tool call visualization
- Output flow between steps
- Error handling and debugging information

### 💬 Query History
- Persistent chat history across sessions
- Query replay and comparison
- Performance metrics and timing
- Export capabilities

### ⚙️ Configuration Management
- Method configuration viewing and editing
- Ontology management interface
- System status monitoring
- Build artifact management

## Installation

### Prerequisites
- Python 3.8+
- DIGIMON backend running on port 5000
- Required Python packages (installed automatically)

### Quick Start

1. **Start the Backend** (in one terminal):
   ```bash
   python api.py
   ```

2. **Start the Streamlit Frontend** (in another terminal):
   ```bash
   ./run_streamlit.sh
   ```
   
   Or manually:
   ```bash
   pip install streamlit plotly pandas requests
   streamlit run streamlit_agent_frontend.py --server.port 8501
   ```

3. **Open your browser** to: http://localhost:8501

## Usage Guide

### Basic Query Flow

1. **Select Dataset**: Choose from available datasets in the sidebar
2. **Choose Method**: Select a GraphRAG method (RAPTOR, HippoRAG, etc.)
3. **Set Agent Mode**: Standard (plan-first) or ReAct (iterative)
4. **Enter Query**: Type your natural language question
5. **Execute**: Click "Execute Query" to start processing
6. **View Results**: See the agent's answer and retrieved context

### Example Queries

```
"What are the main causes of the American Revolution?"
"Tell me about the Zorathian Empire technology"
"How are entities connected in my dataset?"
"Find influential people mentioned in the documents"
```

### Dataset Upload

1. Go to the **Corpus** tab
2. Click "Choose text files" to upload .txt files
3. Enter a new dataset name
4. Click "Upload Files"
5. The dataset will be available for querying

### Building Artifacts

Before querying, ensure artifacts are built:

1. Select your dataset and method in the sidebar
2. Click "🔨 Build Artifacts"
3. Wait for completion (status will show green)
4. You can now execute queries

## Architecture

### Frontend Components

- **Query Interface** (`render_query_interface()`): Main query execution
- **Agent Controls** (`render_sidebar()`): Dataset/method selection and build management
- **Result Display** (`render_query_results()`): Answer and context visualization
- **Chat History** (`render_chat_history()`): Query history and session management
- **Corpus Management** (`render_corpus_management()`): Dataset upload and management
- **Configuration** (`render_configuration()`): System settings and ontology

### API Integration

The frontend communicates with the DIGIMON backend via REST API:

- `POST /api/query`: Execute queries
- `POST /api/build`: Build artifacts
- `GET /api/ontology`: Retrieve ontology
- `POST /api/ontology`: Save ontology
- `POST /api/evaluate`: Run evaluations

### Session State

Streamlit session state maintains:
- Query history and results
- Selected dataset and method
- Build status per dataset/method combination
- Agent mode preferences

## ReAct Mode

The ReAct (Reason-Act-Observe) mode provides iterative query processing:

1. **Reason**: Agent analyzes the current situation
2. **Act**: Executes the next logical step
3. **Observe**: Reviews results and decides next action
4. **Repeat**: Until sufficient information is gathered

### ReAct Features
- Step-by-step reasoning visualization
- Dynamic plan adaptation
- Intermediate result inspection
- Iteration limit protection

## Troubleshooting

### Common Issues

**"Cannot connect to DIGIMON backend"**
- Ensure `python api.py` is running
- Check that port 5000 is not blocked
- Verify API_BASE_URL in the frontend code

**"Build failed"**
- Check dataset exists in Data/ directory
- Verify method configuration files exist
- Review backend logs for detailed errors

**"Query timeout"**
- Large datasets may take time to process
- Try smaller datasets first
- Check system resources

**"No results returned"**
- Ensure artifacts are built before querying
- Verify dataset contains relevant content
- Try different query phrasings

### Debug Mode

Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Performance Tips

- Build artifacts once per dataset/method combination
- Use smaller datasets for testing
- Monitor system resources during large builds
- Clear browser cache if interface becomes slow

## Advanced Features

### Custom Ontologies

1. Go to Configuration tab
2. View current ontology structure
3. Use the backend API for ontology chat to design new ones
4. Ontologies are automatically applied to new builds

### Method Comparison

1. Execute the same query with different methods
2. Compare results in the History tab
3. Analyze which methods work best for your use case

### Batch Processing

While the UI is designed for interactive use, you can:
- Save queries from History tab
- Use the backend API directly for batch operations
- Export results for further analysis

## Development

### Adding New Features

1. **New API Endpoints**: Update `api_request()` function
2. **UI Components**: Add new render functions
3. **Session State**: Update `init_session_state()`
4. **Tabs**: Modify main tab structure

### Customization

- **Styling**: Modify Streamlit theme in `.streamlit/config.toml`
- **Layout**: Adjust column ratios and component sizes
- **Data Display**: Customize table and chart formatting
- **API Integration**: Extend for new backend features

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with proper error handling
4. Test with multiple datasets and methods
5. Submit a pull request

## License

Same as the main DIGIMON project.