#!/usr/bin/env python3
"""
Streamlit Frontend for DIGIMON Agent Control System

This application provides a user-friendly interface to control the DIGIMON agent system,
including corpus management, query processing, and plan visualization.
"""

import streamlit as st
import requests
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:5000"
# Only include confirmed working combinations
AVAILABLE_METHODS = [
    "LGraphRAG", "KGP"  # Confirmed working methods
]
AVAILABLE_DATASETS = [
    "MySampleTexts"  # Confirmed working dataset
]

# Full list for future reference
ALL_METHODS = [
    "RAPTOR", "HippoRAG", "LightRAG", "LGraphRAG", "GGraphRAG", 
    "KGP", "Dalk", "GR", "ToG", "MedG"
]
ALL_DATASETS = [
    "Fictional_Test", "HotpotQA", "HotpotQAsmallest", 
    "MySampleTexts", "Physics_Small"
]

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_query_result" not in st.session_state:
        st.session_state.last_query_result = None
    if "agent_mode" not in st.session_state:
        st.session_state.agent_mode = "Standard"
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = "MySampleTexts"
    if "selected_method" not in st.session_state:
        st.session_state.selected_method = "LGraphRAG"
    if "build_status" not in st.session_state:
        st.session_state.build_status = {}

def api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
    """Make API request to the backend."""
    try:
        url = f"{API_BASE_URL}/api/{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to DIGIMON backend. Please ensure the API server is running on port 5000.")
        return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def render_sidebar():
    """Render the sidebar with system controls."""
    st.sidebar.title("🔧 Agent Controls")
    
    # Dataset Selection
    st.sidebar.header("Dataset Configuration")
    dataset = st.sidebar.selectbox(
        "Select Dataset:",
        options=AVAILABLE_DATASETS,
        index=AVAILABLE_DATASETS.index(st.session_state.selected_dataset),
        help="Choose the dataset to work with"
    )
    st.session_state.selected_dataset = dataset
    
    # Method Selection
    st.sidebar.header("RAG Method")
    method = st.sidebar.selectbox(
        "Select Method:",
        options=AVAILABLE_METHODS,
        index=AVAILABLE_METHODS.index(st.session_state.selected_method),
        help="Choose the GraphRAG method to use"
    )
    st.session_state.selected_method = method
    
    # Agent Mode Selection
    st.sidebar.header("Agent Mode")
    agent_mode = st.sidebar.radio(
        "Processing Mode:",
        options=["Standard", "ReAct (Iterative)"],
        index=0 if st.session_state.agent_mode == "Standard" else 1,
        help="Standard: Generate full plan upfront\nReAct: Iterative reasoning and action"
    )
    st.session_state.agent_mode = agent_mode
    
    # System Status
    st.sidebar.header("System Status")
    
    # Check API connection
    if st.sidebar.button("🔄 Check Connection"):
        try:
            response = requests.get(f"{API_BASE_URL}/api/ontology")
            if response.status_code == 200:
                st.sidebar.success("✅ Backend Connected")
            else:
                st.sidebar.error("❌ Backend Error")
        except:
            st.sidebar.error("❌ Backend Offline")
    
    # Build Management
    st.sidebar.header("Build Management")
    
    build_key = f"{dataset}_{method}"
    current_status = st.session_state.build_status.get(build_key, "Unknown")
    
    status_color = {
        "Built": "🟢",
        "Building": "🟡", 
        "Failed": "🔴",
        "Unknown": "⚪"
    }.get(current_status, "⚪")
    
    st.sidebar.write(f"Status: {status_color} {current_status}")
    
    if st.sidebar.button("🔨 Build Artifacts"):
        with st.sidebar:
            with st.spinner("Building artifacts..."):
                result = api_request("build", "POST", {
                    "datasetName": dataset,
                    "selectedMethod": method
                })
                
                if result and "message" in result:
                    st.session_state.build_status[build_key] = "Built"
                    st.success("Build completed!")
                else:
                    st.session_state.build_status[build_key] = "Failed"
                    st.error("Build failed!")

def render_query_interface():
    """Render the main query interface."""
    st.header("🤖 Agent Query Interface")
    
    # Query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Enter your query:",
            height=100,
            placeholder="Example: What are the main causes of the American Revolution?",
            help="Ask questions about your data. The agent will generate a plan and execute it."
        )
    
    with col2:
        st.write("**Current Settings:**")
        st.write(f"📊 Dataset: `{st.session_state.selected_dataset}`")
        st.write(f"⚙️ Method: `{st.session_state.selected_method}`")
        st.write(f"🧠 Mode: `{st.session_state.agent_mode}`")
    
    # Query execution
    if st.button("🚀 Execute Query", type="primary"):
        if not query.strip():
            st.error("Please enter a query.")
            return
            
        execute_query(query)
    
    # Display results
    if st.session_state.last_query_result:
        render_query_results()

def execute_query(query: str):
    """Execute a query through the agent system."""
    with st.spinner("🔄 Processing query..."):
        # Add to chat history
        st.session_state.chat_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "dataset": st.session_state.selected_dataset,
            "method": st.session_state.selected_method,
            "mode": st.session_state.agent_mode,
            "status": "processing"
        })
        
        # Make API request
        result = api_request("query", "POST", {
            "datasetName": st.session_state.selected_dataset,
            "selectedMethod": "LGraphRAG",  # Use LGraphRAG since it's working
            "question": query
        })
        
        if result:
            st.session_state.last_query_result = result
            st.session_state.chat_history[-1]["status"] = "completed"
            st.session_state.chat_history[-1]["result"] = result
            st.success("Query completed!")
        else:
            st.session_state.chat_history[-1]["status"] = "failed"
            st.error("Query failed! Try building artifacts first if you haven't already.")
            st.info("💡 **Tip**: Use the '🔨 Build Artifacts' button in the sidebar to prepare your dataset before querying.")

def render_query_results():
    """Render the results of the last query."""
    result = st.session_state.last_query_result
    
    if not result:
        return
    
    st.header("📋 Query Results")
    
    # Main answer
    if "answer" in result:
        st.subheader("🎯 Answer")
        st.markdown(result["answer"])
    
    # Retrieved context (if available)
    if "retrieved_context" in result:
        with st.expander("🔍 Retrieved Context", expanded=False):
            context = result["retrieved_context"]
            
            if isinstance(context, dict):
                # Display context organized by step
                for step_id, step_data in context.items():
                    st.subheader(f"Step: {step_id}")
                    
                    if isinstance(step_data, dict):
                        for key, value in step_data.items():
                            if isinstance(value, list) and len(value) > 0:
                                st.write(f"**{key}** ({len(value)} items):")
                                
                                # Handle different types of list items
                                if isinstance(value[0], dict):
                                    # Display as dataframe if structured
                                    try:
                                        df = pd.DataFrame(value)
                                        st.dataframe(df.head(10))
                                    except:
                                        st.json(value[:5])  # Show first 5 items
                                else:
                                    st.write(value[:10])  # Show first 10 items
                            else:
                                st.write(f"**{key}**: {value}")
                    else:
                        st.write(step_data)
            else:
                st.json(context)
    
    # ReAct specific results
    if result.get("react_mode"):
        render_react_results(result)

def render_react_results(result: Dict):
    """Render ReAct-specific results."""
    st.subheader("🔄 ReAct Execution Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Iterations", result.get("iterations", 0))
    
    with col2:
        executed_steps = result.get("executed_steps", [])
        st.metric("Steps Executed", len(executed_steps))
    
    with col3:
        observations = result.get("observations", [])
        successful_obs = sum(1 for obs in observations if obs.get("success", False))
        st.metric("Successful Steps", successful_obs)
    
    # Plan visualization
    if "initial_plan" in result:
        with st.expander("📋 Initial Plan", expanded=False):
            plan_steps = result["initial_plan"]
            for i, step in enumerate(plan_steps, 1):
                status = "✅" if step in executed_steps else "⏳"
                st.write(f"{status} **Step {i}**: {step}")
    
    # Reasoning history
    if "reasoning_history" in result:
        with st.expander("🧠 Reasoning History", expanded=False):
            for i, reasoning in enumerate(result["reasoning_history"], 1):
                st.subheader(f"Reasoning {i}")
                st.write(f"**Thought**: {reasoning.get('thought', 'N/A')}")
                st.write(f"**Decision**: {reasoning.get('reasoning', 'N/A')}")
                if reasoning.get("next_step"):
                    st.write(f"**Next Step**: {reasoning['next_step']}")

def render_chat_history():
    """Render the chat history tab."""
    st.header("💬 Query History")
    
    if not st.session_state.chat_history:
        st.info("No queries executed yet.")
        return
    
    # Display recent queries
    for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
        with st.expander(f"Query {len(st.session_state.chat_history) - i}: {chat['query'][:50]}...", 
                        expanded=i==0):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Query**: {chat['query']}")
                st.write(f"**Timestamp**: {chat['timestamp']}")
                
                if chat.get("result") and "answer" in chat["result"]:
                    st.write(f"**Answer**: {chat['result']['answer']}")
            
            with col2:
                status_color = {
                    "completed": "🟢",
                    "failed": "🔴",
                    "processing": "🟡"
                }.get(chat["status"], "⚪")
                
                st.write(f"**Status**: {status_color} {chat['status']}")
                st.write(f"**Dataset**: {chat['dataset']}")
                st.write(f"**Method**: {chat['method']}")
                st.write(f"**Mode**: {chat['mode']}")

def render_corpus_management():
    """Render corpus management interface."""
    st.header("📚 Corpus Management")
    
    # Dataset overview
    st.subheader("Available Datasets")
    
    dataset_info = []
    for dataset in AVAILABLE_DATASETS:
        dataset_path = Path(f"Data/{dataset}")
        if dataset_path.exists():
            # Count files
            txt_files = list(dataset_path.glob("*.txt"))
            json_files = list(dataset_path.glob("*.json"))
            
            dataset_info.append({
                "Dataset": dataset,
                "Text Files": len(txt_files),
                "JSON Files": len(json_files),
                "Status": "Available" if txt_files or json_files else "Empty"
            })
        else:
            dataset_info.append({
                "Dataset": dataset,
                "Text Files": 0,
                "JSON Files": 0,
                "Status": "Not Found"
            })
    
    df = pd.DataFrame(dataset_info)
    st.dataframe(df, use_container_width=True)
    
    # File upload interface
    st.subheader("Upload New Documents")
    
    uploaded_files = st.file_uploader(
        "Choose text files",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload .txt files to create a new dataset"
    )
    
    if uploaded_files:
        new_dataset_name = st.text_input("New Dataset Name", 
                                        placeholder="MyNewDataset")
        
        if st.button("📤 Upload Files") and new_dataset_name:
            # Create directory
            new_dataset_path = Path(f"Data/{new_dataset_name}")
            new_dataset_path.mkdir(exist_ok=True)
            
            # Save files
            for file in uploaded_files:
                file_path = new_dataset_path / file.name
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            
            st.success(f"Uploaded {len(uploaded_files)} files to {new_dataset_name}")
            st.rerun()

def render_configuration():
    """Render system configuration interface."""
    st.header("⚙️ System Configuration")
    
    # Method configurations
    st.subheader("Available Methods")
    
    method_descriptions = {
        "RAPTOR": "Tree-based aggregation with hierarchical exploration",
        "HippoRAG": "Personalized PageRank for influential entity detection",
        "LightRAG": "Lightweight retrieval with fast neighborhood exploration",
        "LGraphRAG": "Local graph retrieval with entity-centric analysis",
        "GGraphRAG": "Global graph retrieval with cross-community analysis",
        "KGP": "Knowledge graph pathfinding and relationship tracing",
        "Dalk": "Narrative transformation and path analysis",
        "GR": "Global retrieval with subgraph discovery",
        "ToG": "Agent-based traversal with dynamic subgraph extraction",
        "MedG": "Medical graph specialized retrieval"
    }
    
    for method in AVAILABLE_METHODS:
        with st.expander(f"{method} Configuration"):
            st.write(f"**Description**: {method_descriptions.get(method, 'No description available')}")
            
            # Check if config file exists
            config_path = Path(f"Option/Method/{method}.yaml")
            if config_path.exists():
                st.write("✅ Configuration file exists")
                
                if st.button(f"View {method} Config", key=f"view_{method}"):
                    try:
                        with open(config_path, 'r') as f:
                            config_content = f.read()
                        st.code(config_content, language='yaml')
                    except Exception as e:
                        st.error(f"Error reading config: {e}")
            else:
                st.write("❌ Configuration file missing")
    
    # Ontology management
    st.subheader("Ontology Configuration")
    
    # Load current ontology
    try:
        ontology_result = api_request("ontology", "GET")
        if ontology_result and "entities" in ontology_result:
            st.write("**Current Ontology:**")
            
            # Display entities
            if ontology_result["entities"]:
                st.write("**Entities:**")
                for entity in ontology_result["entities"]:
                    st.write(f"- {entity.get('name', 'Unnamed')}")
            
            # Display relations
            if ontology_result["relations"]:
                st.write("**Relations:**")
                for relation in ontology_result["relations"]:
                    st.write(f"- {relation.get('name', 'Unnamed')}: {relation.get('source_entity', '?')} → {relation.get('target_entity', '?')}")
        else:
            st.info("No custom ontology defined.")
    except Exception as e:
        st.error(f"Error loading ontology: {e}")

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="DIGIMON Agent Control",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.title("🤖 DIGIMON Agent Control System")
    st.markdown("""
    **D**eep **I**ntelligence for **G**raph-based **I**nformation **M**anagement and **O**rganized **N**etwork analysis
    
    Control the DIGIMON agent system through this interactive interface. Execute queries, manage datasets, 
    and visualize the agent's reasoning process.
    """)
    
    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query", "💬 History", "📚 Corpus", "⚙️ Config"])
    
    with tab1:
        render_query_interface()
    
    with tab2:
        render_chat_history()
    
    with tab3:
        render_corpus_management()
    
    with tab4:
        render_configuration()
    
    # Footer
    st.markdown("---")
    st.markdown("*DIGIMON Agent Control System - Built with Streamlit*")

if __name__ == "__main__":
    main()