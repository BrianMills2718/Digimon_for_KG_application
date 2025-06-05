#!/usr/bin/env python3
"""
DIGIMON Streamlit Dashboard - Quick usability improvement
"""

import streamlit as st
import asyncio
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from Core.AgentOrchestrator.memory_enhanced_orchestrator import MemoryEnhancedOrchestrator
from Core.AgentBrain.agent_brain import AgentBrain
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Config.LLMConfig import LLMConfig
from Core.Common.Logger import logger

# Page config
st.set_page_config(
    page_title="DIGIMON GraphRAG Dashboard",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_job' not in st.session_state:
    st.session_state.current_job = None
if 'orchestrator' not in st.session_state:
    # Initialize once and reuse
    config_path = Path("Option/Config2.yaml")
    llm_config = LLMConfig.from_yaml(str(config_path))
    llm = LiteLLMProvider(llm_config)
    brain = AgentBrain(llm)
    st.session_state.orchestrator = MemoryEnhancedOrchestrator(brain=brain, llm=llm)

# Header
st.title("🔬 DIGIMON GraphRAG Analysis Dashboard")
st.markdown("**Powerful discourse analysis for social media and text corpora**")

# Sidebar
with st.sidebar:
    st.header("📊 Analysis Options")
    
    # Corpus selection
    corpus_options = {
        "COVID Conspiracy Tweets": "Data/COVID_Conspiracy/Corpus.json",
        "Upload Custom Dataset": "custom"
    }
    
    selected_corpus = st.selectbox(
        "Select Dataset",
        options=list(corpus_options.keys())
    )
    
    if selected_corpus == "Upload Custom Dataset":
        uploaded_file = st.file_uploader(
            "Upload Corpus (JSON/CSV)",
            type=['json', 'csv']
        )
        if uploaded_file:
            # Handle file upload
            corpus_path = f"temp_{uploaded_file.name}"
            with open(corpus_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    else:
        corpus_path = corpus_options[selected_corpus]
    
    st.divider()
    
    # Analysis presets
    st.subheader("🎯 Quick Analysis Templates")
    
    preset_analyses = {
        "Influence Network Analysis": "Identify key influencers and map information flow networks",
        "Narrative Evolution": "Track how narratives change and mutate over time",
        "Community Detection": "Find and characterize different discourse communities",
        "Sentiment Dynamics": "Analyze emotional patterns and polarization",
        "Custom Query": "Write your own analysis query"
    }
    
    selected_preset = st.selectbox(
        "Choose Analysis Type",
        options=list(preset_analyses.keys())
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        max_cost = st.slider("Maximum Cost ($)", 0.0, 50.0, 10.0, 0.5)
        parallel_execution = st.checkbox("Enable Parallel Processing", value=True)
        include_visualizations = st.checkbox("Generate Visualizations", value=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Analysis Configuration")
    
    if selected_preset == "Custom Query":
        query = st.text_area(
            "Enter your analysis query:",
            height=150,
            placeholder="Example: Analyze the network structure of conspiracy theorists and identify the main narratives they promote..."
        )
    else:
        st.info(f"**{selected_preset}**: {preset_analyses[selected_preset]}")
        query = preset_analyses[selected_preset]
    
    # Analysis button
    if st.button("🚀 Start Analysis", type="primary", disabled=not corpus_path):
        with st.spinner("Running analysis..."):
            # Create job
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.current_job = {
                "id": job_id,
                "query": query,
                "corpus": selected_corpus,
                "status": "running",
                "start_time": datetime.now()
            }
            
            # Run analysis
            try:
                # Create async event loop for Streamlit
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    st.session_state.orchestrator.process_query(
                        query=query,
                        context={
                            "corpus_path": corpus_path,
                            "dashboard_mode": True,
                            "cost_limit": max_cost
                        }
                    )
                )
                
                # Update job status
                st.session_state.current_job["status"] = "completed"
                st.session_state.current_job["result"] = result
                st.session_state.current_job["end_time"] = datetime.now()
                
                # Add to history
                st.session_state.analysis_history.append(st.session_state.current_job)
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.session_state.current_job["status"] = "error"
                st.session_state.current_job["error"] = str(e)
                st.error(f"❌ Analysis failed: {e}")

with col2:
    st.header("📈 Quick Stats")
    
    if corpus_path and corpus_path != "custom":
        try:
            # Load corpus stats
            with open(corpus_path, 'r') as f:
                corpus_data = json.load(f)
            
            st.metric("Documents", len(corpus_data))
            
            # Calculate total tokens (rough estimate)
            total_chars = sum(len(doc.get('content', '')) for doc in corpus_data)
            estimated_tokens = total_chars // 4
            
            st.metric("Estimated Tokens", f"{estimated_tokens:,}")
            st.metric("Analyses Run", len(st.session_state.analysis_history))
            
        except:
            st.info("Upload a dataset to see statistics")

# Results section
st.divider()
st.header("📊 Analysis Results")

if st.session_state.current_job and st.session_state.current_job["status"] == "completed":
    result = st.session_state.current_job["result"]
    
    # Display main answer
    st.subheader("Key Findings")
    st.write(result.get("answer", "No answer generated"))
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Entities Found", result.get("entity_count", 0))
    with col2:
        st.metric("Relationships", result.get("relationship_count", 0))
    with col3:
        st.metric("Communities", result.get("community_count", 0))
    with col4:
        execution_time = (st.session_state.current_job["end_time"] - 
                         st.session_state.current_job["start_time"]).total_seconds()
        st.metric("Execution Time", f"{execution_time:.1f}s")
    
    # Visualizations
    if include_visualizations and result.get("graph_data"):
        st.subheader("Network Visualization")
        
        # Create network graph
        graph_data = result["graph_data"]
        
        # Create Plotly network graph
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=[],
            y=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0,l=0,r=0,t=0),
                       ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.subheader("📥 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download JSON"):
            st.download_button(
                label="Download",
                data=json.dumps(result, indent=2),
                file_name=f"{st.session_state.current_job['id']}_results.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Download Report"):
            report = f"""
# DIGIMON Analysis Report
**Job ID**: {st.session_state.current_job['id']}
**Date**: {st.session_state.current_job['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
**Query**: {st.session_state.current_job['query']}

## Key Findings
{result.get('answer', 'No findings')}

## Metrics
- Entities: {result.get('entity_count', 0)}
- Relationships: {result.get('relationship_count', 0)}
- Communities: {result.get('community_count', 0)}
"""
            st.download_button(
                label="Download",
                data=report,
                file_name=f"{st.session_state.current_job['id']}_report.md",
                mime="text/markdown"
            )

# History section
with st.expander("📜 Analysis History"):
    if st.session_state.analysis_history:
        history_df = pd.DataFrame([
            {
                "Time": job["start_time"].strftime("%H:%M:%S"),
                "Query": job["query"][:50] + "...",
                "Status": job["status"],
                "Entities": job.get("result", {}).get("entity_count", 0) if job["status"] == "completed" else "-"
            }
            for job in st.session_state.analysis_history
        ])
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No analyses run yet")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>DIGIMON GraphRAG | Built with Streamlit | 
    <a href='https://github.com/yourusername/digimon'>Documentation</a>
    </p>
</div>
""", unsafe_allow_html=True)