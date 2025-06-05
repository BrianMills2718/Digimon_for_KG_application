#!/usr/bin/env python3
"""
Quick start web interface for DIGIMON - minimal setup required
"""

import gradio as gr
import asyncio
import json
from pathlib import Path
from datetime import datetime

from Core.AgentOrchestrator.memory_enhanced_orchestrator import MemoryEnhancedOrchestrator
from Core.AgentBrain.agent_brain import AgentBrain
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Config.LLMConfig import LLMConfig


class DigimonWebInterface:
    def __init__(self):
        # Initialize DIGIMON components
        config_path = Path("Option/Config2.yaml")
        llm_config = LLMConfig.from_yaml(str(config_path))
        llm = LiteLLMProvider(llm_config)
        brain = AgentBrain(llm)
        self.orchestrator = MemoryEnhancedOrchestrator(brain=brain, llm=llm)
        self.available_corpora = self._find_available_corpora()
    
    def _find_available_corpora(self):
        """Find all available corpus files"""
        corpora = {}
        data_dir = Path("Data")
        
        for corpus_file in data_dir.rglob("Corpus.json"):
            name = corpus_file.parent.name
            corpora[name] = str(corpus_file)
        
        return corpora
    
    def analyze(self, corpus_name, query, analysis_type):
        """Run analysis synchronously for Gradio"""
        if not corpus_name or not query:
            return "Please select a corpus and enter a query", None
        
        corpus_path = self.available_corpora.get(corpus_name)
        if not corpus_path:
            return f"Corpus '{corpus_name}' not found", None
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                self.orchestrator.process_query(
                    query=query,
                    context={
                        "corpus_path": corpus_path,
                        "analysis_type": analysis_type,
                        "web_interface": True
                    }
                )
            )
            
            # Format results
            answer = result.get("answer", "No results generated")
            
            # Create metrics summary
            metrics = f"""
### Analysis Metrics
- **Entities Found**: {result.get('entity_count', 0)}
- **Relationships Discovered**: {result.get('relationship_count', 0)}
- **Communities Detected**: {result.get('community_count', 0)}
- **Execution Time**: {result.get('execution_time', 0):.2f} seconds
"""
            
            return answer, metrics
            
        except Exception as e:
            return f"Error: {str(e)}", None
        finally:
            loop.close()


def create_interface():
    """Create Gradio interface"""
    digimon = DigimonWebInterface()
    
    # Preset queries for different analysis types
    preset_queries = {
        "Influence Network": "Identify the top influencers and analyze their network characteristics",
        "Narrative Analysis": "Extract and analyze the main narratives and how they spread",
        "Community Detection": "Detect and characterize different discourse communities",
        "Temporal Evolution": "Analyze how discussions evolve over time",
        "Custom": ""
    }
    
    def update_query(analysis_type):
        return preset_queries.get(analysis_type, "")
    
    with gr.Blocks(title="DIGIMON GraphRAG") as interface:
        gr.Markdown("""
        # 🔬 DIGIMON GraphRAG Web Interface
        
        **Powerful discourse analysis for social media and text corpora**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                corpus_dropdown = gr.Dropdown(
                    choices=list(digimon.available_corpora.keys()),
                    label="Select Dataset",
                    value=list(digimon.available_corpora.keys())[0] if digimon.available_corpora else None
                )
                
                analysis_type = gr.Radio(
                    choices=list(preset_queries.keys()),
                    label="Analysis Type",
                    value="Custom"
                )
                
                query_input = gr.Textbox(
                    label="Analysis Query",
                    placeholder="Enter your analysis query or select a preset above...",
                    lines=4
                )
                
                # Update query when analysis type changes
                analysis_type.change(
                    fn=update_query,
                    inputs=[analysis_type],
                    outputs=[query_input]
                )
                
                analyze_btn = gr.Button("🚀 Run Analysis", variant="primary")
            
            with gr.Column(scale=2):
                # Results area
                gr.Markdown("## 📊 Analysis Results")
                
                results_output = gr.Markdown(
                    value="*Results will appear here after running analysis...*"
                )
                
                metrics_output = gr.Markdown(
                    value=""
                )
        
        # Examples
        gr.Examples(
            examples=[
                ["COVID_Conspiracy", "Who are the main spreaders of misinformation and what narratives do they promote?"],
                ["COVID_Conspiracy", "How do conspiracy theories evolve and mutate as they spread through the network?"],
                ["COVID_Conspiracy", "What platform features facilitate the spread of misinformation?"],
            ],
            inputs=[corpus_dropdown, query_input],
            label="Example Queries"
        )
        
        # Connect the analyze function
        analyze_btn.click(
            fn=digimon.analyze,
            inputs=[corpus_dropdown, query_input, analysis_type],
            outputs=[results_output, metrics_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 🛠️ Quick Start Guide
        1. Select a dataset from the dropdown
        2. Choose an analysis type or write a custom query
        3. Click "Run Analysis" and wait for results
        
        ### 📚 Available Datasets
        - **COVID_Conspiracy**: 6,590 tweets about COVID-19 conspiracy theories
        - Upload your own corpus by placing a `Corpus.json` file in the `Data/` directory
        """)
    
    return interface


if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates a public link
        inbrowser=True
    )