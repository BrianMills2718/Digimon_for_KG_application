"""Flask API for Social Media Analysis with Full Execution Tracing"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
from pathlib import Path
import tempfile
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import threading
import queue

# Import our social media analysis tools
from Core.AgentTools.social_media_dataset_tools import (
    ingest_covid_conspiracy_dataset,
    DatasetIngestionInput
)
from Core.AgentTools.automated_interrogative_planner import (
    generate_interrogative_analysis_plans,
    AutoInterrogativePlanInput
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store dataset info and execution traces
dataset_cache = {}
execution_traces = {}
trace_queue = queue.Queue()

@app.route('/api/ingest-dataset', methods=['POST'])
def ingest_dataset():
    """Ingest dataset from Hugging Face or local CSV"""
    try:
        data = request.json
        dataset_name = data.get('dataset_name', 'webimmunization/COVID-19-conspiracy-theories-tweets')
        max_rows = data.get('max_rows')
        source_type = data.get('source_type', 'huggingface')
        
        # Check if local file exists
        if source_type == 'auto':
            if Path(dataset_name).exists() and dataset_name.endswith('.csv'):
                source_type = 'local'
            else:
                source_type = 'huggingface'
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / 'dataset')
            
            # Create input for tool
            tool_input = DatasetIngestionInput(
                dataset_name=dataset_name,
                split='train',
                output_path=output_path,
                max_rows=max_rows if max_rows else None,
                source_type=source_type
            )
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(ingest_covid_conspiracy_dataset(tool_input))
            
            if result.success:
                # Cache dataset info
                dataset_cache['current'] = {
                    'total_rows': result.total_rows,
                    'conspiracy_types': list(result.schema_info.get('conspiracy_types', {}).keys()),
                    'label_distribution': result.schema_info.get('label_distribution', {}),
                    'columns': result.schema_info.get('columns', []),
                    'sample_data': result.sample_data[:5]  # First 5 samples
                }
                
                return jsonify({
                    'success': True,
                    **dataset_cache['current']
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to ingest dataset'
                }), 500
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/generate-plan', methods=['POST'])
def generate_plan():
    """Generate analysis plan using interrogative planner"""
    try:
        data = request.json
        
        # Create input for planner
        tool_input = AutoInterrogativePlanInput(
            domain=data.get('domain', 'COVID-19 conspiracy theories on Twitter'),
            dataset_info=data.get('dataset_info', dataset_cache.get('current', {})),
            num_scenarios=data.get('num_scenarios', 5),
            complexity_range=data.get('complexity_range', ['Simple', 'Medium'])
        )
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(generate_interrogative_analysis_plans(tool_input))
        
        if result.success:
            # Convert scenarios to dict format
            scenarios_data = []
            for scenario in result.scenarios:
                scenario_dict = {
                    'title': scenario.title,
                    'research_question': scenario.research_question,
                    'complexity_level': scenario.complexity_level,
                    'interrogative_views': [
                        {
                            'interrogative': view.interrogative,
                            'focus': view.focus,
                            'description': view.description,
                            'entities': view.entities,
                            'relationships': view.relationships,
                            'analysis_goals': view.analysis_goals
                        }
                        for view in scenario.interrogative_views
                    ],
                    'analysis_pipeline': scenario.analysis_pipeline,
                    'expected_insights': scenario.expected_insights
                }
                scenarios_data.append(scenario_dict)
            
            return jsonify({
                'success': True,
                'scenarios': scenarios_data,
                'execution_order': result.execution_order,
                'estimated_complexity': result.estimated_complexity
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate analysis plan'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/execute-analysis', methods=['POST'])
def execute_analysis():
    """Execute analysis scenarios with full tracing"""
    try:
        data = request.json
        scenarios = data.get('scenarios', [])
        
        # Create job ID
        job_id = f"job_{int(time.time())}"
        
        # Initialize trace
        execution_traces[job_id] = {
            'status': 'starting',
            'started': datetime.now().isoformat(),
            'scenarios': scenarios,
            'progress': 0,
            'execution_plan': None,
            'execution_steps': [],
            'results': None,
            'error': None
        }
        
        # Import execution engine - try full version first, fall back to simplified
        executor_module = None
        try:
            from social_media_execution_traced import TracedSocialMediaAnalysisExecutor
            executor_module = TracedSocialMediaAnalysisExecutor
            print("Using full TracedSocialMediaAnalysisExecutor")
        except Exception as e:
            print(f"Warning: Could not load full executor ({e}), trying simplified version...")
            try:
                from social_media_execution_simple import SimplifiedSocialMediaAnalysisExecutor
                executor_module = SimplifiedSocialMediaAnalysisExecutor
                print("Using SimplifiedSocialMediaAnalysisExecutor")
            except Exception as e2:
                print(f"Error: Could not load any executor: {e2}")
                execution_traces[job_id]['error'] = f"Failed to load execution engine: {str(e)}"
                execution_traces[job_id]['status'] = 'failed'
                return jsonify({
                    'success': False,
                    'job_id': job_id,
                    'error': 'Execution engine not available'
                }), 500
        
        # Get dataset info from cache
        dataset_info = dataset_cache.get('current', {})
        dataset_info['path'] = data.get('dataset_path', 'COVID-19-conspiracy-theories-tweets.csv')
        
        # Run analysis in background with tracing
        def run_traced_analysis():
            try:
                # Create executor with trace callback
                def trace_callback(event_type, event_data):
                    trace_event = {
                        'timestamp': datetime.now().isoformat(),
                        'type': event_type,
                        'data': event_data
                    }
                    execution_traces[job_id]['execution_steps'].append(trace_event)
                    
                    # Update progress
                    if event_type == 'progress':
                        execution_traces[job_id]['progress'] = event_data.get('percent', 0)
                    elif event_type == 'execution_plan':
                        execution_traces[job_id]['execution_plan'] = event_data
                    elif event_type == 'error':
                        execution_traces[job_id]['error'] = event_data
                        execution_traces[job_id]['status'] = 'failed'
                
                # Try to create executor - handle config errors gracefully
                try:
                    executor = executor_module(trace_callback=trace_callback)
                except Exception as config_error:
                    print(f"Config error, using simplified executor: {config_error}")
                    # Force simplified executor if config fails
                    from social_media_execution_simple import SimplifiedSocialMediaAnalysisExecutor
                    executor = SimplifiedSocialMediaAnalysisExecutor(trace_callback=trace_callback)
                
                # Update status
                execution_traces[job_id]['status'] = 'initializing'
                
                # Run analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(
                    executor.execute_all_scenarios(scenarios, dataset_info)
                )
                
                # Store results
                execution_traces[job_id]['results'] = results
                execution_traces[job_id]['status'] = 'completed'
                execution_traces[job_id]['completed'] = datetime.now().isoformat()
                execution_traces[job_id]['progress'] = 100
                
            except Exception as e:
                execution_traces[job_id]['error'] = str(e)
                execution_traces[job_id]['status'] = 'failed'
                import traceback
                execution_traces[job_id]['traceback'] = traceback.format_exc()
        
        thread = threading.Thread(target=run_traced_analysis)
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'Analysis execution started for {len(scenarios)} scenarios',
            'status': 'in_progress',
            'trace_url': f'/api/execution-trace/{job_id}'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/execution-trace/<job_id>', methods=['GET'])
def get_execution_trace(job_id):
    """Get detailed execution trace for a job"""
    try:
        if job_id not in execution_traces:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404
        
        trace = execution_traces[job_id]
        
        # Include summary of execution steps
        step_summary = {}
        for step in trace.get('execution_steps', []):
            step_type = step['type']
            step_summary[step_type] = step_summary.get(step_type, 0) + 1
        
        return jsonify({
            'job_id': job_id,
            'status': trace['status'],
            'progress': trace['progress'],
            'started': trace['started'],
            'completed': trace.get('completed'),
            'execution_plan': trace.get('execution_plan'),
            'step_count': len(trace.get('execution_steps', [])),
            'step_summary': step_summary,
            'recent_steps': trace.get('execution_steps', [])[-10:],  # Last 10 steps
            'results': trace.get('results'),
            'error': trace.get('error'),
            'traceback': trace.get('traceback')
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/execution-trace/<job_id>/full', methods=['GET'])
def get_full_execution_trace(job_id):
    """Get complete execution trace with all steps"""
    try:
        if job_id not in execution_traces:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404
        
        return jsonify(execution_traces[job_id]), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analysis-status', methods=['GET'])
def get_analysis_status():
    """Get the status of all analysis jobs"""
    try:
        jobs_summary = []
        for job_id, trace in execution_traces.items():
            jobs_summary.append({
                'job_id': job_id,
                'status': trace['status'],
                'progress': trace['progress'],
                'started': trace['started'],
                'scenario_count': len(trace.get('scenarios', []))
            })
        
        return jsonify({
            'jobs': jobs_summary,
            'total_jobs': len(jobs_summary)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'DIGIMON Social Media Analysis API (Traced)',
        'version': '2.0'
    }), 200

if __name__ == '__main__':
    print("Starting DIGIMON Social Media Analysis API with Execution Tracing...")
    print("This version provides full execution traces and real analysis")
    print("Access the UI by opening social_media_analysis_ui.html in your browser")
    print("Make sure to configure Option/Config2.yaml first!")
    app.run(host='0.0.0.0', port=5000, debug=True)