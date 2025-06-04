"""Simplified Flask API for Social Media Analysis Tools - Working Version"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
from pathlib import Path
import json
import time
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

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

# Store dataset info in memory
dataset_cache = {}
analysis_jobs = {}  # Store analysis job status

@app.route('/api/ingest-dataset', methods=['POST'])
def ingest_dataset():
    """Ingest dataset from Hugging Face or local CSV"""
    try:
        data = request.json
        dataset_name = data.get('dataset_name', 'webimmunization/COVID-19-conspiracy-theories-tweets')
        max_rows = data.get('max_rows')
        source_type = data.get('source_type', 'huggingface')
        
        # For quick demo, just load CSV directly
        if source_type == 'local' and Path(dataset_name).exists():
            df = pd.read_csv(dataset_name, nrows=max_rows if max_rows else None)
            
            # Cache dataset info
            dataset_cache['current'] = {
                'total_rows': len(df),
                'conspiracy_types': ['CT_1', 'CT_2', 'CT_3', 'CT_4', 'CT_5', 'CT_6'],
                'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {},
                'columns': list(df.columns),
                'sample_data': df.head(5).to_dict('records')
            }
            
            return jsonify({
                'success': True,
                **dataset_cache['current']
            }), 200
        else:
            # Use the tool for Hugging Face datasets
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = str(Path(temp_dir) / 'dataset')
                
                tool_input = DatasetIngestionInput(
                    dataset_name=dataset_name,
                    split='train',
                    output_path=output_path,
                    max_rows=max_rows if max_rows else None,
                    source_type=source_type
                )
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(ingest_covid_conspiracy_dataset(tool_input))
                
                if result.success:
                    dataset_cache['current'] = {
                        'total_rows': result.total_rows,
                        'conspiracy_types': list(result.schema_info.get('conspiracy_types', {}).keys()),
                        'label_distribution': result.schema_info.get('label_distribution', {}),
                        'columns': result.schema_info.get('columns', []),
                        'sample_data': result.sample_data[:5]
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
        print(f"Error in ingest_dataset: {str(e)}")
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
        print(f"Error in generate_plan: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/execute-analysis', methods=['POST'])
def execute_analysis():
    """Execute analysis with simulated results"""
    try:
        data = request.json
        scenarios = data.get('scenarios', [])
        
        # Create a job ID
        job_id = f"job_{int(time.time())}"
        
        # Store job info
        analysis_jobs[job_id] = {
            'status': 'running',
            'started': datetime.now().isoformat(),
            'scenarios': scenarios,
            'progress': 0,
            'results': None
        }
        
        # Simulate analysis in background
        import threading
        
        def simulate_analysis():
            # Simulate processing each scenario
            results = {
                'execution_id': job_id,
                'dataset': 'covid_conspiracy_tweets',
                'total_scenarios': len(scenarios),
                'scenario_results': []
            }
            
            for i, scenario in enumerate(scenarios):
                # Update progress
                analysis_jobs[job_id]['progress'] = int((i / len(scenarios)) * 100)
                
                # Simulate some processing time
                time.sleep(2)
                
                # Generate simulated results
                scenario_result = {
                    'scenario': scenario['title'],
                    'research_question': scenario['research_question'],
                    'timestamp': datetime.now().isoformat(),
                    'insights': []
                }
                
                # Add insights based on interrogative views
                for view in scenario.get('interrogative_views', []):
                    if view['interrogative'] == 'Who':
                        scenario_result['insights'].append({
                            'interrogative': 'Who',
                            'focus': view['focus'],
                            'type': 'entities',
                            'findings': [
                                {'entity': 'User_12345', 'score': 0.95, 'description': 'High-influence account spreading CT_3'},
                                {'entity': 'User_67890', 'score': 0.89, 'description': 'Key amplifier of vaccine conspiracies'},
                                {'entity': 'Bot_Network_Alpha', 'score': 0.87, 'description': 'Coordinated bot network'}
                            ]
                        })
                    elif view['interrogative'] == 'What':
                        scenario_result['insights'].append({
                            'interrogative': 'What',
                            'focus': view['focus'],
                            'type': 'narratives',
                            'findings': [
                                {'narrative': 'Bioweapon Origin', 'frequency': 2341, 'sentiment': 'negative'},
                                {'narrative': 'Population Control', 'frequency': 1876, 'sentiment': 'fearful'},
                                {'narrative': '5G Connection', 'frequency': 1234, 'sentiment': 'conspiratorial'}
                            ]
                        })
                    elif view['interrogative'] == 'How':
                        scenario_result['insights'].append({
                            'interrogative': 'How',
                            'focus': view['focus'],
                            'type': 'mechanisms',
                            'findings': [
                                {'mechanism': 'Hashtag Amplification', 'effectiveness': 0.78},
                                {'mechanism': 'Emotional Appeals', 'effectiveness': 0.82},
                                {'mechanism': 'False Authority Claims', 'effectiveness': 0.65}
                            ]
                        })
                
                # Add metrics
                scenario_result['metrics'] = {
                    'entities_found': 156,
                    'relationships_found': 423,
                    'clusters_identified': 8,
                    'processing_time': 2.3
                }
                
                results['scenario_results'].append(scenario_result)
            
            # Mark as complete
            analysis_jobs[job_id]['status'] = 'completed'
            analysis_jobs[job_id]['progress'] = 100
            analysis_jobs[job_id]['results'] = results
            analysis_jobs[job_id]['completed'] = datetime.now().isoformat()
            
            # Also store in cache for backward compatibility
            dataset_cache['analysis_results'] = results
        
        # Start background thread
        thread = threading.Thread(target=simulate_analysis)
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'Analysis execution started for {len(scenarios)} scenarios',
            'status': 'in_progress',
            'check_status_url': f'/api/analysis-status/{job_id}'
        }), 200
        
    except Exception as e:
        print(f"Error in execute_analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analysis-status', methods=['GET'])
@app.route('/api/analysis-status/<job_id>', methods=['GET'])
def get_analysis_status(job_id=None):
    """Get the status of analysis job"""
    try:
        # Support both old and new API
        if job_id and job_id in analysis_jobs:
            job = analysis_jobs[job_id]
            return jsonify({
                'job_id': job_id,
                'status': job['status'],
                'progress': job['progress'],
                'started': job['started'],
                'completed': job.get('completed'),
                'results': job.get('results')
            }), 200
        else:
            # Fallback to old API
            results = dataset_cache.get('analysis_results')
            if results:
                return jsonify({
                    'status': 'completed',
                    'results': results
                }), 200
            else:
                return jsonify({
                    'status': 'in_progress',
                    'message': 'Analysis is still running...'
                }), 200
            
    except Exception as e:
        print(f"Error in get_analysis_status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'DIGIMON Social Media Analysis API (Simplified)',
        'version': '1.0'
    }), 200

if __name__ == '__main__':
    print("Starting DIGIMON Social Media Analysis API (Simplified)...")
    print("This version provides simulated results for demo purposes")
    print("Access the UI by opening social_media_analysis_ui.html in your browser")
    print("Make sure to run this API server first!")
    app.run(host='0.0.0.0', port=5000, debug=True)