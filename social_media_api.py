"""Flask API for Social Media Analysis Tools"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
from pathlib import Path
import tempfile
from typing import Dict, Any

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

# Store dataset info in memory (in production, use proper storage)
dataset_cache = {}

@app.route('/api/ingest-dataset', methods=['POST'])
def ingest_dataset():
    """Ingest dataset from Hugging Face"""
    try:
        data = request.json
        dataset_name = data.get('dataset_name', 'webimmunization/COVID-19-conspiracy-theories-tweets')
        max_rows = data.get('max_rows')
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / 'dataset')
            
            # Create input for tool
            tool_input = DatasetIngestionInput(
                dataset_name=dataset_name,
                split='train',
                output_path=output_path,
                max_rows=max_rows if max_rows else None
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
    """Execute analysis scenarios (placeholder for now)"""
    try:
        data = request.json
        scenarios = data.get('scenarios', [])
        
        # TODO: Implement actual execution using DIGIMON's graph building and analysis tools
        # For now, return a placeholder response
        
        return jsonify({
            'success': True,
            'message': f'Analysis execution started for {len(scenarios)} scenarios',
            'status': 'in_progress'
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
        'service': 'DIGIMON Social Media Analysis API'
    }), 200

if __name__ == '__main__':
    print("Starting DIGIMON Social Media Analysis API...")
    print("Access the UI by opening social_media_analysis_ui.html in your browser")
    print("Make sure to run this API server first!")
    app.run(host='0.0.0.0', port=5000, debug=True)