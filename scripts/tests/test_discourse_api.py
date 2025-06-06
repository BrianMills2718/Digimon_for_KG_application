#!/usr/bin/env python3
"""Test the discourse-enhanced API endpoints"""

import requests
import json
import time

API_BASE = "http://localhost:5000"

def test_discourse_plan_generation():
    """Test generating a plan with discourse framework"""
    print("Testing discourse-enhanced plan generation...")
    
    # First, ensure dataset is loaded
    print("1. Loading dataset...")
    response = requests.post(f"{API_BASE}/api/ingest-dataset", json={
        "dataset_name": "COVID-19-conspiracy-theories-tweets.csv",
        "source_type": "local"
    })
    if response.status_code == 200:
        print("   ✓ Dataset loaded successfully")
    else:
        print(f"   ✗ Failed to load dataset: {response.text}")
        return
    
    # Generate plan with discourse framework
    print("\n2. Generating discourse-enhanced plan...")
    response = requests.post(f"{API_BASE}/api/generate-plan", json={
        "domain": "COVID-19 conspiracy theories on Twitter",
        "num_scenarios": 3,
        "use_discourse_framework": True,
        "research_focus": "How do conspiracy theories spread through social networks and what psychological effects do they have on different audiences?"
    })
    
    if response.status_code == 200:
        data = response.json()
        print("   ✓ Plan generated successfully")
        print(f"   - Scenarios: {len(data.get('scenarios', []))}")
        
        # Display first scenario
        if data.get('scenarios'):
            scenario = data['scenarios'][0]
            print(f"\n   First Scenario: {scenario.get('title', 'N/A')}")
            print(f"   Research Question: {scenario.get('research_question', 'N/A')}")
            print(f"   Interrogative Views: {len(scenario.get('interrogative_views', []))}")
            
            # Show mini-ontology for first view
            if scenario.get('interrogative_views'):
                view = scenario['interrogative_views'][0]
                print(f"\n   First View - {view.get('interrogative', 'N/A')}: {view.get('focus', 'N/A')}")
                print(f"   Entities: {', '.join(view.get('entities', []))}")
                print(f"   Relationships: {', '.join(view.get('relationships', []))}")
        
        return data
    else:
        print(f"   ✗ Failed to generate plan: {response.text}")
        return None

def test_discourse_execution(scenarios):
    """Test executing analysis with discourse framework"""
    print("\n3. Executing discourse-enhanced analysis...")
    
    response = requests.post(f"{API_BASE}/api/execute-analysis", json={
        "scenarios": scenarios,
        "dataset_path": "COVID-19-conspiracy-theories-tweets.csv",
        "use_discourse_framework": True
    })
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success') and data.get('job_id'):
            print(f"   ✓ Analysis started: Job ID = {data['job_id']}")
            
            # Poll for results
            job_id = data['job_id']
            print("\n   Monitoring execution...")
            
            for i in range(30):  # Poll for up to 60 seconds
                time.sleep(2)
                trace_response = requests.get(f"{API_BASE}/api/execution-trace/{job_id}")
                if trace_response.status_code == 200:
                    trace = trace_response.json()
                    status = trace.get('status', 'unknown')
                    progress = trace.get('progress', 0)
                    
                    print(f"   Status: {status} - Progress: {progress}%", end='\r')
                    
                    if status == 'completed':
                        print(f"\n   ✓ Analysis completed!")
                        if trace.get('results'):
                            print(f"   - Total insights: {trace['results'].get('execution_summary', {}).get('total_insights_generated', 0)}")
                            print(f"   - Entities found: {trace['results'].get('execution_summary', {}).get('total_entities_found', 0)}")
                        break
                    elif status == 'failed':
                        print(f"\n   ✗ Analysis failed: {trace.get('error', 'Unknown error')}")
                        break
                
                if i == 29:
                    print("\n   ⚠ Timeout waiting for analysis to complete")
        else:
            print(f"   ✗ Failed to start analysis: {data}")
    else:
        print(f"   ✗ Failed to execute analysis: {response.text}")

def main():
    print("DIGIMON Discourse-Enhanced API Test")
    print("===================================\n")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE}/api/health")
        if response.status_code != 200:
            print("API is not running. Start it with: python social_media_api_traced.py")
            return
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API. Start it with: python social_media_api_traced.py")
        return
    
    # Test discourse framework
    plan_data = test_discourse_plan_generation()
    
    if plan_data and plan_data.get('scenarios'):
        # Execute with first scenario only for testing
        test_discourse_execution(plan_data['scenarios'][:1])
    
    print("\n\nTest complete!")

if __name__ == "__main__":
    main()