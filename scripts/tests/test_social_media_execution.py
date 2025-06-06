#!/usr/bin/env python3
"""Test script for social media analysis execution"""

import asyncio
import json
from pathlib import Path
from social_media_execution import SocialMediaAnalysisExecutor

async def test_simple_execution():
    """Test basic execution functionality"""
    print("Starting social media analysis test...")
    
    # Create executor
    executor = SocialMediaAnalysisExecutor()
    
    # Initialize
    print("Initializing DIGIMON system...")
    success = await executor.initialize()
    if not success:
        print("Failed to initialize!")
        return
    
    print("DIGIMON initialized successfully")
    
    # Simple test scenario
    test_scenarios = [{
        "title": "Simple Influence Test",
        "research_question": "Who are the key influencers spreading COVID conspiracy theories?",
        "complexity_level": "Simple",
        "interrogative_views": [{
            "interrogative": "Who",
            "focus": "Key influencers",
            "description": "Identify influential users",
            "entities": ["User", "Tweet", "Hashtag"],
            "relationships": ["POSTS", "MENTIONS", "USES"]
        }]
    }]
    
    # Dataset info
    dataset_info = {
        "path": "COVID-19-conspiracy-theories-tweets.csv",
        "total_rows": 100  # Limit for testing
    }
    
    print("\nExecuting analysis scenario...")
    results = await executor.execute_all_scenarios(test_scenarios, dataset_info)
    
    print("\n=== ANALYSIS RESULTS ===")
    print(json.dumps(results, indent=2))
    
    # Save results
    output_file = Path("test_analysis_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(test_simple_execution())