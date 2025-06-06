#!/usr/bin/env python3

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append('/home/brian/digimon_cc')

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config

async def test_query():
    """Test the query functionality directly."""
    
    print("🔍 Testing DIGIMON query functionality...")
    
    # Try to create a config for an existing build
    try:
        # Use the existing MySampleTexts build with er_graph
        config_options = Config.parse(
            Path("Option/Method/LGraphRAG.yaml"), 
            dataset_name="MySampleTexts", 
            exp_name="MySampleTexts"  # Use dataset name as exp_name
        )
        
        print(f"✅ Config created successfully")
        print(f"📊 Dataset: {config_options.exp_name}")
        print(f"📁 Working dir: {config_options.working_dir}")
        
        # Create GraphRAG instance
        graphrag_instance = GraphRAG(config=config_options)
        print(f"✅ GraphRAG instance created")
        
        # Try to setup for querying
        setup_result = await graphrag_instance.setup_for_querying()
        if setup_result:
            print(f"✅ Setup for querying successful")
            
            # Try a simple query
            question = "who were some main people in the american revolution"
            print(f"🔍 Testing query: {question}")
            
            answer = await graphrag_instance.query(question)
            print(f"✅ Query successful!")
            print(f"📝 Answer: {answer}")
            
        else:
            print("❌ Setup for querying failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_query())