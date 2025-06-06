#!/usr/bin/env python3
"""
Build Fictional Test Dataset
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append('/home/brian/digimon_cc')

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config

async def build_fictional_test():
    """Build the fictional test dataset"""
    print("🔨 Building Fictional Test Dataset")
    print("=" * 50)
    
    os.chdir('/home/brian/digimon_cc')
    
    # Check fictional data exists
    fictional_path = Path("Data/Fictional_Test")
    if not fictional_path.exists():
        print(f"❌ Fictional test data not found at: {fictional_path}")
        return False
    
    files = list(fictional_path.glob("*.txt"))
    print(f"📁 Found {len(files)} files: {[f.name for f in files]}")
    
    try:
        # Create config
        config_options = Config.parse(
            Path("Option/Method/LGraphRAG.yaml"),
            dataset_name="Fictional_Test",
            exp_name="Fictional_Test"
        )
        
        print(f"✅ Config created for Fictional_Test")
        
        # Create GraphRAG instance
        graphrag_instance = GraphRAG(config=config_options)
        print(f"✅ GraphRAG instance created")
        
        # Build artifacts
        print(f"🔨 Starting build process...")
        build_result = await graphrag_instance.build_and_persist_artifacts(str(fictional_path))
        
        if isinstance(build_result, dict) and "error" in build_result:
            print(f"❌ Build failed: {build_result['error']}")
            return False
        
        print(f"✅ Build completed successfully")
        
        # Test setup for querying
        setup_success = await graphrag_instance.setup_for_querying()
        if not setup_success:
            print("❌ Setup for querying failed")
            return False
        
        print(f"✅ Setup for querying successful")
        
        # Test a simple query
        test_query = "What is crystal technology?"
        print(f"🔍 Testing query: {test_query}")
        
        answer = await graphrag_instance.query(test_query)
        if answer and len(str(answer).strip()) > 0:
            print(f"✅ Query successful!")
            print(f"📝 Answer preview: {str(answer)[:300]}...")
            return True
        else:
            print(f"❌ Query returned empty answer")
            return False
            
    except Exception as e:
        print(f"❌ Build failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(build_fictional_test())
    if success:
        print(f"\n🎉 Fictional test dataset built and tested successfully!")
    else:
        print(f"\n❌ Failed to build fictional test dataset")