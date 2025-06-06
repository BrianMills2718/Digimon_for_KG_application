"""Minimal test to verify GraphRAGContext initialization"""

from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config

try:
    # Load config
    config = Config.from_yaml_file("Option/Config2.yaml")
    print("✓ Config loaded successfully")
    
    # Create context with correct field names
    context = GraphRAGContext(
        target_dataset_name="test_dataset",
        main_config=config
    )
    print("✓ Context created successfully")
    print(f"  - Dataset: {context.target_dataset_name}")
    print(f"  - Config type: {type(context.main_config)}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()