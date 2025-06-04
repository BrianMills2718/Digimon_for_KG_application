#!/usr/bin/env python3
"""Test script to verify MCP integration plan completeness"""

import os
import sys

def test_mcp_plan_exists():
    """Test that MCP integration plan file exists"""
    plan_path = "MCP_INTEGRATION_PLAN.md"
    assert os.path.exists(plan_path), f"MCP integration plan not found at {plan_path}"
    print("✓ MCP integration plan file exists")
    return plan_path

def test_plan_structure(plan_path):
    """Test that plan contains all required sections"""
    with open(plan_path, 'r') as f:
        content = f.read()
    
    required_sections = [
        "# MCP (Model Context Protocol) Integration Plan",
        "## Executive Summary",
        "## Why MCP is Critical for DIGIMON",
        "## Architecture Overview",
        "## Implementation Milestones",
        "### 📍 **Milestone 1: MCP Foundation**",
        "### 📍 **Milestone 2: Tool Migration**",
        "### 📍 **Milestone 3: Multi-Agent Coordination**",
        "### 📍 **Milestone 4: Performance Optimization**",
        "### 📍 **Milestone 5: Cross-Modal Integration**",
        "### 📍 **Milestone 6: Production Deployment**",
        "## Testing Strategy",
        "## Risk Mitigation",
        "## Success Metrics"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
    
    if missing_sections:
        print("✗ Missing sections:")
        for section in missing_sections:
            print(f"  - {section}")
        return False
    
    print("✓ All required sections present")
    return True

def test_code_examples(plan_path):
    """Test that plan contains code examples"""
    with open(plan_path, 'r') as f:
        content = f.read()
    
    # Check for Python code blocks
    code_block_count = content.count("```python")
    assert code_block_count >= 10, f"Expected at least 10 code examples, found {code_block_count}"
    print(f"✓ Found {code_block_count} code examples")
    
    # Check for key classes
    key_classes = [
        "DigimonMCPServer",
        "MCPClientManager",
        "MCPEnabledAgent",
        "MCPParallelExecutor",
        "CrossModalMCPBridge"
    ]
    
    missing_classes = []
    for class_name in key_classes:
        if f"class {class_name}" not in content:
            missing_classes.append(class_name)
    
    if missing_classes:
        print("✗ Missing key classes:")
        for class_name in missing_classes:
            print(f"  - {class_name}")
        return False
    
    print("✓ All key classes defined")
    return True

def test_milestone_details(plan_path):
    """Test that each milestone has proper details"""
    with open(plan_path, 'r') as f:
        content = f.read()
    
    # Check each milestone has tasks and test criteria
    for i in range(1, 7):
        milestone_section = f"### 📍 **Milestone {i}:"
        if milestone_section not in content:
            print(f"✗ Milestone {i} not found")
            return False
        
        # Find the milestone section
        start_idx = content.find(milestone_section)
        next_milestone = content.find("### 📍 **Milestone", start_idx + 1)
        if next_milestone == -1:
            next_milestone = len(content)
        
        milestone_content = content[start_idx:next_milestone]
        
        # Check for required subsections
        if "#### Tasks:" not in milestone_content:
            print(f"✗ Milestone {i} missing Tasks section")
            return False
        
        if "#### Test Criteria:" not in milestone_content:
            print(f"✗ Milestone {i} missing Test Criteria section")
            return False
        
        # Check for checkboxes in test criteria
        checkbox_count = milestone_content.count("- [ ]")
        if checkbox_count < 3:
            print(f"✗ Milestone {i} has insufficient test criteria ({checkbox_count} items)")
            return False
    
    print("✓ All milestones have proper structure")
    return True

def test_integration_points(plan_path):
    """Test that plan addresses key integration points"""
    with open(plan_path, 'r') as f:
        content = f.read()
    
    integration_keywords = [
        "UKRF",
        "StructGPT",
        "Autocoder",
        "GraphRAG",
        "cross-modal",
        "entity linking",
        "performance",
        "latency",
        "parallel",
        "security"
    ]
    
    missing_keywords = []
    for keyword in integration_keywords:
        if keyword.lower() not in content.lower():
            missing_keywords.append(keyword)
    
    if missing_keywords:
        print("✗ Missing integration keywords:")
        for keyword in missing_keywords:
            print(f"  - {keyword}")
        return False
    
    print("✓ All key integration points addressed")
    return True

def main():
    """Run all tests"""
    print("Testing MCP Integration Plan...")
    print("=" * 50)
    
    try:
        # Test 1: File exists
        plan_path = test_mcp_plan_exists()
        
        # Test 2: Structure
        if not test_plan_structure(plan_path):
            return 1
        
        # Test 3: Code examples
        if not test_code_examples(plan_path):
            return 1
        
        # Test 4: Milestone details
        if not test_milestone_details(plan_path):
            return 1
        
        # Test 5: Integration points
        if not test_integration_points(plan_path):
            return 1
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! MCP integration plan is comprehensive.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())