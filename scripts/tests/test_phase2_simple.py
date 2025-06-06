#!/usr/bin/env python3
"""
Simple Phase 2 Test - Execute basic discourse analysis without full graph building
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from Core.AgentTools.discourse_enhanced_planner import DiscourseEnhancedPlanner
from social_media_discourse_executor import DiscourseEnhancedSocialMediaExecutor


async def test_phase2_simple():
    """Test basic discourse analysis functionality"""
    print("=== SIMPLE PHASE 2 TEST ===\n")
    
    tests_passed = 0
    
    # Test 1: Generate analysis plan
    print("Test 1: Generate discourse analysis plan...")
    try:
        planner = DiscourseEnhancedPlanner()
        
        question = "Who are the super-spreaders of COVID conspiracy theories?"
        scenarios = planner.generate_scenarios([question], "COVID-19 conspiracy theories")
        
        if scenarios:
            scenario = scenarios[0]
            print(f"✓ Scenario generated: {scenario.title}")
            
            # Show plan components
            if hasattr(scenario, 'mini_ontologies') and scenario.mini_ontologies:
                print(f"  - Mini-ontologies: {list(scenario.mini_ontologies.keys())}")
            if hasattr(scenario, 'retrieval_chains') and scenario.retrieval_chains:
                print(f"  - Retrieval steps: {len(scenario.retrieval_chains)}")
            if hasattr(scenario, 'transformation_chains') and scenario.transformation_chains:
                print(f"  - Transformation steps: {len(scenario.transformation_chains)}")
            
            tests_passed += 1
        else:
            print("✗ Failed to generate scenario")
            
    except Exception as e:
        print(f"✗ Plan generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Test mini-ontology generation
    print("\nTest 2: Test mini-ontology for WHO analysis...")
    try:
        views = planner.generate_discourse_views(
            "Analyze influencers in COVID conspiracy discourse",
            ["Who"]
        )
        
        if views:
            view = views[0]
            ontology = planner.generate_mini_ontology(view)
            
            print("✓ Mini-ontology generated:")
            print(f"  - Entities: {list(ontology['entities'].keys())[:3]}...")
            print(f"  - Relationships: {list(ontology['relationships'].keys())[:3]}...")
            
            tests_passed += 1
        else:
            print("✗ Failed to generate views")
            
    except Exception as e:
        print(f"✗ Ontology generation failed: {e}")
    
    # Test 3: Test retrieval chain generation
    print("\nTest 3: Test retrieval chain generation...")
    try:
        if 'scenario' in locals() and 'view' in locals():
            retrieval_chain = planner.generate_retrieval_chain(scenario, view)
            
            if retrieval_chain:
                print(f"✓ Retrieval chain with {len(retrieval_chain)} steps:")
                for i, step in enumerate(retrieval_chain[:3]):
                    print(f"  Step {i+1}: {step.get('description', 'N/A')}")
                    print(f"    - Operator: {step.get('operator', 'N/A')}")
                
                tests_passed += 1
            else:
                print("✗ Empty retrieval chain")
        else:
            print("✗ Missing scenario or view")
            
    except Exception as e:
        print(f"✗ Retrieval chain failed: {e}")
    
    # Test 4: Test transformation chain
    print("\nTest 4: Test transformation chain generation...")
    try:
        if 'scenario' in locals() and 'view' in locals():
            transform_chain = planner.generate_transformation_chain(scenario, view)
            
            if transform_chain:
                print(f"✓ Transformation chain with {len(transform_chain)} steps:")
                for i, step in enumerate(transform_chain[:2]):
                    print(f"  Step {i+1}: {step.get('description', 'N/A')}")
                    print(f"    - Operator: {step.get('operator', 'N/A')}")
                
                tests_passed += 1
            else:
                print("✗ Empty transformation chain")
        else:
            print("✗ Missing scenario or view")
            
    except Exception as e:
        print(f"✗ Transformation chain failed: {e}")
    
    # Test 5: Save analysis plan
    print("\nTest 5: Save analysis plan for manual inspection...")
    try:
        if 'scenario' in locals():
            plan_data = {
                "title": scenario.title,
                "research_question": scenario.research_question,
                "interrogative_views": [
                    {
                        "interrogative": iv.interrogative,
                        "focus": iv.focus,
                        "entities": iv.entities,
                        "relationships": iv.relationships
                    }
                    for iv in scenario.interrogative_views
                ] if hasattr(scenario, 'interrogative_views') else [],
                "has_mini_ontologies": bool(hasattr(scenario, 'mini_ontologies') and scenario.mini_ontologies),
                "retrieval_steps": len(scenario.retrieval_chains) if hasattr(scenario, 'retrieval_chains') else 0,
                "transformation_steps": len(scenario.transformation_chains) if hasattr(scenario, 'transformation_chains') else 0,
            }
            
            with open("phase2_analysis_plan.json", 'w') as f:
                json.dump(plan_data, f, indent=2)
            
            print("✓ Analysis plan saved to: phase2_analysis_plan.json")
            tests_passed += 1
        else:
            print("✗ No scenario to save")
            
    except Exception as e:
        print(f"✗ Save failed: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Tests passed: {tests_passed}/5")
    print(f"Status: {'READY' if tests_passed >= 3 else 'NOT READY'} for discourse analysis")
    
    return tests_passed >= 3


if __name__ == "__main__":
    success = asyncio.run(test_phase2_simple())
    sys.exit(0 if success else 1)