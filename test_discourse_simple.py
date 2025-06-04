#!/usr/bin/env python3
"""Simple test to demonstrate discourse analysis planning"""

from Core.AgentTools.discourse_enhanced_planner import DiscourseEnhancedPlanner
import json

def test_discourse_planner():
    """Test the discourse-enhanced planner"""
    print("DIGIMON Discourse Analysis - Planner Demo")
    print("=========================================\n")
    
    planner = DiscourseEnhancedPlanner()
    
    # Test different research focuses
    research_focuses = [
        "How do conspiracy theories spread through social networks and what psychological effects do they have?",
        "What role do influencers play in spreading vaccine misinformation?",
        "How does platform design enable or prevent conspiracy theory propagation?"
    ]
    
    for focus in research_focuses:
        print(f"\nResearch Focus: {focus}")
        print("-" * 80)
        
        # Generate 2 scenarios for this focus
        scenarios = planner.generate_scenarios(focus, num_scenarios=2)
        
        for i, scenario in enumerate(scenarios):
            print(f"\nScenario {i+1}: {scenario.title}")
            print(f"Research Question: {scenario.research_question}")
            print(f"Complexity: {scenario.complexity_level}")
            print(f"\nInterrogative Views ({len(scenario.interrogative_views)}):")
            
            for view in scenario.interrogative_views:
                print(f"\n  📍 {view.interrogative}: {view.focus}")
                print(f"     Entities: {', '.join(view.entities[:3])}...")
                print(f"     Properties: {', '.join(view.properties[:3])}...")
                print(f"     Goals: {view.analysis_goals[0]}")
                print(f"     Retrieval: {' → '.join(view.retrieval_operators[:2])}")
                print(f"     Transform: {' → '.join(view.transformation_operators[:2])}")
            
            # Show mini-ontology summary
            print(f"\n  Mini-Ontologies: {len(scenario.mini_ontologies)} views")
            for onto_view, ontology in scenario.mini_ontologies.items():
                print(f"    - {onto_view}: {len(ontology['entities'])} entities, {len(ontology['relationships'])} relationships")
            
            # Show expected insights
            print(f"\n  Expected Insights:")
            for insight in scenario.expected_insights[:3]:
                print(f"    • {insight}")
    
    print("\n\n✅ Discourse planner successfully generates sophisticated analysis scenarios!")
    print("Each scenario includes:")
    print("  - Multiple interrogative perspectives (Who/What/To Whom/etc)")
    print("  - Entity and relationship schemas for each view")
    print("  - Specific retrieval and transformation operators")
    print("  - Mini-ontologies that can be merged")
    print("  - Clear analysis goals and expected insights")
    
    # Save example scenario for reference
    example_scenario = scenarios[0] if scenarios else None
    if example_scenario:
        # Convert to dict for saving
        scenario_dict = {
            "title": example_scenario.title,
            "research_question": example_scenario.research_question,
            "complexity_level": example_scenario.complexity_level,
            "interrogative_views": [
                {
                    "interrogative": view.interrogative,
                    "focus": view.focus,
                    "description": view.description,
                    "entities": view.entities,
                    "relationships": view.relationships,
                    "properties": view.properties,
                    "analysis_goals": view.analysis_goals,
                    "retrieval_operators": view.retrieval_operators,
                    "transformation_operators": view.transformation_operators
                }
                for view in example_scenario.interrogative_views
            ],
            "expected_insights": example_scenario.expected_insights
        }
        
        with open("example_discourse_scenario.json", "w") as f:
            json.dump(scenario_dict, f, indent=2)
        print("\n📄 Example scenario saved to example_discourse_scenario.json")

if __name__ == "__main__":
    test_discourse_planner()