#!/usr/bin/env python3
"""Demonstration of discourse-enhanced social media analysis"""

import json
from Core.AgentTools.discourse_enhanced_planner import DiscourseEnhancedPlanner

def demonstrate_discourse_analysis():
    """Show how discourse analysis works for COVID conspiracy theories"""
    
    print("=" * 80)
    print("DIGIMON Discourse Analysis Demonstration")
    print("Analyzing COVID-19 Conspiracy Theories on Social Media")
    print("=" * 80)
    
    # Initialize planner
    planner = DiscourseEnhancedPlanner()
    
    # Define research focus
    research_focus = """
    How do COVID-19 conspiracy theories spread through social media platforms, 
    who are the key actors involved, and what measurable effects do these 
    narratives have on public health behavior?
    """
    
    print(f"\nResearch Focus: {research_focus.strip()}")
    print("\nGenerating discourse-based analysis plan...\n")
    
    # Generate comprehensive analysis scenario
    scenarios = planner.generate_scenarios(research_focus, num_scenarios=1)
    scenario = scenarios[0]
    
    print(f"📊 Analysis Scenario: {scenario.title}")
    print(f"📝 Research Question: {scenario.research_question}")
    print(f"🎯 Complexity Level: {scenario.complexity_level}")
    print(f"\n🔍 Interrogative Analysis Framework:")
    print("   Using the five interrogatives of discourse analysis")
    
    # Display each interrogative view
    for i, view in enumerate(scenario.interrogative_views, 1):
        print(f"\n{i}. {view.interrogative}: {view.focus}")
        print("   " + "-" * 60)
        
        # Show ontology
        print(f"   📦 Entity Types: {', '.join(view.entities[:3])}...")
        print(f"   🔗 Relationships: {', '.join(view.relationships[:3])}...")
        print(f"   📊 Properties: {', '.join(view.properties[:3])}...")
        
        # Show analysis approach
        print(f"\n   🎯 Analysis Goals:")
        for goal in view.analysis_goals:
            print(f"      • {goal}")
        
        print(f"\n   🔧 Retrieval Strategy:")
        print(f"      {' → '.join(view.retrieval_operators)}")
        
        print(f"\n   🔄 Transformation Pipeline:")
        print(f"      {' → '.join(view.transformation_operators)}")
    
    # Show cross-interrogative analysis
    print("\n\n🌐 Cross-Interrogative Analysis:")
    print("   The system will identify patterns across views:")
    
    cross_patterns = [
        "• Who (influencers) → Says What (narratives) → To Whom (audiences)",
        "• In What Setting (platforms) → Enables → With What Effect (outcomes)",
        "• Temporal patterns: How narratives evolve and spread over time",
        "• Network effects: How influencer networks amplify conspiracy theories",
        "• Intervention points: Where moderation could be most effective"
    ]
    
    for pattern in cross_patterns:
        print(f"   {pattern}")
    
    # Show expected insights
    print("\n\n💡 Expected Insights from This Analysis:")
    for insight in scenario.expected_insights:
        print(f"   • {insight}")
    
    # Example findings (what the system would discover)
    print("\n\n📈 Example Findings (from COVID conspiracy tweet analysis):")
    example_findings = {
        "Who": [
            "• @conspiracy_theorist_1 (95% influence score) - 50K followers",
            "• @health_skeptic_2025 (89% influence score) - Anti-vaccine community leader",
            "• Coordinated network of 150+ accounts spreading similar narratives"
        ],
        "Says What": [
            "• 'Vaccine contains microchips' - Most viral narrative (10K+ retweets)",
            "• 'Economic control through pandemic' - Links to wealth transfer themes",
            "• Evolution: Simple claims → Complex theories → Call to action"
        ],
        "To Whom": [
            "• Primary audience: Parents concerned about children (65% engagement)",
            "• Secondary: Economic anxiety demographics (25% engagement)",
            "• High receptivity in communities with low institutional trust"
        ],
        "In What Setting": [
            "• Peak spreading hours: 8-10 PM local time",
            "• Platform features exploited: Quote tweets, hashtag hijacking",
            "• Cross-platform coordination: Twitter → Facebook groups"
        ],
        "With What Effect": [
            "• 15% increase in vaccine hesitancy in exposed communities",
            "• Measurable decrease in booster uptake (-12%)",
            "• Radicalization pathway: Vaccine skeptic → General conspiracy believer"
        ]
    }
    
    for interrogative, findings in example_findings.items():
        print(f"\n{interrogative}:")
        for finding in findings:
            print(f"  {finding}")
    
    print("\n\n✅ Summary:")
    print("The discourse analysis framework provides a sophisticated approach to")
    print("understanding conspiracy theory propagation by analyzing:")
    print("- Multiple perspectives (5 interrogatives)")
    print("- Rich ontologies for each view")
    print("- Cross-interrogative patterns")
    print("- Actionable insights for intervention")
    
    print("\nThis approach goes beyond simple sentiment analysis to understand")
    print("the full ecosystem of misinformation spread.")

if __name__ == "__main__":
    demonstrate_discourse_analysis()