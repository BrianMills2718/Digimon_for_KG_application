from typing import Dict, List, Any, Optional, Tuple
import json
import random
from pydantic import BaseModel, Field
from Core.AgentSchema.tool_contracts import BaseToolParams, BaseToolOutput
from Core.Common.Logger import logger

class InterrogativeView(BaseModel):
    """Represents an interrogative perspective for analysis"""
    interrogative: str = Field(description="Who/What/When/Where/Why/How")
    focus: str = Field(description="Specific focus of the interrogative")
    description: str = Field(description="Detailed description of what this view analyzes")
    entities: List[str] = Field(description="Key entity types for this view")
    relationships: List[str] = Field(description="Key relationship types")
    analysis_goals: List[str] = Field(description="What we want to achieve with this view")

class AnalysisScenario(BaseModel):
    """Represents a complete analysis scenario"""
    title: str = Field(description="Scenario title")
    research_question: str = Field(description="Main research question")
    interrogative_views: List[InterrogativeView] = Field(description="Multiple interrogative perspectives")
    analysis_pipeline: List[str] = Field(description="Ordered steps for analysis")
    expected_insights: List[str] = Field(description="What insights we expect to gain")
    complexity_level: str = Field(description="Simple/Medium/Complex")

class AutoInterrogativePlanInput(BaseToolParams):
    """Input for automated interrogative planning"""
    domain: str = Field(description="Analysis domain (e.g., 'social media conspiracy theories')")
    dataset_info: Dict[str, Any] = Field(description="Information about available dataset")
    num_scenarios: int = Field(default=5, description="Number of analysis scenarios to generate")
    complexity_range: List[str] = Field(default=["Simple", "Medium", "Complex"], description="Complexity levels to include")
    focus_areas: Optional[List[str]] = Field(default=None, description="Specific areas to focus on")

class AutoInterrogativePlanOutput(BaseToolOutput):
    """Output from automated interrogative planning"""
    success: bool = Field(description="Whether planning succeeded")
    scenarios: List[AnalysisScenario] = Field(description="Generated analysis scenarios")
    execution_order: List[str] = Field(description="Recommended execution order")
    estimated_complexity: Dict[str, int] = Field(description="Complexity estimates for each scenario")

class AutomatedInterrogativePlanner:
    """Generates diverse analysis scenarios automatically"""
    
    def __init__(self):
        self.logger = logger
        
        # Predefined interrogative templates for social media analysis
        self.interrogative_templates = {
            "Who": [
                "Who are the key influencers in {domain}?",
                "Who spreads misinformation most effectively?", 
                "Who forms the core communities around conspiracy theories?",
                "Who are the bridge users connecting different conspiracy communities?",
                "Who changes their stance over time?"
            ],
            "What": [
                "What narratives dominate {domain}?",
                "What arguments are most persuasive?",
                "What content gets the most engagement?",
                "What patterns emerge in conspiracy theory evolution?",
                "What linguistic markers indicate conspiracy thinking?"
            ],
            "When": [
                "When do conspiracy theories peak in popularity?",
                "When do users become radicalized?",
                "When do narratives shift or evolve?",
                "When is misinformation most likely to spread?"
            ],
            "Where": [
                "Where do conspiracy theories originate?",
                "Where do they spread most rapidly?",
                "Where are the geographic hotspots?",
                "Where do different communities intersect?"
            ],
            "Why": [
                "Why do certain conspiracy theories gain traction?",
                "Why do people share misinformation?",
                "Why do some narratives persist while others fade?",
                "Why do users adopt conspiracy thinking?"
            ],
            "How": [
                "How do conspiracy theories spread through networks?",
                "How do influencers shape discourse?",
                "How do communities form around shared beliefs?",
                "How can misinformation spread be predicted?",
                "How effective are interventions?"
            ]
        }
        
        # Analysis pipeline templates
        self.pipeline_templates = {
            "influence_analysis": [
                "Extract user entities from tweets",
                "Build interaction network", 
                "Compute centrality measures",
                "Identify key influencers",
                "Analyze influence patterns"
            ],
            "narrative_tracking": [
                "Extract topics from tweets",
                "Track topic evolution over time",
                "Identify narrative themes",
                "Measure narrative spread",
                "Analyze narrative persistence"
            ],
            "community_detection": [
                "Build user interaction network",
                "Apply community detection algorithms",
                "Characterize community properties",
                "Analyze inter-community dynamics",
                "Identify bridge users"
            ],
            "stance_dynamics": [
                "Classify tweet stances",
                "Track stance changes over time",
                "Identify stance-switching users",
                "Analyze persuasion patterns",
                "Model stance evolution"
            ],
            "cross_modal_analysis": [
                "Combine multiple interrogative views",
                "Link entities across views",
                "Perform joint analysis",
                "Synthesize insights",
                "Generate comprehensive report"
            ]
        }
    
    def generate_interrogative_views(self, domain: str, num_views: int = 2) -> List[InterrogativeView]:
        """Generate diverse interrogative views for analysis"""
        views = []
        interrogatives = list(self.interrogative_templates.keys())
        
        # Randomly select interrogatives for diversity
        selected = random.sample(interrogatives, min(num_views, len(interrogatives)))
        
        for interrogative in selected:
            questions = self.interrogative_templates[interrogative]
            question = random.choice(questions).format(domain=domain)
            
            # Generate entities and relationships based on interrogative
            entities, relationships = self._get_entities_relationships(interrogative)
            
            view = InterrogativeView(
                interrogative=interrogative,
                focus=question,
                description=f"Analyzes {domain} from the perspective of '{interrogative}' to understand {question.lower()}",
                entities=entities,
                relationships=relationships,
                analysis_goals=self._get_analysis_goals(interrogative)
            )
            views.append(view)
        
        return views
    
    def _get_entities_relationships(self, interrogative: str) -> Tuple[List[str], List[str]]:
        """Get relevant entities and relationships for an interrogative"""
        mappings = {
            "Who": (
                ["User", "Influencer", "Community", "BridgeUser"],
                ["FOLLOWS", "MENTIONS", "RETWEETS", "MEMBER_OF"]
            ),
            "What": (
                ["Tweet", "Topic", "Narrative", "Hashtag", "Argument"],
                ["DISCUSSES", "CONTAINS", "REFERENCES", "EVOLVES_TO"]
            ),
            "When": (
                ["Tweet", "Event", "TrendingPeriod", "User"],
                ["POSTED_AT", "OCCURRED_DURING", "PRECEDED", "FOLLOWED"]
            ),
            "Where": (
                ["Location", "Platform", "Community", "GeographicRegion"],
                ["LOCATED_IN", "POSTED_ON", "SPREADS_TO", "ORIGINATES_FROM"]
            ),
            "Why": (
                ["Motivation", "User", "SocialFactor", "PsychologicalFactor"],
                ["MOTIVATED_BY", "INFLUENCES", "CAUSES", "RESULTS_IN"]
            ),
            "How": (
                ["Process", "Mechanism", "Pathway", "Strategy"],
                ["ENABLES", "FACILITATES", "LEADS_TO", "IMPLEMENTS"]
            )
        }
        return mappings.get(interrogative, ([], []))
    
    def _get_analysis_goals(self, interrogative: str) -> List[str]:
        """Get analysis goals for an interrogative"""
        goals = {
            "Who": ["Identify key actors", "Measure influence", "Detect communities", "Find bridge users"],
            "What": ["Extract themes", "Classify content", "Track narratives", "Measure engagement"],
            "When": ["Temporal analysis", "Trend detection", "Event correlation", "Timing patterns"],
            "Where": ["Spatial analysis", "Platform comparison", "Geographic spread", "Location influence"],
            "Why": ["Causal analysis", "Motivation inference", "Factor identification", "Explanation generation"],
            "How": ["Process analysis", "Mechanism discovery", "Pathway tracing", "Strategy evaluation"]
        }
        return goals.get(interrogative, [])
    
    def generate_analysis_scenarios(self, input_data: AutoInterrogativePlanInput) -> List[AnalysisScenario]:
        """Generate diverse analysis scenarios"""
        scenarios = []
        
        # Define scenario templates with varying complexity
        scenario_templates = [
            {
                "title": "Influence Network Analysis",
                "research_question": "Who are the most influential users in conspiracy theory discourse and how do they shape narratives?",
                "complexity": "Simple",
                "num_views": 2,
                "pipeline_type": "influence_analysis"
            },
            {
                "title": "Narrative Evolution Tracking", 
                "research_question": "How do conspiracy theory narratives evolve and spread over time?",
                "complexity": "Medium",
                "num_views": 2,
                "pipeline_type": "narrative_tracking"
            },
            {
                "title": "Community Structure Analysis",
                "research_question": "What communities form around conspiracy theories and how do they interact?",
                "complexity": "Medium",
                "num_views": 2,
                "pipeline_type": "community_detection"
            },
            {
                "title": "Stance Change Dynamics",
                "research_question": "How and why do users change their stance on conspiracy theories?",
                "complexity": "Complex",
                "num_views": 3,
                "pipeline_type": "stance_dynamics"
            },
            {
                "title": "Cross-Modal Conspiracy Analysis",
                "research_question": "What insights emerge from analyzing conspiracy theories across multiple dimensions?",
                "complexity": "Complex",
                "num_views": 4,
                "pipeline_type": "cross_modal_analysis"
            },
            {
                "title": "Misinformation Amplification Patterns",
                "research_question": "How do specific users and communities amplify misinformation?",
                "complexity": "Medium",
                "num_views": 2,
                "pipeline_type": "influence_analysis"
            },
            {
                "title": "Conspiracy Theory Persuasion Mechanisms", 
                "research_question": "What makes conspiracy theory content persuasive and engaging?",
                "complexity": "Complex",
                "num_views": 3,
                "pipeline_type": "cross_modal_analysis"
            }
        ]
        
        # Filter by complexity range and select scenarios
        filtered_templates = [
            t for t in scenario_templates 
            if t["complexity"] in input_data.complexity_range
        ]
        
        selected_templates = random.sample(
            filtered_templates,
            min(input_data.num_scenarios, len(filtered_templates))
        )
        
        for template in selected_templates:
            # Generate interrogative views
            views = self.generate_interrogative_views(
                input_data.domain, 
                template["num_views"]
            )
            
            # Get analysis pipeline
            pipeline = self.pipeline_templates[template["pipeline_type"]]
            
            # Generate expected insights
            insights = self._generate_expected_insights(template, views)
            
            scenario = AnalysisScenario(
                title=template["title"],
                research_question=template["research_question"],
                interrogative_views=views,
                analysis_pipeline=pipeline,
                expected_insights=insights,
                complexity_level=template["complexity"]
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_expected_insights(self, template: Dict, views: List[InterrogativeView]) -> List[str]:
        """Generate expected insights for a scenario"""
        base_insights = {
            "Simple": [
                "Identify key patterns in the data",
                "Measure basic metrics and statistics",
                "Generate descriptive summaries"
            ],
            "Medium": [
                "Discover non-obvious relationships",
                "Identify temporal or spatial patterns",
                "Predict future trends or behaviors",
                "Compare across different groups or time periods"
            ],
            "Complex": [
                "Uncover causal relationships",
                "Generate actionable recommendations",
                "Synthesize insights across multiple dimensions",
                "Model complex system dynamics",
                "Predict intervention effectiveness"
            ]
        }
        
        complexity = template["complexity"]
        insights = base_insights[complexity].copy()
        
        # Add view-specific insights
        for view in views:
            if view.interrogative == "Who":
                insights.append("Map influence networks and key actors")
            elif view.interrogative == "What":
                insights.append("Identify dominant narratives and themes")
            elif view.interrogative == "How":
                insights.append("Understand mechanisms and processes")
        
        return insights

# Tool function following DIGIMON pattern
async def generate_interrogative_analysis_plans(
    tool_input: AutoInterrogativePlanInput,
    context: Optional[Any] = None  # GraphRAGContext would go here if needed
) -> AutoInterrogativePlanOutput:
    """Automatically generate diverse analysis scenarios with interrogative views"""
    planner = AutomatedInterrogativePlanner()
    try:
        # Generate analysis scenarios
        scenarios = planner.generate_analysis_scenarios(tool_input)
        
        # Determine execution order (simple to complex)
        execution_order = sorted(
            [s.title for s in scenarios],
            key=lambda title: next(s.complexity_level for s in scenarios if s.title == title)
        )
        
        # Estimate complexity
        complexity_estimates = {
            s.title: len(s.analysis_pipeline) * len(s.interrogative_views)
            for s in scenarios
        }
        
        return AutoInterrogativePlanOutput(
            success=True,
            scenarios=scenarios,
            execution_order=execution_order,
            estimated_complexity=complexity_estimates
        )
        
    except Exception as e:
        planner.logger.error(f"Auto planning failed: {str(e)}")
        return AutoInterrogativePlanOutput(
            success=False,
            scenarios=[],
            execution_order=[],
            estimated_complexity={}
        )

# Export for registration
__all__ = [
    'generate_interrogative_analysis_plans',
    'AutomatedInterrogativePlanner',
    'AutoInterrogativePlanInput', 
    'AutoInterrogativePlanOutput',
    'AnalysisScenario',
    'InterrogativeView'
]