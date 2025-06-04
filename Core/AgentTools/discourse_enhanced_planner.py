"""
Enhanced Interrogative Planner with Discourse Analysis Framework

This planner generates analysis scenarios based on the five interrogatives
of discourse analysis: Who, Says What, To Whom, In What Setting, With What Effect
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import random
from pydantic import BaseModel, Field
from Core.AgentSchema.tool_contracts import BaseToolParams, BaseToolOutput
from Core.Common.Logger import logger
from Core.Common.LLM import LLM
from Core.AgentTools.discourse_analysis_prompts import (
    generate_analysis_prompt,
    MINI_ONTOLOGY_GENERATION_PROMPT,
    RETRIEVAL_CHAIN_PROMPT,
    TRANSFORMATION_CHAIN_PROMPT
)

class DiscourseInterrogativeView(BaseModel):
    """Enhanced interrogative view for discourse analysis"""
    interrogative: str = Field(description="Who/Says What/To Whom/In What Setting/With What Effect")
    focus: str = Field(description="Specific focus of the interrogative")
    description: str = Field(description="Detailed description of what this view analyzes")
    entities: List[str] = Field(description="Key entity types for this view")
    relationships: List[str] = Field(description="Key relationship types")
    properties: List[str] = Field(description="Key properties to extract")
    analysis_goals: List[str] = Field(description="What we want to achieve with this view")
    retrieval_operators: List[str] = Field(description="Preferred retrieval operators")
    transformation_operators: List[str] = Field(description="Preferred transformation operators")

class DiscourseAnalysisScenario(BaseModel):
    """Enhanced analysis scenario with discourse framework"""
    title: str = Field(description="Scenario title")
    research_question: str = Field(description="Main research question")
    interrogative_views: List[DiscourseInterrogativeView] = Field(description="Multiple interrogative perspectives")
    mini_ontologies: Dict[str, Dict] = Field(description="Mini-ontology for each view")
    unified_ontology: Dict = Field(description="Merged ontology across views")
    retrieval_chains: List[Dict] = Field(description="Retrieval chain specifications")
    transformation_chains: List[Dict] = Field(description="Transformation chain specifications")
    expected_insights: List[str] = Field(description="What insights we expect to gain")
    complexity_level: str = Field(description="Simple/Medium/Complex")
    analysis_pipeline: List[str] = Field(default_factory=list, description="Analysis pipeline steps")

class DiscourseEnhancedPlanner:
    """Generates analysis plans using discourse analysis framework"""
    
    def __init__(self):
        self.logger = logger
        self.llm = LLM()
        
        # Define entity types for each interrogative
        self.interrogative_entities = {
            "Who": {
                "entities": ["User", "Influencer", "Community", "Network", "Actor"],
                "relationships": ["FOLLOWS", "INFLUENCES", "MEMBER_OF", "ALLIES_WITH", "OPPOSES"],
                "properties": ["follower_count", "engagement_score", "centrality", "authority_score"]
            },
            "Says What": {
                "entities": ["Post", "Topic", "Narrative", "Argument", "Claim", "Source"],
                "relationships": ["DISCUSSES", "SUPPORTS", "CONTRADICTS", "CITES", "EVOLVES_TO"],
                "properties": ["content", "sentiment", "toxicity", "virality", "credibility"]
            },
            "To Whom": {
                "entities": ["Audience", "Target", "Recipient", "Community", "Demographic"],
                "relationships": ["TARGETS", "REACHES", "ENGAGES", "CONVINCES", "RESONATES_WITH"],
                "properties": ["receptivity", "engagement_rate", "demographic_profile", "belief_system"]
            },
            "In What Setting": {
                "entities": ["Platform", "Context", "Channel", "Environment", "TimeWindow"],
                "relationships": ["POSTED_ON", "OCCURRED_IN", "SPREAD_THROUGH", "ENABLED_BY"],
                "properties": ["platform_features", "temporal_context", "spatial_context", "moderation_level"]
            },
            "With What Effect": {
                "entities": ["Effect", "Outcome", "Change", "Impact", "Response"],
                "relationships": ["CAUSED", "LED_TO", "CHANGED", "PREVENTED", "AMPLIFIED"],
                "properties": ["magnitude", "duration", "reversibility", "measurability"]
            }
        }
        
        # Define retrieval operators for each type
        self.retrieval_operators = {
            "chunk": ["by_ppr", "by_relationship", "entity_occurrence"],
            "entity": ["by_relationship", "by_vdb", "by_agent", "by_ppr"],
            "relationship": ["by_vdb", "by_agent", "by_entity", "by_ppr"],
            "community": ["by_entity", "by_level"],
            "subgraph": ["by_path", "by_SteinerTree", "induced_subgraph"]
        }
        
        # Define transformation operators by goal
        self.transformation_operators = {
            "descriptive": ["to_categorical_distribution", "to_statistical_distribution", "to_summary"],
            "predictive": ["predict_edge_weight", "predict_missing_edges", "simulate_network_evolution"],
            "explanatory": ["map_to_causal_model", "find_causal_paths", "compute_indirect_influence"],
            "intervention": ["simulate_intervention", "optimize_community_moderation", "analyze_network_resilience"]
        }
    
    def generate_discourse_views(self, research_question: str, selected_interrogatives: List[str]) -> List[DiscourseInterrogativeView]:
        """Generate interrogative views for discourse analysis"""
        views = []
        
        for interrogative in selected_interrogatives:
            # Get base entities and relationships
            base_config = self.interrogative_entities.get(interrogative, {})
            
            # Create focused view based on research question
            if interrogative == "Who":
                focus = "Key actors spreading conspiracy theories"
                goals = ["Identify influencers", "Map authority structures", "Detect coordinated behavior"]
                retrieval_ops = ["by_ppr", "by_relationship", "by_agent"]
                transform_ops = ["to_categorical_distribution", "compute_indirect_influence"]
                
            elif interrogative == "Says What":
                focus = "Dominant conspiracy narratives and arguments"
                goals = ["Extract narratives", "Track evolution", "Identify rhetorical strategies"]
                retrieval_ops = ["entity_occurrence", "by_vdb", "by_relationship"]
                transform_ops = ["to_summary", "find_causal_paths", "to_statistical_distribution"]
                
            elif interrogative == "To Whom":
                focus = "Target audiences and their receptivity"
                goals = ["Identify audiences", "Measure engagement", "Track belief changes"]
                retrieval_ops = ["by_entity", "by_level", "induced_subgraph"]
                transform_ops = ["to_categorical_distribution", "predict_edge_weight"]
                
            elif interrogative == "In What Setting":
                focus = "Platform and contextual factors"
                goals = ["Map platform patterns", "Identify enabling contexts", "Track temporal spread"]
                retrieval_ops = ["by_entity", "entity_occurrence", "by_ppr"]
                transform_ops = ["to_statistical_distribution", "simulate_network_evolution"]
                
            elif interrogative == "With What Effect":
                focus = "Measurable impacts on beliefs and behavior"
                goals = ["Measure belief changes", "Track radicalization", "Identify intervention points"]
                retrieval_ops = ["by_path", "by_SteinerTree", "by_agent"]
                transform_ops = ["map_to_causal_model", "simulate_intervention", "compute_indirect_influence"]
            
            else:
                continue
            
            view = DiscourseInterrogativeView(
                interrogative=interrogative,
                focus=focus,
                description=f"Analyzes {research_question} from the perspective of '{interrogative}'",
                entities=base_config.get("entities", []),
                relationships=base_config.get("relationships", []),
                properties=base_config.get("properties", []),
                analysis_goals=goals,
                retrieval_operators=retrieval_ops,
                transformation_operators=transform_ops
            )
            views.append(view)
        
        return views
    
    def generate_mini_ontology(self, view: DiscourseInterrogativeView) -> Dict:
        """Generate mini-ontology for an interrogative view"""
        ontology = {
            "entities": {},
            "relationships": {},
            "constraints": {},
            "inference_rules": {}
        }
        
        # Define entities with properties
        for entity in view.entities:
            ontology["entities"][entity] = {
                "description": f"{entity} in the context of {view.interrogative}",
                "properties": {prop: "type_definition" for prop in view.properties},
                "key_property": view.properties[0] if view.properties else None
            }
        
        # Define relationships
        for relation in view.relationships:
            ontology["relationships"][relation] = {
                "description": f"{relation} relationship for {view.interrogative}",
                "domain": view.entities[0] if view.entities else None,
                "range": view.entities[1] if len(view.entities) > 1 else view.entities[0],
                "properties": ["timestamp", "strength", "confidence"]
            }
        
        # Add constraints
        ontology["constraints"] = {
            "multiplicity": {rel: "many-to-many" for rel in view.relationships},
            "required_properties": view.properties[:2] if len(view.properties) >= 2 else view.properties
        }
        
        # Add inference rules
        if view.interrogative == "Who":
            ontology["inference_rules"]["influence_transitivity"] = "If A influences B and B influences C, then A indirectly influences C"
        elif view.interrogative == "Says What":
            ontology["inference_rules"]["narrative_evolution"] = "If narrative A evolves to B and B evolves to C, track A->C transformation"
        
        return ontology
    
    def unify_ontologies(self, mini_ontologies: Dict[str, Dict]) -> Dict:
        """Merge mini-ontologies into unified ontology"""
        unified = {
            "entities": {},
            "relationships": {},
            "shared_entities": [],
            "bridge_relationships": []
        }
        
        # Merge entities
        for view_name, ontology in mini_ontologies.items():
            for entity, definition in ontology["entities"].items():
                if entity in unified["entities"]:
                    # Entity exists in multiple views - mark as shared
                    unified["shared_entities"].append(entity)
                    # Merge properties
                    existing_props = unified["entities"][entity].get("properties", {})
                    new_props = definition.get("properties", {})
                    unified["entities"][entity]["properties"] = {**existing_props, **new_props}
                else:
                    unified["entities"][entity] = definition
        
        # Merge relationships
        for view_name, ontology in mini_ontologies.items():
            for relation, definition in ontology["relationships"].items():
                if relation not in unified["relationships"]:
                    unified["relationships"][relation] = definition
        
        # Identify bridge relationships between views
        if "User" in unified["shared_entities"] and "Post" in unified["entities"]:
            unified["bridge_relationships"].append({
                "name": "POSTED",
                "connects": ["Who", "Says What"],
                "description": "Links users to their content"
            })
        
        return unified
    
    def generate_retrieval_chain(self, scenario: DiscourseAnalysisScenario, view: DiscourseInterrogativeView) -> List[Dict]:
        """Generate retrieval chain for a view"""
        chain = []
        
        # Start with broad retrieval
        chain.append({
            "step": 1,
            "description": f"Initial {view.interrogative} retrieval",
            "operator": view.retrieval_operators[0],
            "parameters": {
                "entity_types": view.entities[:2],
                "limit": 1000
            },
            "output": f"initial_{view.interrogative.lower()}_entities"
        })
        
        # Progressive refinement
        chain.append({
            "step": 2,
            "description": f"Refine {view.interrogative} entities",
            "operator": view.retrieval_operators[1],
            "parameters": {
                "input": f"initial_{view.interrogative.lower()}_entities",
                "relationships": view.relationships[:2],
                "threshold": 0.7
            },
            "output": f"refined_{view.interrogative.lower()}_entities"
        })
        
        # Context expansion
        chain.append({
            "step": 3,
            "description": f"Expand {view.interrogative} context",
            "operator": view.retrieval_operators[2] if len(view.retrieval_operators) > 2 else "by_relationship",
            "parameters": {
                "input": f"refined_{view.interrogative.lower()}_entities",
                "hop_distance": 1,
                "include_properties": view.properties
            },
            "output": f"complete_{view.interrogative.lower()}_subgraph"
        })
        
        return chain
    
    def generate_transformation_chain(self, scenario: DiscourseAnalysisScenario, view: DiscourseInterrogativeView) -> List[Dict]:
        """Generate transformation chain for a view"""
        chain = []
        
        # Descriptive transformation
        chain.append({
            "step": 1,
            "description": f"Describe {view.interrogative} patterns",
            "operator": view.transformation_operators[0],
            "input": f"complete_{view.interrogative.lower()}_subgraph",
            "parameters": {
                "group_by": view.properties[0] if view.properties else "entity_type",
                "metrics": ["count", "mean", "std"]
            },
            "output": f"{view.interrogative.lower()}_distribution"
        })
        
        # Analytical transformation
        if len(view.transformation_operators) > 1:
            chain.append({
                "step": 2,
                "description": f"Analyze {view.interrogative} dynamics",
                "operator": view.transformation_operators[1],
                "input": f"{view.interrogative.lower()}_distribution",
                "parameters": {
                "method": "advanced_analysis",
                    "confidence_threshold": 0.8
                },
                "output": f"{view.interrogative.lower()}_insights"
            })
        
        return chain
    
    def generate_scenarios(self, research_focus: str, num_scenarios: int = 5) -> List[DiscourseAnalysisScenario]:
        """Generate discourse analysis scenarios from a research focus"""
        # Generate research questions from the focus
        research_questions = self._generate_research_questions(research_focus, num_scenarios)
        return self._generate_scenarios_from_questions(research_questions, research_focus)
    
    def _generate_research_questions(self, research_focus: str, num_questions: int) -> List[str]:
        """Generate specific research questions from a broad focus"""
        # For COVID conspiracy theories, generate relevant questions
        base_questions = [
            "Who are the key influencers spreading COVID-19 conspiracy theories?",
            "What are the main conspiracy narratives and how do they evolve?",
            "Which audiences are most receptive to vaccine misinformation?",
            "How do platform features enable conspiracy theory spread?",
            "What are the measurable effects of conspiracy theories on public health behavior?"
        ]
        return base_questions[:num_questions]
    
    def _generate_scenarios_from_questions(self, research_questions: List[str], domain: str) -> List[DiscourseAnalysisScenario]:
        """Generate complete discourse analysis scenarios"""
        scenarios = []
        
        # Predefined scenario templates
        templates = [
            {
                "title": "Influence and Narrative Analysis",
                "interrogatives": ["Who", "Says What"],
                "complexity": "Simple",
                "focus": "Understanding who spreads what narratives"
            },
            {
                "title": "Audience Targeting Analysis",
                "interrogatives": ["Says What", "To Whom"],
                "complexity": "Simple",
                "focus": "How narratives target specific audiences"
            },
            {
                "title": "Platform Context Analysis",
                "interrogatives": ["Who", "In What Setting", "Says What"],
                "complexity": "Medium",
                "focus": "How platform contexts shape discourse"
            },
            {
                "title": "Impact Assessment",
                "interrogatives": ["Who", "Says What", "With What Effect"],
                "complexity": "Medium",
                "focus": "Measuring discourse impacts"
            },
            {
                "title": "Full Discourse Analysis",
                "interrogatives": ["Who", "Says What", "To Whom", "In What Setting", "With What Effect"],
                "complexity": "Complex",
                "focus": "Comprehensive discourse analysis"
            }
        ]
        
        for i, question in enumerate(research_questions[:5]):  # Limit to 5 scenarios
            template = templates[i % len(templates)]
            
            # Generate views
            views = self.generate_discourse_views(question, template["interrogatives"])
            
            # Generate mini-ontologies
            mini_ontologies = {}
            for view in views:
                mini_ontologies[view.interrogative] = self.generate_mini_ontology(view)
            
            # Unify ontologies
            unified_ontology = self.unify_ontologies(mini_ontologies)
            
            # Generate retrieval chains
            retrieval_chains = []
            for view in views:
                retrieval_chains.extend(self.generate_retrieval_chain(None, view))
            
            # Generate transformation chains
            transformation_chains = []
            for view in views:
                transformation_chains.extend(self.generate_transformation_chain(None, view))
            
            # Expected insights based on complexity
            if template["complexity"] == "Simple":
                insights = [
                    f"Key patterns in {template['focus']}",
                    "Basic statistical distributions",
                    "Entity identification and ranking"
                ]
            elif template["complexity"] == "Medium":
                insights = [
                    f"Relationships in {template['focus']}",
                    "Temporal and spatial patterns",
                    "Predictive indicators",
                    "Cross-dimensional correlations"
                ]
            else:
                insights = [
                    f"Causal mechanisms in {template['focus']}",
                    "System dynamics modeling",
                    "Intervention effectiveness predictions",
                    "Emergent phenomena identification",
                    "Comprehensive recommendations"
                ]
            
            scenario = DiscourseAnalysisScenario(
                title=template["title"],
                research_question=question,
                interrogative_views=views,
                mini_ontologies=mini_ontologies,
                unified_ontology=unified_ontology,
                retrieval_chains=retrieval_chains,
                transformation_chains=transformation_chains,
                expected_insights=insights,
                complexity_level=template["complexity"]
            )
            
            scenarios.append(scenario)
        
        return scenarios

# Tool functions for DIGIMON integration
class DiscourseAnalysisPlanInput(BaseToolParams):
    """Input for discourse analysis planning"""
    research_questions: List[str] = Field(description="Research questions to analyze")
    domain: str = Field(description="Analysis domain")
    selected_interrogatives: Optional[List[str]] = Field(
        default=None,
        description="Specific interrogatives to focus on"
    )

class DiscourseAnalysisPlanOutput(BaseToolOutput):
    """Output from discourse analysis planning"""
    success: bool = Field(description="Whether planning succeeded")
    scenarios: List[DiscourseAnalysisScenario] = Field(description="Generated scenarios")
    execution_order: List[str] = Field(description="Recommended execution order")

async def generate_discourse_analysis_plans(
    tool_input: DiscourseAnalysisPlanInput,
    context: Optional[Any] = None
) -> DiscourseAnalysisPlanOutput:
    """Generate discourse analysis plans"""
    planner = DiscourseEnhancedPlanner()
    
    try:
        # Use research questions from input
        research_questions = tool_input.research_questions
        if not research_questions:
            # Default questions for COVID conspiracy analysis
            research_questions = [
                "Who are the key actors spreading COVID conspiracy theories?",
                "What narratives dominate the conspiracy discourse?",
                "Which audiences are most receptive to conspiracy theories?",
                "How do platform features enable conspiracy spread?",
                "What are the measurable effects on public health behavior?"
            ]
        
        # Generate scenarios
        scenarios = planner.generate_scenarios(research_questions, tool_input.domain)
        
        # Order by complexity
        execution_order = sorted(
            [s.title for s in scenarios],
            key=lambda t: {"Simple": 1, "Medium": 2, "Complex": 3}.get(
                next(s.complexity_level for s in scenarios if s.title == t), 4
            )
        )
        
        return DiscourseAnalysisPlanOutput(
            success=True,
            scenarios=scenarios,
            execution_order=execution_order
        )
        
    except Exception as e:
        logger.error(f"Discourse analysis planning failed: {str(e)}")
        return DiscourseAnalysisPlanOutput(
            success=False,
            scenarios=[],
            execution_order=[]
        )

__all__ = [
    'DiscourseEnhancedPlanner',
    'DiscourseInterrogativeView',
    'DiscourseAnalysisScenario',
    'generate_discourse_analysis_plans',
    'DiscourseAnalysisPlanInput',
    'DiscourseAnalysisPlanOutput'
]