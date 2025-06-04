"""
Discourse Analysis Prompts for Social Media Conspiracy Theory Analysis

These prompts guide the DIGIMON agent to understand and execute sophisticated
discourse analysis based on interrogative views and mini-ontologies.
"""

DISCOURSE_ANALYSIS_SYSTEM_PROMPT = """You are an expert discourse analyst specializing in social media conspiracy theory analysis. Your goal is to analyze discourse through multiple interrogative lenses to understand how conspiracy theories spread, who spreads them, and what effects they have.

You work with a knowledge graph that represents:
- WHO: Users, influencers, and communities in the discourse
- SAYS WHAT: Arguments, narratives, and conspiracy theories being discussed
- TO WHOM: Target audiences, followers, and community members
- IN WHAT SETTING: Platforms, contexts, and discussion environments
- WITH WHAT EFFECT: Engagement, radicalization, belief changes

Your analysis follows this structured approach:
1. Define interrogative views based on research questions
2. Generate mini-ontologies for each view
3. Apply retrievals to extract relevant data
4. Transform data to generate insights
5. Synthesize findings across multiple views"""

INTERROGATIVE_VIEW_PROMPTS = {
    "who": """Analyze the WHO dimension:
- Identify key influencers based on engagement metrics and network centrality
- Map user communities and their characteristics
- Track influence patterns and authority structures
- Consider: follower_count, engagement_score, toxicity_score, centrality measures""",
    
    "says_what": """Analyze the SAYS WHAT dimension:
- Extract dominant narratives and conspiracy theories
- Track argument evolution and mutation
- Identify rhetorical strategies and persuasion techniques
- Consider: topic frequency, sentiment, toxicity, narrative coherence""",
    
    "to_whom": """Analyze the TO WHOM dimension:
- Identify target audiences for different narratives
- Map information flow between communities
- Track audience receptivity and engagement
- Consider: community membership, engagement patterns, demographic indicators""",
    
    "in_what_setting": """Analyze the IN WHAT SETTING dimension:
- Map platform-specific discourse patterns
- Identify contextual factors affecting spread
- Track temporal and spatial patterns
- Consider: platform features, timing, geographic patterns""",
    
    "with_what_effect": """Analyze the WITH WHAT EFFECT dimension:
- Measure belief changes and radicalization
- Track narrative effectiveness and spread
- Identify intervention opportunities
- Consider: engagement metrics, stance changes, network growth"""
}

MINI_ONTOLOGY_GENERATION_PROMPT = """Given the interrogative view '{view}' for analyzing {domain}, generate a mini-ontology that includes:

1. ENTITIES (Nodes):
   - List relevant entity types with descriptions
   - Define key properties for each entity type
   - Consider measurement scales (categorical, ordinal, continuous)

2. RELATIONS (Edges):
   - Define directed relationships between entities
   - Specify properties on relationships
   - Include temporal aspects where relevant

3. CONSTRAINTS:
   - Domain and range constraints for relations
   - Multiplicity constraints (one-to-many, many-to-many)
   - Logical inference rules (transitivity, symmetry, inverse)

4. INFERENCE RULES:
   - Type propagation rules
   - Compositional inferences
   - Derived relationships

Focus on elements directly relevant to answering: {research_question}"""

RETRIEVAL_CHAIN_PROMPT = """Design a retrieval chain for the analysis goal: {goal}

Consider these retrieval operators:
- Chunk Retrieval: by_ppr, by_relationship, entity_occurrence
- Entity Retrieval: by_relationship, by_vdb, by_agent, by_ppr
- Relationship Retrieval: by_vdb, by_agent, by_entity, by_ppr
- Community Retrieval: by_entity, by_level
- Subgraph Retrieval: by_path, by_SteinerTree, induced_subgraph

Create a step-by-step retrieval plan that:
1. Starts with broad retrieval to identify relevant elements
2. Progressively focuses on specific patterns
3. Combines multiple retrieval methods for comprehensive coverage
4. Outputs structured data ready for transformation"""

TRANSFORMATION_CHAIN_PROMPT = """Design transformations for the retrieved data to achieve: {goal}

Consider these transformation operators:
- Descriptive: to_categorical_distribution, to_statistical_distribution, to_summary
- Predictive: predict_edge_weight, predict_missing_edges, simulate_network_evolution
- Explanatory: map_to_causal_model, find_causal_paths, compute_indirect_influence
- Intervention: simulate_intervention, optimize_community_moderation

Create transformations that:
1. Convert raw graph data into analytical insights
2. Apply appropriate statistical or ML methods
3. Generate interpretable outputs
4. Support decision-making and intervention planning"""

SYNTHESIS_PROMPT = """Synthesize findings across multiple interrogative views:

Views analyzed:
{views_summary}

Key findings:
{findings}

Generate a coherent narrative that:
1. Integrates insights from different views
2. Identifies cross-cutting patterns
3. Highlights unexpected connections
4. Provides actionable recommendations
5. Suggests areas for further investigation

Consider interactions between:
- WHO influences WHAT narratives
- HOW arguments spread TO WHOM
- WHEN and WHERE effects manifest
- WHAT interventions might work"""

def generate_analysis_prompt(interrogative_views, research_question, domain="COVID-19 conspiracy theories"):
    """Generate a comprehensive analysis prompt for the agent"""
    
    prompt = DISCOURSE_ANALYSIS_SYSTEM_PROMPT + "\n\n"
    prompt += f"Research Question: {research_question}\n"
    prompt += f"Domain: {domain}\n\n"
    
    prompt += "You will analyze this through the following interrogative views:\n"
    for view in interrogative_views:
        prompt += f"\n{view.upper()}:\n"
        prompt += INTERROGATIVE_VIEW_PROMPTS.get(view.lower(), "")
    
    prompt += "\n\nProceed with systematic analysis following the discourse analysis framework."
    
    return prompt

def generate_entity_extraction_prompt(text, interrogative_view):
    """Generate prompt for extracting entities relevant to a specific view"""
    
    view_specific_entities = {
        "who": ["users", "influencers", "communities", "networks"],
        "says_what": ["claims", "narratives", "arguments", "theories", "sources"],
        "to_whom": ["audiences", "followers", "targets", "recipients"],
        "in_what_setting": ["platforms", "contexts", "channels", "environments"],
        "with_what_effect": ["outcomes", "changes", "impacts", "responses"]
    }
    
    entities_to_extract = view_specific_entities.get(interrogative_view.lower(), [])
    
    prompt = f"""Extract entities relevant to the '{interrogative_view}' analysis from this text:

{text}

Focus on identifying:
{', '.join(entities_to_extract)}

For each entity, provide:
- Type (from the categories above)
- Name/identifier
- Key properties
- Relevance score (0-1)
- Evidence from text

Return as structured JSON."""
    
    return prompt

def generate_relationship_inference_prompt(entities, interrogative_view):
    """Generate prompt for inferring relationships between entities"""
    
    view_specific_relations = {
        "who": ["FOLLOWS", "INFLUENCES", "MEMBER_OF", "ALLIES_WITH"],
        "says_what": ["CLAIMS", "SUPPORTS", "CONTRADICTS", "CITES"],
        "to_whom": ["TARGETS", "REACHES", "ENGAGES", "CONVINCES"],
        "in_what_setting": ["POSTED_ON", "OCCURRED_IN", "SPREAD_THROUGH"],
        "with_what_effect": ["CAUSED", "LED_TO", "CHANGED", "PREVENTED"]
    }
    
    relations_to_infer = view_specific_relations.get(interrogative_view.lower(), [])
    
    prompt = f"""Given these entities from the '{interrogative_view}' view:

{entities}

Infer relationships of these types:
{', '.join(relations_to_infer)}

For each relationship, provide:
- Source entity
- Relationship type
- Target entity
- Confidence score (0-1)
- Properties (e.g., timestamp, strength, sentiment)
- Evidence/reasoning

Consider both explicit and implicit relationships."""
    
    return prompt

# Export all prompts and functions
__all__ = [
    'DISCOURSE_ANALYSIS_SYSTEM_PROMPT',
    'INTERROGATIVE_VIEW_PROMPTS',
    'MINI_ONTOLOGY_GENERATION_PROMPT',
    'RETRIEVAL_CHAIN_PROMPT',
    'TRANSFORMATION_CHAIN_PROMPT',
    'SYNTHESIS_PROMPT',
    'generate_analysis_prompt',
    'generate_entity_extraction_prompt',
    'generate_relationship_inference_prompt'
]