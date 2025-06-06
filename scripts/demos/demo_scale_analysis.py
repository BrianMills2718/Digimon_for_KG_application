#!/usr/bin/env python3
"""
Comprehensive demonstration of DIGIMON's capabilities at scale
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from Core.AgentOrchestrator.memory_enhanced_orchestrator import MemoryEnhancedOrchestrator
from Core.AgentBrain.agent_brain import AgentBrain
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Config.LLMConfig import LLMConfig
from Core.Common.Logger import logger


async def run_scale_demonstrations():
    """Run multiple sophisticated analyses to demonstrate DIGIMON's capabilities"""
    
    # Initialize components
    config_path = Path("Option/Config2.yaml")
    llm_config = LLMConfig.from_yaml(str(config_path))
    llm = LiteLLMProvider(llm_config)
    brain = AgentBrain(llm)
    orchestrator = MemoryEnhancedOrchestrator(brain=brain, llm=llm)
    
    corpus_path = Path("Data/COVID_Conspiracy/Corpus.json")
    
    demonstrations = [
        {
            "name": "Multi-Dimensional Network Analysis",
            "query": """Perform a comprehensive multi-dimensional analysis of the COVID conspiracy network:
            1. Map the complete influence network showing who influences whom
            2. Identify bridge nodes that connect different conspiracy communities
            3. Analyze information cascade patterns and viral spread dynamics
            4. Compute network centrality metrics (betweenness, eigenvector, PageRank)
            5. Detect echo chambers and filter bubbles
            6. Track how misinformation mutates as it spreads through the network
            """,
            "focus": "Advanced network science at scale"
        },
        {
            "name": "Temporal Evolution Analysis",
            "query": """Analyze the temporal evolution of conspiracy narratives:
            1. Track how conspiracy theories evolved over time
            2. Identify critical events that triggered narrative shifts
            3. Measure the half-life of different conspiracy memes
            4. Analyze daily/weekly patterns in conspiracy spreading
            5. Identify early warning signals of viral misinformation
            6. Map the lifecycle stages of successful vs failed conspiracies
            """,
            "focus": "Time-series analysis of discourse evolution"
        },
        {
            "name": "Cross-Theory Synthesis",
            "query": """Analyze interconnections between different conspiracy theories:
            1. Map how 5G, bioweapon, and population control theories interconnect
            2. Identify users who bridge multiple conspiracy communities
            3. Analyze linguistic patterns that indicate conspiracy fusion
            4. Track how conspiracies merge and split over time
            5. Identify the core beliefs that unite different conspiracy groups
            6. Measure ideological distance between conspiracy clusters
            """,
            "focus": "Complex cross-domain analysis"
        },
        {
            "name": "Predictive Risk Assessment",
            "query": """Build predictive models for conspiracy spread:
            1. Identify early indicators of users becoming conspiracy spreaders
            2. Predict which narratives are likely to go viral
            3. Assess community vulnerability to different conspiracy types
            4. Model the effectiveness of different intervention strategies
            5. Identify high-risk network configurations
            6. Forecast future conspiracy evolution trajectories
            """,
            "focus": "Predictive analytics and risk modeling"
        },
        {
            "name": "Semantic Deep Dive",
            "query": """Perform deep semantic analysis of conspiracy language:
            1. Extract and analyze rhetorical strategies used in conspiracies
            2. Identify emotional manipulation techniques
            3. Analyze metaphors and framing devices
            4. Track semantic drift in conspiracy terminology
            5. Identify dog whistles and coded language
            6. Measure sentiment polarization across communities
            """,
            "focus": "Advanced NLP and discourse analysis"
        }
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "COVID-19 Conspiracy Tweets (6,590 tweets)",
        "demonstrations": {}
    }
    
    for demo in demonstrations:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {demo['name']}")
        logger.info(f"Focus: {demo['focus']}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Execute the analysis
            context = {
                "corpus_path": str(corpus_path),
                "operation": demo["name"],
                "scale_test": True
            }
            
            result = await orchestrator.process_query(
                query=demo["query"],
                context=context
            )
            
            results["demonstrations"][demo["name"]] = {
                "query": demo["query"],
                "focus": demo["focus"],
                "status": "completed",
                "summary": result.get("answer", ""),
                "metrics": {
                    "entities_analyzed": result.get("entity_count", 0),
                    "relationships_found": result.get("relationship_count", 0),
                    "communities_detected": result.get("community_count", 0),
                    "insights_generated": len(result.get("insights", [])),
                    "execution_time": result.get("execution_time", 0)
                }
            }
            
            logger.info(f"\nCompleted: {demo['name']}")
            logger.info(f"Summary: {result.get('answer', '')[:500]}...")
            
        except Exception as e:
            logger.error(f"Error in {demo['name']}: {e}")
            results["demonstrations"][demo["name"]] = {
                "status": "error",
                "error": str(e)
            }
    
    # Save comprehensive results
    output_path = Path("scale_demonstration_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Scale demonstration complete!")
    logger.info(f"Results saved to: {output_path}")
    
    # Print summary statistics
    completed = sum(1 for d in results["demonstrations"].values() if d.get("status") == "completed")
    logger.info(f"\nSummary:")
    logger.info(f"- Demonstrations completed: {completed}/{len(demonstrations)}")
    
    if completed > 0:
        total_entities = sum(d.get("metrics", {}).get("entities_analyzed", 0) 
                           for d in results["demonstrations"].values())
        total_relationships = sum(d.get("metrics", {}).get("relationships_found", 0) 
                                for d in results["demonstrations"].values())
        total_insights = sum(d.get("metrics", {}).get("insights_generated", 0) 
                           for d in results["demonstrations"].values())
        
        logger.info(f"- Total entities analyzed: {total_entities:,}")
        logger.info(f"- Total relationships found: {total_relationships:,}")
        logger.info(f"- Total insights generated: {total_insights}")


if __name__ == "__main__":
    asyncio.run(run_scale_demonstrations())