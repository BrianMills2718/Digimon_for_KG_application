#!/usr/bin/env python3
"""
Automated research generation - Agent autonomously generates and executes research questions
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
from Core.AgentTools.automated_interrogative_planner import AutomatedInterrogativePlanner


async def run_automated_research():
    """Let the agent autonomously generate and execute research questions"""
    
    # Initialize components
    config_path = Path("Option/Config2.yaml")
    llm_config = LLMConfig.from_yaml(str(config_path))
    llm = LiteLLMProvider(llm_config)
    brain = AgentBrain(llm)
    orchestrator = MemoryEnhancedOrchestrator(brain=brain, llm=llm)
    planner = AutomatedInterrogativePlanner(llm=llm)
    
    corpus_path = Path("Data/COVID_Conspiracy/Corpus.json")
    
    # Let the agent analyze the corpus and generate research questions
    logger.info("Phase 1: Autonomous Research Question Generation")
    logger.info("="*80)
    
    # First, have the agent understand the dataset
    understanding_query = """Analyze this COVID conspiracy tweet dataset and identify:
    1. Main themes and narratives present
    2. Key actors and communities
    3. Temporal patterns
    4. Interesting phenomena worth investigating
    Then generate 15 diverse research questions that would provide valuable insights for:
    - Public health officials
    - Social media platforms
    - Researchers studying misinformation
    - Policy makers
    """
    
    context = {
        "corpus_path": str(corpus_path),
        "mode": "research_generation"
    }
    
    # Get agent's understanding and generated questions
    understanding_result = await orchestrator.process_query(understanding_query, context)
    
    # Extract generated questions (assuming the agent returns them in a structured way)
    # For now, we'll use the planner to generate questions based on the understanding
    research_questions = await planner.generate_research_questions(
        dataset_description=understanding_result.get("answer", ""),
        num_questions=15,
        diversity_focus=True
    )
    
    logger.info(f"\nGenerated {len(research_questions)} research questions")
    
    # Phase 2: Execute all research questions
    logger.info("\nPhase 2: Autonomous Research Execution")
    logger.info("="*80)
    
    research_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "COVID-19 Conspiracy Tweets (6,590 tweets)",
        "mode": "fully_autonomous_research",
        "agent_generated_questions": research_questions,
        "research_findings": {}
    }
    
    for i, question in enumerate(research_questions, 1):
        logger.info(f"\nResearch Question {i}/{len(research_questions)}:")
        logger.info(f"Q: {question}")
        
        try:
            # Let the agent tackle each question with full autonomy
            result = await orchestrator.process_query(
                query=question,
                context={
                    "corpus_path": str(corpus_path),
                    "research_mode": True,
                    "question_number": i,
                    "allow_deep_analysis": True
                }
            )
            
            research_results["research_findings"][f"RQ{i:02d}"] = {
                "question": question,
                "status": "completed",
                "key_findings": result.get("answer", ""),
                "methodology": result.get("methodology", ""),
                "entities_analyzed": result.get("entity_count", 0),
                "relationships_explored": result.get("relationship_count", 0),
                "confidence_score": result.get("confidence", 0),
                "insights": result.get("insights", [])
            }
            
            logger.info(f"✓ Completed analysis")
            
        except Exception as e:
            logger.error(f"✗ Error: {e}")
            research_results["research_findings"][f"RQ{i:02d}"] = {
                "question": question,
                "status": "error",
                "error": str(e)
            }
    
    # Phase 3: Synthesis and Meta-Analysis
    logger.info("\nPhase 3: Research Synthesis and Meta-Analysis")
    logger.info("="*80)
    
    synthesis_query = f"""Based on the {len(research_questions)} research investigations conducted:
    1. Synthesize the major findings across all research questions
    2. Identify surprising or unexpected discoveries
    3. Find patterns that emerge across multiple investigations
    4. Generate actionable recommendations for each stakeholder group
    5. Suggest follow-up research directions
    6. Assess the overall threat level of COVID conspiracy theories
    """
    
    synthesis_result = await orchestrator.process_query(
        synthesis_query,
        context={
            "corpus_path": str(corpus_path),
            "previous_findings": research_results["research_findings"],
            "synthesis_mode": True
        }
    )
    
    research_results["synthesis"] = {
        "meta_analysis": synthesis_result.get("answer", ""),
        "key_themes": synthesis_result.get("themes", []),
        "recommendations": synthesis_result.get("recommendations", {}),
        "threat_assessment": synthesis_result.get("threat_level", ""),
        "future_research": synthesis_result.get("future_directions", [])
    }
    
    # Save comprehensive results
    output_path = Path("automated_research_results.json")
    with open(output_path, "w") as f:
        json.dump(research_results, f, indent=2)
    
    # Generate executive summary
    logger.info("\nGenerating Executive Summary...")
    
    executive_summary = f"""
AUTOMATED RESEARCH REPORT
========================
Dataset: COVID-19 Conspiracy Tweets (6,590 tweets)
Research Questions Generated: {len(research_questions)}
Successfully Analyzed: {sum(1 for r in research_results["research_findings"].values() if r["status"] == "completed")}

TOP FINDINGS:
{synthesis_result.get("answer", "")[:1000]}...

RECOMMENDATIONS:
{json.dumps(synthesis_result.get("recommendations", {}), indent=2)[:500]}...

Full report saved to: {output_path}
"""
    
    summary_path = Path("automated_research_executive_summary.txt")
    with open(summary_path, "w") as f:
        f.write(executive_summary)
    
    logger.info(executive_summary)
    logger.info(f"\nAutomated research complete!")
    logger.info(f"Full results: {output_path}")
    logger.info(f"Executive summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(run_automated_research())