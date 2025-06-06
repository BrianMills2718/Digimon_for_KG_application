#!/usr/bin/env python3
"""
Parallel batch analysis to demonstrate DIGIMON's scalability
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
import time

from Core.AgentOrchestrator.parallel_orchestrator import ParallelOrchestrator
from Core.AgentBrain.agent_brain import AgentBrain
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Config.LLMConfig import LLMConfig
from Core.Common.Logger import logger


async def run_parallel_analysis():
    """Run parallel analyses across different aspects of the dataset"""
    
    # Initialize components
    config_path = Path("Option/Config2.yaml")
    llm_config = LLMConfig.from_yaml(str(config_path))
    llm = LiteLLMProvider(llm_config)
    brain = AgentBrain(llm)
    orchestrator = ParallelOrchestrator(brain=brain, llm=llm)
    
    corpus_path = Path("Data/COVID_Conspiracy/Corpus.json")
    
    # Define 20 parallel analyses covering different aspects
    parallel_queries = [
        # Network Analysis Batch
        "Identify the top 10 super-spreaders and their network characteristics",
        "Map all conspiracy communities and their interconnections",
        "Analyze information cascade patterns for viral tweets",
        "Compute PageRank scores for all users in the network",
        
        # Content Analysis Batch
        "Extract all unique conspiracy narratives and their variants",
        "Identify emotional manipulation techniques in top 100 tweets",
        "Analyze metaphors and framing devices across all tweets",
        "Track semantic evolution of key conspiracy terms",
        
        # Temporal Analysis Batch
        "Create hourly timeline of conspiracy spread patterns",
        "Identify critical events that triggered narrative shifts",
        "Measure decay rates of different conspiracy types",
        "Detect periodicity in conspiracy posting behavior",
        
        # Cross-Theory Analysis Batch
        "Map connections between 5G and bioweapon theories",
        "Identify users bridging multiple conspiracy types",
        "Analyze linguistic markers of conspiracy fusion",
        "Measure ideological distances between theories",
        
        # Predictive Analysis Batch
        "Build user risk profiles for conspiracy susceptibility",
        "Predict next likely narrative mutations",
        "Identify early warning signals of viral spread",
        "Model intervention effectiveness scenarios"
    ]
    
    logger.info(f"Starting parallel analysis of {len(parallel_queries)} queries...")
    start_time = time.time()
    
    # Execute all queries in parallel
    context = {
        "corpus_path": str(corpus_path),
        "parallel_execution": True,
        "batch_mode": True
    }
    
    tasks = []
    for i, query in enumerate(parallel_queries):
        task_context = context.copy()
        task_context["query_id"] = f"Q{i+1:02d}"
        tasks.append(orchestrator.process_query(query, task_context))
    
    # Run all analyses in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    execution_time = time.time() - start_time
    
    # Process results
    analysis_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "COVID-19 Conspiracy Tweets (6,590 tweets)",
        "execution_mode": "parallel",
        "total_queries": len(parallel_queries),
        "execution_time_seconds": execution_time,
        "queries_per_second": len(parallel_queries) / execution_time,
        "results": {}
    }
    
    successful = 0
    total_entities = 0
    total_relationships = 0
    
    for i, (query, result) in enumerate(zip(parallel_queries, results)):
        query_id = f"Q{i+1:02d}"
        
        if isinstance(result, Exception):
            analysis_results["results"][query_id] = {
                "query": query,
                "status": "error",
                "error": str(result)
            }
        else:
            successful += 1
            analysis_results["results"][query_id] = {
                "query": query,
                "status": "success",
                "answer_preview": result.get("answer", "")[:200] + "...",
                "entities_found": result.get("entity_count", 0),
                "relationships_found": result.get("relationship_count", 0)
            }
            total_entities += result.get("entity_count", 0)
            total_relationships += result.get("relationship_count", 0)
    
    analysis_results["summary"] = {
        "successful_queries": successful,
        "failed_queries": len(parallel_queries) - successful,
        "success_rate": successful / len(parallel_queries),
        "total_entities_processed": total_entities,
        "total_relationships_found": total_relationships,
        "avg_time_per_query": execution_time / len(parallel_queries)
    }
    
    # Save results
    output_path = Path("parallel_batch_results.json")
    with open(output_path, "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Parallel Batch Analysis Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Total queries: {len(parallel_queries)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Total execution time: {execution_time:.2f} seconds")
    logger.info(f"Average time per query: {execution_time/len(parallel_queries):.2f} seconds")
    logger.info(f"Queries per second: {len(parallel_queries)/execution_time:.2f}")
    logger.info(f"Total entities processed: {total_entities:,}")
    logger.info(f"Total relationships found: {total_relationships:,}")
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(run_parallel_analysis())