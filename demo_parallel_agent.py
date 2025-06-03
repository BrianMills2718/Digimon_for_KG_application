#!/usr/bin/env python3
"""
Demo script showing parallel execution in the enhanced agent
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.Common.Logger import logger
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentOrchestrator.parallel_orchestrator import ParallelAgentOrchestrator
from Core.AgentSchema.context import GraphRAGContext
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory
from Option.Config2 import default_config

# Replace original tools with enhanced versions
import Core.AgentTools.entity_vdb_tools as original_entity_vdb
import Core.AgentTools.relationship_tools as original_relationship
from Core.AgentTools.enhanced_entity_vdb_tools import entity_vdb_build_tool as enhanced_entity_vdb_build
from Core.AgentTools.enhanced_relationship_tools import relationship_vdb_build_tool as enhanced_relationship_vdb_build

# Monkey patch the original modules with enhanced versions
original_entity_vdb.entity_vdb_build_tool = enhanced_entity_vdb_build
original_relationship.relationship_vdb_build_tool = enhanced_relationship_vdb_build


async def demo_parallel_agent():
    """Demonstrate the agent using parallel orchestrator."""
    
    logger.info("=== DIGIMON Parallel Agent Demo ===")
    logger.info("Features enabled:")
    logger.info("  ✓ Parallel execution of independent steps")
    logger.info("  ✓ Performance monitoring")
    logger.info("  ✓ Structured error handling")
    logger.info("  ✓ Batch embedding processing")
    logger.info("  ✓ Adaptive timeouts\n")
    
    try:
        # Use default config
        config = default_config
        
        # Create LLM provider
        logger.info("Creating LLM provider...")
        llm_provider = LiteLLMProvider(config)
        
        # Create embedding provider
        logger.info("Creating embedding provider...")
        embed_provider = get_rag_embedding(config)
        
        # Create chunk factory
        chunk_factory = ChunkFactory(config)
        
        # Initialize context
        logger.info("Initializing GraphRAG context...")
        context = GraphRAGContext(
            main_config=config,
            embedding_provider=embed_provider,
            target_dataset_name="MySampleTexts"
        )
        
        # Create PARALLEL orchestrator
        logger.info("Creating parallel orchestrator...")
        orchestrator = ParallelAgentOrchestrator(
            main_config=config,
            llm_instance=llm_provider,
            encoder_instance=embed_provider,
            chunk_factory=chunk_factory,
            graphrag_context=context
        )
        
        # Create planning agent
        logger.info("Creating planning agent...")
        agent = PlanningAgent(
            llm=orchestrator.enhanced_llm,
            orchestrator=orchestrator
        )
        
        # Demo query that will benefit from parallel execution
        test_query = "Compare the American and French revolutions"
        dataset = "MySampleTexts"
        
        logger.info(f"\nExecuting demo query:")
        logger.info(f"  Query: '{test_query}'")
        logger.info(f"  Dataset: {dataset}\n")
        
        # Generate plan
        logger.info("Generating execution plan...")
        plan = await agent.generate_plan(
            query=test_query,
            dataset_name=dataset,
            available_tools=list(orchestrator._tool_registry.keys())
        )
        
        if plan:
            logger.info(f"\nGenerated plan with {len(plan.steps)} steps:")
            for i, step in enumerate(plan.steps, 1):
                logger.info(f"  Step {i} ({step.step_id}): {step.description}")
                if hasattr(step.action, 'tools'):
                    for tool in step.action.tools:
                        logger.info(f"    └─ Tool: {tool.tool_id}")
            
            # Analyze dependencies
            logger.info("\nAnalyzing dependencies for parallel execution...")
            deps = orchestrator._analyze_dependencies(plan)
            groups = orchestrator._create_execution_groups(plan, deps)
            
            logger.info("\nExecution strategy:")
            for i, group in enumerate(groups, 1):
                if len(group) > 1:
                    logger.info(f"  Group {i}: {', '.join(group)} (PARALLEL)")
                else:
                    logger.info(f"  Group {i}: {', '.join(group)}")
            
            # Execute plan with parallel orchestrator
            logger.info("\nExecuting plan with parallel orchestrator...")
            results = await orchestrator.execute_plan(plan)
            
            logger.info("\n✅ Execution complete!")
            
            # Check for errors
            errors = []
            successful_steps = []
            for step_id, output in results.items():
                if isinstance(output, dict) and "error" in output:
                    errors.append(f"{step_id}: {output['error']}")
                else:
                    successful_steps.append(step_id)
            
            logger.info(f"\nResults summary:")
            logger.info(f"  Successful steps: {len(successful_steps)}")
            logger.info(f"  Failed steps: {len(errors)}")
            
            if errors:
                logger.error("\nErrors encountered:")
                for error in errors:
                    logger.error(f"  - {error}")
            
            # Get performance metrics
            if hasattr(orchestrator, 'performance_monitor'):
                logger.info("\nPerformance metrics:")
                summary = orchestrator.performance_monitor.get_summary()
                
                # Show parallel vs sequential timing
                if "parallel_plan_execution" in summary:
                    parallel_time = summary["parallel_plan_execution"]["duration"]["total"]
                    
                    # Calculate theoretical sequential time
                    sequential_time = 0
                    for operation, stats in summary.items():
                        if operation.startswith("step_") and stats['call_count'] > 0:
                            sequential_time += stats['duration']['total']
                    
                    if sequential_time > 0 and parallel_time > 0:
                        speedup = sequential_time / parallel_time
                        logger.info(f"  Parallel speedup: {speedup:.2f}x")
                        logger.info(f"  Time saved: {sequential_time - parallel_time:.1f}s")
                
                # Show individual operation times
                for operation, stats in summary.items():
                    if stats['call_count'] > 0:
                        logger.info(
                            f"  {operation}: "
                            f"{stats['call_count']} calls, "
                            f"avg {stats['duration']['mean']:.2f}s"
                        )
                
        else:
            logger.error("Failed to generate plan")
            
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return False
        
    return True


async def main():
    """Run the demo."""
    logger.info("Starting parallel agent demo...\n")
    
    success = await demo_parallel_agent()
    
    if success:
        logger.info("\n✅ Demo completed successfully!")
        logger.info("\nThe parallel orchestrator provides:")
        logger.info("  • Automatic detection of independent steps")
        logger.info("  • Concurrent execution for better performance")
        logger.info("  • Clear visualization of execution strategy")
        logger.info("  • Performance metrics showing speedup achieved")
    else:
        logger.error("\n❌ Demo failed")


if __name__ == "__main__":
    asyncio.run(main())