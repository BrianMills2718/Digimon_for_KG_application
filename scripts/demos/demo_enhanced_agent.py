#!/usr/bin/env python3
"""
Demo script showing enhanced agent with integrated improvements
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.Common.Logger import logger
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentOrchestrator.enhanced_orchestrator import EnhancedAgentOrchestrator
from Core.AgentSchema.context import GraphRAGContext
from Core.Provider.EnhancedLiteLLMProvider import EnhancedLiteLLMProvider
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

async def demo_enhanced_agent():
    """Demonstrate the enhanced agent with all improvements."""
    
    logger.info("=== Enhanced DIGIMON Agent Demo ===")
    logger.info("Features integrated:")
    logger.info("  ✓ Performance monitoring")
    logger.info("  ✓ Structured error handling")
    logger.info("  ✓ Batch embedding processing")
    logger.info("  ✓ Adaptive timeouts")
    logger.info("  ✓ Enhanced LLM provider\n")
    
    try:
        # Use default config
        config = default_config
        
        # Create enhanced LLM provider
        logger.info("Creating enhanced LLM provider...")
        # Note: EnhancedLiteLLMProvider wraps the base provider
        from Core.Provider.LiteLLMProvider import LiteLLMProvider
        base_llm = LiteLLMProvider(config)
        
        # Create embedding provider
        logger.info("Creating embedding provider...")
        embed_provider = get_rag_embedding(config)
        
        # Create chunk factory
        chunk_factory = ChunkFactory(config)
        
        # Initialize context
        logger.info("Initializing GraphRAG context...")
        context = GraphRAGContext(
            main_config=config,
            embedding_provider=embed_provider
        )
        
        # Create enhanced orchestrator
        logger.info("Creating enhanced orchestrator...")
        orchestrator = EnhancedAgentOrchestrator(
            main_config=config,
            llm_instance=base_llm,  # Will be wrapped internally
            encoder_instance=embed_provider,
            chunk_factory=chunk_factory,
            graphrag_context=context
        )
        
        # Create planning agent
        logger.info("Creating planning agent...")
        agent = PlanningAgent(
            llm=orchestrator.enhanced_llm,  # Use the enhanced LLM
            orchestrator=orchestrator
        )
        
        # Demo query
        test_query = "What are the key themes in the revolutionary documents?"
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
                logger.info(f"  Step {i}: {step.description}")
                if hasattr(step.action, 'tools'):
                    for tool in step.action.tools:
                        logger.info(f"    └─ Tool: {tool.tool_id}")
        
            # Execute plan with enhanced orchestrator
            logger.info("\nExecuting plan with enhanced orchestrator...")
            results = await orchestrator.execute_plan(plan)
            
            logger.info("\n✅ Execution complete!")
            
            # Check for errors
            errors = []
            successful_steps = []
            for step_id, output in results.items():
                if "error" in output:
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
    logger.info("Starting enhanced agent demo...\n")
    
    success = await demo_enhanced_agent()
    
    if success:
        logger.info("\n✅ Demo completed successfully!")
        logger.info("\nThe enhanced agent now includes:")
        logger.info("  • Automatic performance monitoring for all operations")
        logger.info("  • Structured error handling with recovery strategies")
        logger.info("  • Batch embedding processing with deduplication")
        logger.info("  • Adaptive timeouts based on operation complexity")
        logger.info("  • Enhanced LLM provider with retry logic")
    else:
        logger.error("\n❌ Demo failed")

if __name__ == "__main__":
    asyncio.run(main())