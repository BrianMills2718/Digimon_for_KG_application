#!/usr/bin/env python3
"""
Phase 1.1 Test: Enhanced Plan Generation with Mandatory Text Retrieval
Tests that all generated plans include text retrieval steps.
"""

import asyncio
import sys
from pathlib import Path
import json

# Add project root to Python path
sys.path.append('/home/brian/digimon_cc')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Config.LLMConfig import LLMType
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.context import GraphRAGContext
from Core.Chunk.ChunkFactory import ChunkFactory
from Option.Config2 import Config

class PlanGenerationTester:
    def __init__(self):
        """Initialize the testing environment."""
        # Create config
        self.config = Config.parse(
            Path("Option/Method/LGraphRAG.yaml"),
            dataset_name="Fictional_Test",
            exp_name="test_plan_generation"
        )
        
        # Initialize components
        self.llm_provider = LiteLLMProvider(self.config.llm)
        
        # Get embedding provider
        from Core.Index.EmbeddingFactory import get_rag_embedding
        encoder_instance = get_rag_embedding(config=self.config)
        
        self.chunk_factory = ChunkFactory(self.config)
        self.context = GraphRAGContext(
            target_dataset_name="Fictional_Test",
            main_config=self.config,
            llm_provider=self.llm_provider,
            embedding_provider=encoder_instance,
            chunk_storage_manager=self.chunk_factory
        )
        self.orchestrator = AgentOrchestrator(
            main_config=self.config,
            llm_instance=self.llm_provider,
            encoder_instance=encoder_instance,
            chunk_factory=self.chunk_factory,
            graphrag_context=self.context
        )
        
        self.agent = PlanningAgent(
            config=self.config,
            graphrag_context=self.context
        )
        
    async def test_query_plan(self, query: str, test_name: str):
        """Test if a query generates a plan with text retrieval."""
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        try:
            # Generate plan
            plan = await self.agent.generate_plan(query, "Fictional_Test")
            
            if plan:
                # Check if plan includes text retrieval
                has_text_retrieval = False
                text_retrieval_steps = []
                
                for step in plan.steps:
                    for tool in step.action.tools:
                        if tool.tool_id in ["Chunk.GetTextForEntities", "Chunk.FromRelationships"]:
                            has_text_retrieval = True
                            text_retrieval_steps.append({
                                "step_id": step.step_id,
                                "tool": tool.tool_id,
                                "description": step.description
                            })
                
                # Print plan summary
                print(f"\n📋 Generated Plan: {plan.plan_id}")
                print(f"Description: {plan.plan_description}")
                print(f"Total Steps: {len(plan.steps)}")
                
                print("\n🔧 Steps:")
                for i, step in enumerate(plan.steps, 1):
                    tools = [t.tool_id for t in step.action.tools]
                    print(f"  {i}. {step.step_id}: {tools[0] if tools else 'No tool'}")
                    print(f"     Description: {step.description}")
                
                # Check text retrieval
                if has_text_retrieval:
                    print(f"\n✅ SUCCESS: Plan includes text retrieval!")
                    for tr_step in text_retrieval_steps:
                        print(f"   - Step '{tr_step['step_id']}' uses {tr_step['tool']}")
                else:
                    print(f"\n❌ FAILURE: Plan missing text retrieval steps!")
                    print("   Entity discovery found but no Chunk.* tools used")
                
                return has_text_retrieval
            else:
                print(f"\n❌ ERROR: Failed to generate plan")
                return False
                
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Run all plan generation tests."""
    tester = PlanGenerationTester()
    
    # Test queries
    test_cases = [
        {
            "query": "What is crystal technology?",
            "name": "Basic Entity Query",
            "expected": "Should search for crystal technology entity and retrieve its text"
        },
        {
            "query": "How are the Zorathian Empire and crystal technology related?",
            "name": "Relationship Query", 
            "expected": "Should find both entities and retrieve relationship text"
        },
        {
            "query": "Explain the role of Crystal Keepers in the fall of Aerophantis",
            "name": "Complex Query",
            "expected": "Should find multiple entities and retrieve all relevant text"
        },
        {
            "query": "Tell me about levitite crystals",
            "name": "Simple Information Query",
            "expected": "Should find levitite crystal entity and get its description"
        },
        {
            "query": "What caused the Crystal Plague?",
            "name": "Causal Query",
            "expected": "Should find Crystal Plague and related entities with text"
        }
    ]
    
    results = []
    
    print("🧪 PHASE 1.1 TEST: Enhanced Plan Generation")
    print("Testing that all plans include mandatory text retrieval steps")
    
    for test_case in test_cases:
        success = await tester.test_query_plan(
            test_case["query"], 
            test_case["name"]
        )
        results.append({
            "name": test_case["name"],
            "query": test_case["query"],
            "success": success,
            "expected": test_case["expected"]
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} - {result['name']}")
        if not result["success"]:
            print(f"     Expected: {result['expected']}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Plan generation now includes text retrieval.")
    else:
        print("\n⚠️ Some tests failed. Plans still missing text retrieval steps.")

if __name__ == "__main__":
    asyncio.run(main())