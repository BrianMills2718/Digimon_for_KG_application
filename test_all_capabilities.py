#!/usr/bin/env python3
"""Comprehensive test suite for all DIGIMON capabilities"""

import asyncio
import json
import os
import sys
sys.path.append('.')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Common.Logger import logger

class DigimonCapabilityTester:
    def __init__(self):
        # Initialize components
        self.config = Config.default()
        self.llm = create_llm_instance(self.config.llm)
        
        # Create embedding instance
        emb_factory = RAGEmbeddingFactory()
        self.encoder = emb_factory.get_rag_embedding(config=self.config)
        
        self.chunk_factory = ChunkFactory(self.config)
        self.context = GraphRAGContext(
            main_config=self.config,
            target_dataset_name="Russian_Troll_Sample",
            llm_provider=self.llm,
            embedding_provider=self.encoder,
            chunk_storage_manager=self.chunk_factory
        )
        
        self.orchestrator = AgentOrchestrator(
            main_config=self.config,
            llm_instance=self.llm,
            encoder_instance=self.encoder,
            chunk_factory=self.chunk_factory,
            graphrag_context=self.context
        )
        
        self.agent = PlanningAgent(
            config=self.config,
            graphrag_context=self.context
        )
        
        self.results = {}
        
    async def test_capability_1_corpus_preparation(self):
        """Test 1: Corpus preparation from raw text files"""
        print("\n" + "="*60)
        print("TEST 1: Corpus Preparation")
        print("="*60)
        
        query = "Prepare the corpus from the Russian_Troll_Sample directory"
        try:
            result = await self.agent.process_query(query, "Russian_Troll_Sample")
            
            # Check if corpus was prepared
            corpus_path = "results/Russian_Troll_Sample/corpus/Corpus.json"
            if os.path.exists(corpus_path):
                with open(corpus_path, 'r') as f:
                    lines = f.readlines()
                    doc_count = len(lines)
                print(f"✓ SUCCESS: Corpus prepared with {doc_count} documents")
                self.results["corpus_preparation"] = {"status": "success", "doc_count": doc_count}
            else:
                print("✗ FAILURE: Corpus file not created")
                self.results["corpus_preparation"] = {"status": "failure", "error": "Corpus file not found"}
                
        except Exception as e:
            print(f"✗ FAILURE: {str(e)}")
            self.results["corpus_preparation"] = {"status": "failure", "error": str(e)}
            
    async def test_capability_2_graph_construction(self):
        """Test 2: Graph construction (all 5 types)"""
        print("\n" + "="*60)
        print("TEST 2: Graph Construction (All Types)")
        print("="*60)
        
        graph_types = [
            ("ER Graph", "Build an entity-relationship graph for Russian_Troll_Sample"),
            ("RK Graph", "Build a relation-knowledge graph for Russian_Troll_Sample"),
            ("Tree Graph", "Build a tree graph for Russian_Troll_Sample"),
            ("Tree Graph Balanced", "Build a balanced tree graph for Russian_Troll_Sample"),
            ("Passage Graph", "Build a passage graph for Russian_Troll_Sample")
        ]
        
        self.results["graph_construction"] = {}
        
        for graph_name, query in graph_types:
            print(f"\nTesting {graph_name}...")
            try:
                result = await self.agent.process_query(query, "Russian_Troll_Sample")
                
                # Check if graph was built
                answer = result.get("generated_answer", "")
                if "successfully" in answer.lower() or "built" in answer.lower():
                    print(f"✓ SUCCESS: {graph_name} built")
                    self.results["graph_construction"][graph_name] = "success"
                else:
                    print(f"✗ FAILURE: {graph_name} - {answer[:100]}")
                    self.results["graph_construction"][graph_name] = f"failure: {answer[:100]}"
                    
            except Exception as e:
                print(f"✗ FAILURE: {graph_name} - {str(e)}")
                self.results["graph_construction"][graph_name] = f"error: {str(e)}"
                
    async def test_capability_3_entity_search(self):
        """Test 3: Entity search and retrieval"""
        print("\n" + "="*60)
        print("TEST 3: Entity Search and Retrieval")
        print("="*60)
        
        # First ensure ER graph and VDB exist
        setup_query = "Build an ER graph and create entity VDB for Russian_Troll_Sample"
        await self.agent.process_query(setup_query, "Russian_Troll_Sample")
        
        queries = [
            "Find entities related to Trump",
            "Search for entities about protests",
            "What entities are related to fake news"
        ]
        
        self.results["entity_search"] = {}
        
        for query in queries:
            print(f"\nTesting: {query}")
            try:
                result = await self.agent.process_query(query, "Russian_Troll_Sample")
                answer = result.get("generated_answer", "")
                
                if "found" in answer.lower() or "entities" in answer.lower():
                    print(f"✓ SUCCESS: Entities found")
                    self.results["entity_search"][query] = "success"
                else:
                    print(f"? UNCERTAIN: {answer[:100]}")
                    self.results["entity_search"][query] = answer[:100]
                    
            except Exception as e:
                print(f"✗ FAILURE: {str(e)}")
                self.results["entity_search"][query] = f"error: {str(e)}"
                
    async def test_capability_4_relationship_analysis(self):
        """Test 4: Relationship analysis"""
        print("\n" + "="*60)
        print("TEST 4: Relationship Analysis")
        print("="*60)
        
        queries = [
            "What relationships exist between Trump and protests?",
            "Show me one-hop relationships from the entity 'Trump'",
            "Find relationships in the ER graph"
        ]
        
        self.results["relationship_analysis"] = {}
        
        for query in queries:
            print(f"\nTesting: {query}")
            try:
                result = await self.agent.process_query(query, "Russian_Troll_Sample")
                answer = result.get("generated_answer", "")
                
                if "relationship" in answer.lower() or "connected" in answer.lower():
                    print(f"✓ SUCCESS: Relationships found")
                    self.results["relationship_analysis"][query] = "success"
                else:
                    print(f"? UNCERTAIN: {answer[:100]}")
                    self.results["relationship_analysis"][query] = answer[:100]
                    
            except Exception as e:
                print(f"✗ FAILURE: {str(e)}")
                self.results["relationship_analysis"][query] = f"error: {str(e)}"
                
    async def test_capability_5_community_detection(self):
        """Test 5: Community detection and analysis"""
        print("\n" + "="*60)
        print("TEST 5: Community Detection")
        print("="*60)
        
        # This requires specific community detection tools - check if available
        query = "Detect communities in the ER graph for Russian_Troll_Sample"
        
        try:
            result = await self.agent.process_query(query, "Russian_Troll_Sample")
            answer = result.get("generated_answer", "")
            
            if "community" in answer.lower() or "cluster" in answer.lower():
                print("✓ SUCCESS: Community detection attempted")
                self.results["community_detection"] = "success"
            else:
                print(f"? UNCERTAIN: {answer[:100]}")
                self.results["community_detection"] = answer[:100]
                
        except Exception as e:
            print(f"✗ FAILURE: {str(e)}")
            self.results["community_detection"] = f"error: {str(e)}"
            
    async def test_capability_6_chunk_retrieval(self):
        """Test 6: Chunk/text retrieval"""
        print("\n" + "="*60)
        print("TEST 6: Chunk/Text Retrieval")
        print("="*60)
        
        queries = [
            "Get the text content about Trump from the corpus",
            "Retrieve chunks that mention protests",
            "Show me the actual text about fake news"
        ]
        
        self.results["chunk_retrieval"] = {}
        
        for query in queries:
            print(f"\nTesting: {query}")
            try:
                result = await self.agent.process_query(query, "Russian_Troll_Sample")
                answer = result.get("generated_answer", "")
                
                # Check if actual text content is in the answer
                if len(answer) > 200 and ("tweet" in answer.lower() or "author" in answer.lower()):
                    print(f"✓ SUCCESS: Text content retrieved")
                    self.results["chunk_retrieval"][query] = "success"
                else:
                    print(f"? UNCERTAIN: {answer[:100]}")
                    self.results["chunk_retrieval"][query] = answer[:100]
                    
            except Exception as e:
                print(f"✗ FAILURE: {str(e)}")
                self.results["chunk_retrieval"][query] = f"error: {str(e)}"
                
    async def test_capability_7_subgraph_extraction(self):
        """Test 7: Subgraph extraction"""
        print("\n" + "="*60)
        print("TEST 7: Subgraph Extraction")
        print("="*60)
        
        query = "Extract a subgraph around the entity 'Trump' with 2-hop neighbors"
        
        try:
            result = await self.agent.process_query(query, "Russian_Troll_Sample")
            answer = result.get("generated_answer", "")
            
            if "subgraph" in answer.lower() or "neighbors" in answer.lower():
                print("✓ SUCCESS: Subgraph extraction attempted")
                self.results["subgraph_extraction"] = "success"
            else:
                print(f"? UNCERTAIN: {answer[:100]}")
                self.results["subgraph_extraction"] = answer[:100]
                
        except Exception as e:
            print(f"✗ FAILURE: {str(e)}")
            self.results["subgraph_extraction"] = f"error: {str(e)}"
            
    async def test_capability_8_graph_visualization(self):
        """Test 8: Graph visualization"""
        print("\n" + "="*60)
        print("TEST 8: Graph Visualization")
        print("="*60)
        
        query = "Visualize the ER graph for Russian_Troll_Sample"
        
        try:
            result = await self.agent.process_query(query, "Russian_Troll_Sample")
            answer = result.get("generated_answer", "")
            
            # Check if visualization file was created
            viz_files = []
            for ext in ['.png', '.jpg', '.svg', '.html']:
                if os.path.exists(f"results/Russian_Troll_Sample/er_graph/graph_visualization{ext}"):
                    viz_files.append(ext)
                    
            if viz_files:
                print(f"✓ SUCCESS: Visualization created ({', '.join(viz_files)})")
                self.results["graph_visualization"] = f"success: {', '.join(viz_files)}"
            else:
                print(f"? UNCERTAIN: {answer[:100]}")
                self.results["graph_visualization"] = answer[:100]
                
        except Exception as e:
            print(f"✗ FAILURE: {str(e)}")
            self.results["graph_visualization"] = f"error: {str(e)}"
            
    async def test_capability_9_multi_step_reasoning(self):
        """Test 9: Multi-step reasoning"""
        print("\n" + "="*60)
        print("TEST 9: Multi-step Reasoning")
        print("="*60)
        
        query = "First find entities about Trump, then get their relationships, and finally retrieve the text content for those relationships"
        
        try:
            result = await self.agent.process_query(query, "Russian_Troll_Sample")
            answer = result.get("generated_answer", "")
            
            # Check if the answer shows multiple steps were executed
            context = result.get("retrieved_context", {})
            steps_executed = len(context)
            
            if steps_executed >= 3:
                print(f"✓ SUCCESS: Multi-step reasoning with {steps_executed} steps")
                self.results["multi_step_reasoning"] = f"success: {steps_executed} steps"
            else:
                print(f"? PARTIAL: Only {steps_executed} steps executed")
                self.results["multi_step_reasoning"] = f"partial: {steps_executed} steps"
                
        except Exception as e:
            print(f"✗ FAILURE: {str(e)}")
            self.results["multi_step_reasoning"] = f"error: {str(e)}"
            
    async def test_capability_10_react_mode(self):
        """Test 10: ReAct mode (iterative reasoning)"""
        print("\n" + "="*60)
        print("TEST 10: ReAct Mode")
        print("="*60)
        
        query = "Analyze the discourse about protests in the Russian troll tweets"
        
        try:
            result = await self.agent.process_query_react(query, "Russian_Troll_Sample")
            
            iterations = result.get("iterations", 0)
            executed_steps = result.get("executed_steps", [])
            
            if iterations > 1 and len(executed_steps) > 0:
                print(f"✓ SUCCESS: ReAct mode with {iterations} iterations, {len(executed_steps)} steps")
                self.results["react_mode"] = f"success: {iterations} iterations"
            else:
                print(f"? PARTIAL: ReAct mode with {iterations} iterations")
                self.results["react_mode"] = f"partial: {iterations} iterations"
                
        except Exception as e:
            print(f"✗ FAILURE: {str(e)}")
            self.results["react_mode"] = f"error: {str(e)}"
            
    async def test_capability_11_query_expansion(self):
        """Test 11: Query expansion and refinement"""
        print("\n" + "="*60)
        print("TEST 11: Query Expansion")
        print("="*60)
        
        query = "Tell me about political polarization"  # Vague query that needs expansion
        
        try:
            result = await self.agent.process_query(query, "Russian_Troll_Sample")
            answer = result.get("generated_answer", "")
            
            # Check if the system expanded the query to search for related concepts
            if len(answer) > 100 and any(term in answer.lower() for term in ["democrat", "republican", "trump", "liberal", "conservative"]):
                print("✓ SUCCESS: Query expanded to find relevant content")
                self.results["query_expansion"] = "success"
            else:
                print(f"? UNCERTAIN: {answer[:100]}")
                self.results["query_expansion"] = answer[:100]
                
        except Exception as e:
            print(f"✗ FAILURE: {str(e)}")
            self.results["query_expansion"] = f"error: {str(e)}"
            
    async def test_capability_12_discourse_analysis(self):
        """Test 12: Discourse analysis"""
        print("\n" + "="*60)
        print("TEST 12: Discourse Analysis")
        print("="*60)
        
        query = "Perform a discourse analysis on how Russian trolls discuss the NFL protests"
        
        try:
            result = await self.agent.process_query(query, "Russian_Troll_Sample")
            answer = result.get("generated_answer", "")
            
            # Check if discourse analysis concepts are present
            discourse_terms = ["narrative", "framing", "rhetoric", "discourse", "theme", "pattern"]
            if any(term in answer.lower() for term in discourse_terms):
                print("✓ SUCCESS: Discourse analysis performed")
                self.results["discourse_analysis"] = "success"
            else:
                print(f"? UNCERTAIN: {answer[:100]}")
                self.results["discourse_analysis"] = answer[:100]
                
        except Exception as e:
            print(f"✗ FAILURE: {str(e)}")
            self.results["discourse_analysis"] = f"error: {str(e)}"
            
    async def run_all_tests(self):
        """Run all capability tests"""
        print("\n" + "="*70)
        print("DIGIMON COMPREHENSIVE CAPABILITY TEST SUITE")
        print("="*70)
        
        # Run tests in order
        await self.test_capability_1_corpus_preparation()
        await self.test_capability_2_graph_construction()
        await self.test_capability_3_entity_search()
        await self.test_capability_4_relationship_analysis()
        await self.test_capability_5_community_detection()
        await self.test_capability_6_chunk_retrieval()
        await self.test_capability_7_subgraph_extraction()
        await self.test_capability_8_graph_visualization()
        await self.test_capability_9_multi_step_reasoning()
        await self.test_capability_10_react_mode()
        await self.test_capability_11_query_expansion()
        await self.test_capability_12_discourse_analysis()
        
        # Generate summary report
        self.generate_report()
        
    def generate_report(self):
        """Generate final test report"""
        print("\n" + "="*70)
        print("FINAL TEST REPORT")
        print("="*70)
        
        # Count successes
        total_tests = 0
        successful_tests = 0
        
        for capability, result in self.results.items():
            if isinstance(result, dict):
                for sub_test, status in result.items():
                    total_tests += 1
                    if isinstance(status, str) and "success" in status:
                        successful_tests += 1
            else:
                total_tests += 1
                if isinstance(result, str) and "success" in result:
                    successful_tests += 1
                    
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nOverall Success Rate: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        
        # Detailed results
        print("\nDetailed Results:")
        for capability, result in self.results.items():
            print(f"\n{capability}:")
            if isinstance(result, dict):
                for sub_test, status in result.items():
                    status_icon = "✓" if "success" in str(status) else "✗"
                    print(f"  {status_icon} {sub_test}: {status}")
            else:
                status_icon = "✓" if "success" in str(result) else "✗"
                print(f"  {status_icon} {result}")
                
        # Save results to file
        with open("test_results.json", "w") as f:
            json.dump({
                "timestamp": str(asyncio.get_event_loop().time()),
                "success_rate": f"{success_rate:.1f}%",
                "results": self.results
            }, f, indent=2)
        print("\nResults saved to test_results.json")

async def main():
    tester = DigimonCapabilityTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())