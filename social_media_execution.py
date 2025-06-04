"""Social Media Analysis Execution Engine

This module provides the actual execution logic for social media analysis scenarios
using DIGIMON's graph building and analysis capabilities.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import tempfile

from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, DynamicToolChainConfig
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from Option.Config2 import Config
from Core.Graph.GraphFactory import get_graph

class SocialMediaAnalysisExecutor:
    """Executes social media analysis scenarios using DIGIMON tools"""
    
    def __init__(self, config_path: str = "Option/Config2.yaml"):
        """Initialize the executor with DIGIMON configuration"""
        self.config = Config.from_yaml(config_path)
        self.graphrag = None
        self.context = None
        self.orchestrator = None
        self.execution_results = {}
        
    async def initialize(self):
        """Initialize DIGIMON components"""
        try:
            # Import necessary components
            from Core.Provider.LLMProviderRegister import llm_provider_register
            from Core.Provider.EnhancedLiteLLMProvider import EnhancedLiteLLMProvider
            from Core.Chunk.ChunkFactory import ChunkFactory
            
            # Register LLM provider
            llm_provider_register.register("litellm", EnhancedLiteLLMProvider)
            
            # Create LLM instance
            llm_class = llm_provider_register.get(self.config.llm.provider)
            self.llm = llm_class(self.config)
            
            # Create encoder instance
            from Core.Index.EmbeddingFactory import EmbeddingFactory
            embedding_factory = EmbeddingFactory(self.config)
            self.encoder = embedding_factory.get_embedding()
            
            # Create chunk factory
            self.chunk_factory = ChunkFactory(self.config)
            
            # Create context
            self.context = GraphRAGContext(
                dataset_name="social_media_analysis",
                config=self.config
            )
            
            # Create orchestrator
            self.orchestrator = AgentOrchestrator(
                main_config=self.config,
                llm_instance=self.llm,
                encoder_instance=self.encoder,
                chunk_factory=self.chunk_factory,
                graphrag_context=self.context
            )
            
            logger.info("Social media analysis executor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize executor: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def prepare_dataset(self, dataset_path: str, dataset_name: str = "covid_conspiracy") -> bool:
        """Prepare the dataset for analysis"""
        try:
            # Create a corpus directory structure
            corpus_dir = Path(f"./social_media_corpus_{dataset_name}")
            corpus_dir.mkdir(exist_ok=True)
            
            # Copy the CSV file to corpus directory
            import shutil
            dest_path = corpus_dir / "dataset.csv"
            if Path(dataset_path).exists():
                shutil.copy(dataset_path, dest_path)
            
            # Create corpus preparation plan
            plan = ExecutionPlan(
                plan_id=f"prepare_{dataset_name}",
                plan_description=f"Prepare corpus from {dataset_path}",
                steps=[
                    ExecutionStep(
                        step_id="prepare_corpus",
                        description="Convert CSV to corpus format",
                        action=DynamicToolChainConfig(
                            tools=[
                                ToolCall(
                                    tool_id="corpus.PrepareFromDirectory",
                                    inputs={
                                        "directory_path": str(corpus_dir),
                                        "corpus_name": dataset_name,
                                        "output_path": str(corpus_dir),
                                        "file_extension": ".csv"
                                    }
                                )
                            ]
                        )
                    )
                ]
            )
            
            # Execute corpus preparation
            results = await self.orchestrator.execute_plan(plan)
            logger.info(f"Corpus preparation results: {results}")
            
            return "prepare_corpus" in results and not results["prepare_corpus"].get("error")
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {str(e)}")
            return False
    
    async def build_graph_for_scenario(self, scenario: Dict[str, Any], dataset_name: str) -> Optional[str]:
        """Build appropriate graph for analysis scenario"""
        try:
            # Determine graph type based on scenario complexity
            complexity = scenario.get("complexity_level", "Simple")
            if complexity == "Simple":
                graph_type = "er_graph"
                tool_id = "graph.BuildERGraph"
            elif complexity == "Medium":
                graph_type = "rk_graph"
                tool_id = "graph.BuildRKGraph"
            else:
                graph_type = "tree_graph_balanced"
                tool_id = "graph.BuildTreeGraphBalanced"
            
            graph_id = f"{dataset_name}_{graph_type}"
            
            # Create graph building plan
            plan = ExecutionPlan(
                plan_id=f"build_{graph_id}",
                plan_description=f"Build {graph_type} for {scenario['title']}",
                steps=[
                    ExecutionStep(
                        step_id="build_graph",
                        description=f"Build {graph_type}",
                        action=DynamicToolChainConfig(
                            tools=[
                                ToolCall(
                                    tool_id=tool_id,
                                    inputs={
                                        "dataset_name": dataset_name,
                                        "use_existing_corpus": True,
                                        "chunk_size": 1024,
                                        "chunk_overlap": 200
                                    }
                                )
                            ]
                        )
                    )
                ]
            )
            
            # Execute graph building
            results = await self.orchestrator.execute_plan(plan)
            
            if "build_graph" in results and not results["build_graph"].get("error"):
                logger.info(f"Successfully built graph: {graph_id}")
                return graph_id
            else:
                logger.error(f"Failed to build graph: {results}")
                return None
                
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            return None
    
    async def analyze_scenario(self, scenario: Dict[str, Any], graph_id: str) -> Dict[str, Any]:
        """Execute analysis for a specific scenario"""
        try:
            analysis_results = {
                "scenario": scenario["title"],
                "research_question": scenario["research_question"],
                "timestamp": datetime.now().isoformat(),
                "insights": [],
                "entities_found": [],
                "relationships_found": [],
                "metrics": {}
            }
            
            # Extract focus areas from interrogative views
            for view in scenario.get("interrogative_views", []):
                interrogative = view["interrogative"]
                focus = view["focus"]
                
                # Create analysis plan based on interrogative type
                if interrogative == "Who":
                    # Search for influential entities
                    plan = self._create_who_analysis_plan(graph_id, view)
                elif interrogative == "What":
                    # Search for dominant topics/narratives
                    plan = self._create_what_analysis_plan(graph_id, view)
                elif interrogative == "When":
                    # Temporal analysis (if timestamps available)
                    plan = self._create_when_analysis_plan(graph_id, view)
                elif interrogative == "How":
                    # Process/mechanism analysis
                    plan = self._create_how_analysis_plan(graph_id, view)
                else:
                    continue
                
                # Execute the analysis plan
                results = await self.orchestrator.execute_plan(plan)
                
                # Extract insights from results
                insights = self._extract_insights_from_results(results, view)
                analysis_results["insights"].extend(insights)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing scenario: {str(e)}")
            return {"error": str(e)}
    
    def _create_who_analysis_plan(self, graph_id: str, view: Dict) -> ExecutionPlan:
        """Create analysis plan for 'Who' interrogative"""
        return ExecutionPlan(
            plan_id=f"who_analysis_{graph_id}",
            plan_description="Identify key influencers and actors",
            steps=[
                ExecutionStep(
                    step_id="build_entity_vdb",
                    description="Build entity vector database",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDB.Build",
                                inputs={
                                    "graph_id": graph_id,
                                    "embed_dim": 768
                                }
                            )
                        ]
                    )
                ),
                ExecutionStep(
                    step_id="search_influencers",
                    description="Search for influential entities",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDBSearch",
                                inputs={
                                    "graph_id": graph_id,
                                    "query": "influential users conspiracy spreaders key actors",
                                    "top_k": 20
                                }
                            )
                        ]
                    )
                ),
                ExecutionStep(
                    step_id="analyze_influence",
                    description="Analyze influence patterns",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="graph.Analyze",
                                inputs={
                                    "graph_id": graph_id,
                                    "analysis_type": "centrality",
                                    "metrics": ["degree", "betweenness", "pagerank"]
                                }
                            )
                        ]
                    )
                )
            ]
        )
    
    def _create_what_analysis_plan(self, graph_id: str, view: Dict) -> ExecutionPlan:
        """Create analysis plan for 'What' interrogative"""
        return ExecutionPlan(
            plan_id=f"what_analysis_{graph_id}",
            plan_description="Identify dominant narratives and topics",
            steps=[
                ExecutionStep(
                    step_id="search_narratives",
                    description="Search for conspiracy narratives",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDBSearch",
                                inputs={
                                    "graph_id": graph_id,
                                    "query": "conspiracy theory narrative bioweapon vaccine control",
                                    "top_k": 30
                                }
                            )
                        ]
                    )
                ),
                ExecutionStep(
                    step_id="extract_relationships",
                    description="Extract narrative relationships",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Relationship.VDB.Search",
                                inputs={
                                    "graph_id": graph_id,
                                    "query": "spreads promotes claims supports",
                                    "top_k": 50
                                }
                            )
                        ]
                    )
                )
            ]
        )
    
    def _create_when_analysis_plan(self, graph_id: str, view: Dict) -> ExecutionPlan:
        """Create analysis plan for 'When' interrogative"""
        return ExecutionPlan(
            plan_id=f"when_analysis_{graph_id}",
            plan_description="Analyze temporal patterns",
            steps=[
                ExecutionStep(
                    step_id="temporal_entities",
                    description="Find time-related entities",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDBSearch",
                                inputs={
                                    "graph_id": graph_id,
                                    "query": "timeline temporal evolution spread progression",
                                    "top_k": 20
                                }
                            )
                        ]
                    )
                )
            ]
        )
    
    def _create_how_analysis_plan(self, graph_id: str, view: Dict) -> ExecutionPlan:
        """Create analysis plan for 'How' interrogative"""
        return ExecutionPlan(
            plan_id=f"how_analysis_{graph_id}",
            plan_description="Analyze mechanisms and processes",
            steps=[
                ExecutionStep(
                    step_id="mechanism_search",
                    description="Search for mechanism patterns",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDBSearch",
                                inputs={
                                    "graph_id": graph_id,
                                    "query": "mechanism process method strategy technique",
                                    "top_k": 25
                                }
                            )
                        ]
                    )
                ),
                ExecutionStep(
                    step_id="process_relationships",
                    description="Extract process relationships",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Relationship.OneHopNeighbors",
                                inputs={
                                    "graph_id": graph_id,
                                    "source_ids": {"from_step_id": "mechanism_search", "named_output_key": "entity_ids"}
                                }
                            )
                        ]
                    )
                )
            ]
        )
    
    def _extract_insights_from_results(self, results: Dict, view: Dict) -> List[Dict]:
        """Extract meaningful insights from analysis results"""
        insights = []
        
        for step_id, step_results in results.items():
            if "error" in step_results:
                continue
                
            # Extract entity search results
            if "similar_entities" in step_results:
                entities = step_results["similar_entities"]
                insight = {
                    "interrogative": view["interrogative"],
                    "focus": view["focus"],
                    "type": "entities",
                    "findings": [
                        {
                            "entity": e.get("entity_name", "Unknown"),
                            "score": e.get("score", 0.0),
                            "description": e.get("description", "")
                        }
                        for e in entities[:5]  # Top 5
                    ]
                }
                insights.append(insight)
            
            # Extract relationship results
            if "relationships" in step_results:
                relationships = step_results["relationships"]
                insight = {
                    "interrogative": view["interrogative"],
                    "focus": view["focus"],
                    "type": "relationships",
                    "findings": [
                        {
                            "source": r.get("source", ""),
                            "target": r.get("target", ""),
                            "type": r.get("relationship_type", ""),
                            "description": r.get("description", "")
                        }
                        for r in relationships[:5]  # Top 5
                    ]
                }
                insights.append(insight)
            
            # Extract metrics/analysis results
            if "metrics" in step_results:
                insight = {
                    "interrogative": view["interrogative"],
                    "focus": view["focus"],
                    "type": "metrics",
                    "findings": step_results["metrics"]
                }
                insights.append(insight)
        
        return insights
    
    async def execute_all_scenarios(self, scenarios: List[Dict], dataset_info: Dict) -> Dict[str, Any]:
        """Execute all analysis scenarios"""
        try:
            # Initialize if not already done
            if not self.orchestrator:
                success = await self.initialize()
                if not success:
                    return {"error": "Failed to initialize DIGIMON"}
            
            # Prepare dataset
            dataset_name = "covid_conspiracy_tweets"
            dataset_path = dataset_info.get("path", "COVID-19-conspiracy-theories-tweets.csv")
            
            success = await self.prepare_dataset(dataset_path, dataset_name)
            if not success:
                return {"error": "Failed to prepare dataset"}
            
            # Execute each scenario
            all_results = {
                "execution_id": datetime.now().isoformat(),
                "dataset": dataset_name,
                "total_scenarios": len(scenarios),
                "scenario_results": []
            }
            
            for i, scenario in enumerate(scenarios):
                logger.info(f"Executing scenario {i+1}/{len(scenarios)}: {scenario['title']}")
                
                # Build appropriate graph
                graph_id = await self.build_graph_for_scenario(scenario, dataset_name)
                if not graph_id:
                    all_results["scenario_results"].append({
                        "scenario": scenario["title"],
                        "error": "Failed to build graph"
                    })
                    continue
                
                # Analyze scenario
                analysis_results = await self.analyze_scenario(scenario, graph_id)
                all_results["scenario_results"].append(analysis_results)
                
                # Save intermediate results
                self._save_results(all_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error executing scenarios: {str(e)}")
            return {"error": str(e)}
    
    def _save_results(self, results: Dict):
        """Save analysis results to file"""
        try:
            output_dir = Path("./social_media_analysis_results")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"analysis_results_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")


# Standalone execution function for testing
async def test_execution():
    """Test the social media analysis execution"""
    executor = SocialMediaAnalysisExecutor()
    
    # Test scenario
    test_scenarios = [{
        "title": "Test Influence Analysis",
        "research_question": "Who are the key influencers?",
        "complexity_level": "Simple",
        "interrogative_views": [{
            "interrogative": "Who",
            "focus": "Key influencers",
            "description": "Identify influential users",
            "entities": ["User", "Influencer"],
            "relationships": ["MENTIONS", "RETWEETS"]
        }]
    }]
    
    dataset_info = {
        "path": "COVID-19-conspiracy-theories-tweets.csv",
        "total_rows": 6591
    }
    
    results = await executor.execute_all_scenarios(test_scenarios, dataset_info)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(test_execution())