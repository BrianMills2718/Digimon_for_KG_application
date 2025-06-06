"""Social Media Analysis Execution Engine with Full Tracing

This module provides real execution with detailed tracing of all operations.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import tempfile

from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, DynamicToolChainConfig
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from Option.Config2 import Config
from Core.Graph.GraphFactory import get_graph

class TracedSocialMediaAnalysisExecutor:
    """Executes social media analysis with detailed execution tracing"""
    
    def __init__(self, config_path: str = "Option/Config2.yaml", trace_callback: Optional[Callable] = None):
        """Initialize with optional trace callback"""
        try:
            # Try to load config
            self.config = Config.from_yaml_file(config_path)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}, using default")
            self.config = Config.default()
        
        self.llm = None
        self.encoder = None
        self.chunk_factory = None
        self.context = None
        self.orchestrator = None
        self.trace_callback = trace_callback or (lambda event_type, data: None)
        self.execution_results = {}
        
    def _trace(self, event_type: str, data: Dict[str, Any]):
        """Send trace event"""
        try:
            self.trace_callback(event_type, data)
        except Exception as e:
            logger.error(f"Trace callback error: {e}")
    
    async def initialize(self):
        """Initialize DIGIMON components with tracing"""
        try:
            self._trace("init_start", {"component": "DIGIMON"})
            
            # Import necessary components
            from Core.Provider.LLMProviderRegister import LLM_REGISTRY, create_llm_instance
            from Core.Provider.EnhancedLiteLLMProvider import EnhancedLiteLLMProvider
            from Core.Chunk.ChunkFactory import ChunkFactory
            from Config.LLMConfig import LLMType
            
            # Register LLM provider
            self._trace("init_step", {"step": "Registering LLM provider"})
            LLM_REGISTRY.register(LLMType.LITELLM, EnhancedLiteLLMProvider)
            
            # Create LLM instance
            self._trace("init_step", {"step": "Creating LLM instance"})
            self.llm = create_llm_instance(self.config.llm)
            
            # Create encoder instance
            self._trace("init_step", {"step": "Creating encoder instance"})
            from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
            from Config.EmbConfig import EmbeddingType
            embedding_factory = RAGEmbeddingFactory()
            self.encoder = embedding_factory.get_rag_embedding(EmbeddingType.OPENAI, self.config)
            
            # Create chunk factory
            self._trace("init_step", {"step": "Creating chunk factory"})
            self.chunk_factory = ChunkFactory(self.config)
            
            # Create context
            self._trace("init_step", {"step": "Creating GraphRAG context"})
            self.context = GraphRAGContext(
                target_dataset_name="social_media_analysis",
                main_config=self.config,
                llm_provider=self.llm,
                embedding_provider=self.encoder,
                chunk_storage_manager=self.chunk_factory
            )
            
            # Create orchestrator
            self._trace("init_step", {"step": "Creating orchestrator"})
            self.orchestrator = AgentOrchestrator(
                main_config=self.config,
                llm_instance=self.llm,
                encoder_instance=self.encoder,
                chunk_factory=self.chunk_factory,
                graphrag_context=self.context
            )
            
            self._trace("init_complete", {"status": "success"})
            logger.info("Social media analysis executor initialized successfully")
            return True
            
        except Exception as e:
            self._trace("init_failed", {"error": str(e)})
            logger.error(f"Failed to initialize executor: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def prepare_dataset(self, dataset_path: str, dataset_name: str = "covid_conspiracy") -> bool:
        """Prepare the dataset for analysis with tracing"""
        try:
            self._trace("dataset_prep_start", {"path": dataset_path, "name": dataset_name})
            
            # Create a corpus directory structure
            corpus_dir = Path(f"./social_media_corpus_{dataset_name}")
            corpus_dir.mkdir(exist_ok=True)
            
            # For CSV files, we need to convert to text format
            if dataset_path.endswith('.csv'):
                self._trace("dataset_prep_step", {"step": "Converting CSV to corpus"})
                
                import pandas as pd
                df = pd.read_csv(dataset_path)
                
                # Create individual text files for chunks of tweets
                chunk_size = 100  # tweets per file
                for i in range(0, len(df), chunk_size):
                    chunk_df = df.iloc[i:i+chunk_size]
                    
                    # Combine tweets into a document
                    doc_text = ""
                    for _, row in chunk_df.iterrows():
                        doc_text += f"Tweet ID: {row.get('tweet_id', i)}\n"
                        doc_text += f"Tweet: {row['tweet']}\n"
                        doc_text += f"Conspiracy Type: {row.get('conspiracy_theory', 'Unknown')}\n"
                        doc_text += f"Label: {row.get('label', 'Unknown')}\n"
                        doc_text += "---\n\n"
                    
                    # Save as text file
                    doc_path = corpus_dir / f"tweets_chunk_{i//chunk_size}.txt"
                    with open(doc_path, 'w', encoding='utf-8') as f:
                        f.write(doc_text)
                    
                    self._trace("dataset_prep_progress", {
                        "files_created": i//chunk_size + 1,
                        "total_tweets": min(i+chunk_size, len(df))
                    })
            
            # Create corpus preparation plan
            plan = ExecutionPlan(
                plan_id=f"prepare_{dataset_name}",
                plan_description=f"Prepare corpus from {corpus_dir}",
                steps=[
                    ExecutionStep(
                        step_id="prepare_corpus",
                        description="Convert text files to corpus format",
                        action=DynamicToolChainConfig(
                            tools=[
                                ToolCall(
                                    tool_id="corpus.PrepareFromDirectory",
                                    inputs={
                                        "directory_path": str(corpus_dir),
                                        "corpus_name": dataset_name,
                                        "output_path": str(corpus_dir)
                                    }
                                )
                            ]
                        )
                    )
                ]
            )
            
            # Trace the execution plan
            self._trace("execution_plan", {
                "plan_id": plan.plan_id,
                "steps": [{"id": s.step_id, "description": s.description} for s in plan.steps]
            })
            
            # Execute corpus preparation
            self._trace("tool_execution_start", {"tool": "corpus.PrepareFromDirectory"})
            results = await self.orchestrator.execute_plan(plan)
            self._trace("tool_execution_complete", {"tool": "corpus.PrepareFromDirectory", "results": str(results)})
            
            success = "prepare_corpus" in results and not results["prepare_corpus"].get("error")
            self._trace("dataset_prep_complete", {"status": "success" if success else "failed"})
            
            return success
            
        except Exception as e:
            self._trace("dataset_prep_failed", {"error": str(e)})
            logger.error(f"Failed to prepare dataset: {str(e)}")
            return False
    
    async def build_graph_for_scenario(self, scenario: Dict[str, Any], dataset_name: str) -> Optional[str]:
        """Build appropriate graph for analysis scenario with detailed tracing"""
        try:
            self._trace("graph_build_start", {
                "scenario": scenario["title"],
                "complexity": scenario.get("complexity_level", "Simple")
            })
            
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
            
            self._trace("graph_type_selected", {
                "graph_type": graph_type,
                "graph_id": graph_id,
                "tool_id": tool_id
            })
            
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
                                        "chunk_size": 512,  # Smaller chunks for tweets
                                        "chunk_overlap": 100
                                    }
                                )
                            ]
                        )
                    )
                ]
            )
            
            # Trace the execution plan
            self._trace("execution_plan", {
                "plan_id": plan.plan_id,
                "graph_type": graph_type,
                "tool_id": tool_id
            })
            
            # Execute graph building
            self._trace("tool_execution_start", {"tool": tool_id, "graph_id": graph_id})
            results = await self.orchestrator.execute_plan(plan)
            
            if "build_graph" in results and not results["build_graph"].get("error"):
                self._trace("graph_build_complete", {
                    "graph_id": graph_id,
                    "status": "success"
                })
                logger.info(f"Successfully built graph: {graph_id}")
                return graph_id
            else:
                self._trace("graph_build_failed", {
                    "graph_id": graph_id,
                    "error": str(results)
                })
                logger.error(f"Failed to build graph: {results}")
                return None
                
        except Exception as e:
            self._trace("graph_build_error", {"error": str(e)})
            logger.error(f"Error building graph: {str(e)}")
            return None
    
    async def analyze_scenario(self, scenario: Dict[str, Any], graph_id: str) -> Dict[str, Any]:
        """Execute analysis for a specific scenario with detailed tracing"""
        try:
            self._trace("scenario_analysis_start", {
                "scenario": scenario["title"],
                "graph_id": graph_id
            })
            
            analysis_results = {
                "scenario": scenario["title"],
                "research_question": scenario["research_question"],
                "timestamp": datetime.now().isoformat(),
                "insights": [],
                "entities_found": [],
                "relationships_found": [],
                "metrics": {},
                "execution_trace": []
            }
            
            # Build entity VDB first (required for searches)
            self._trace("building_entity_vdb", {"graph_id": graph_id})
            vdb_plan = ExecutionPlan(
                plan_id=f"build_vdb_{graph_id}",
                plan_description="Build entity vector database",
                steps=[
                    ExecutionStep(
                        step_id="build_entity_vdb",
                        description="Build entity VDB for searches",
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
                    )
                ]
            )
            
            vdb_results = await self.orchestrator.execute_plan(vdb_plan)
            self._trace("vdb_build_complete", {"results": str(vdb_results)})
            
            # Extract focus areas from interrogative views
            for view_idx, view in enumerate(scenario.get("interrogative_views", [])):
                interrogative = view["interrogative"]
                focus = view["focus"]
                
                self._trace("analyzing_view", {
                    "interrogative": interrogative,
                    "focus": focus,
                    "view_index": view_idx
                })
                
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
                
                # Trace the plan
                self._trace("view_execution_plan", {
                    "interrogative": interrogative,
                    "plan_id": plan.plan_id,
                    "steps": len(plan.steps)
                })
                
                # Execute the analysis plan
                self._trace("view_execution_start", {"interrogative": interrogative})
                results = await self.orchestrator.execute_plan(plan)
                self._trace("view_execution_complete", {
                    "interrogative": interrogative,
                    "results_keys": list(results.keys())
                })
                
                # Extract insights from results
                insights = self._extract_insights_from_results(results, view)
                analysis_results["insights"].extend(insights)
                
                # Trace insights found
                self._trace("insights_extracted", {
                    "interrogative": interrogative,
                    "insight_count": len(insights)
                })
            
            # Add execution trace to results
            analysis_results["execution_trace"].append({
                "timestamp": datetime.now().isoformat(),
                "graph_id": graph_id,
                "views_analyzed": len(scenario.get("interrogative_views", []))
            })
            
            self._trace("scenario_analysis_complete", {
                "scenario": scenario["title"],
                "insights_found": len(analysis_results["insights"])
            })
            
            return analysis_results
            
        except Exception as e:
            self._trace("scenario_analysis_error", {"error": str(e)})
            logger.error(f"Error analyzing scenario: {str(e)}")
            return {"error": str(e)}
    
    def _create_who_analysis_plan(self, graph_id: str, view: Dict) -> ExecutionPlan:
        """Create analysis plan for 'Who' interrogative"""
        return ExecutionPlan(
            plan_id=f"who_analysis_{graph_id}_{int(datetime.now().timestamp())}",
            plan_description="Identify key influencers and actors",
            steps=[
                ExecutionStep(
                    step_id="search_influencers",
                    description="Search for influential entities",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDBSearch",
                                inputs={
                                    "graph_id": graph_id,
                                    "query": "user influencer spreader key actor person account",
                                    "top_k": 20
                                },
                                named_outputs={
                                    "found_entities": "similar_entities"
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
            plan_id=f"what_analysis_{graph_id}_{int(datetime.now().timestamp())}",
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
                                    "query": "conspiracy theory narrative bioweapon vaccine control misinformation",
                                    "top_k": 30
                                },
                                named_outputs={
                                    "narrative_entities": "similar_entities"
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
            plan_id=f"when_analysis_{graph_id}_{int(datetime.now().timestamp())}",
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
                                    "query": "time date period trend evolution spread timeline",
                                    "top_k": 20
                                },
                                named_outputs={
                                    "temporal_entities": "similar_entities"
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
            plan_id=f"how_analysis_{graph_id}_{int(datetime.now().timestamp())}",
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
                                    "query": "mechanism process method hashtag retweet share spread amplify",
                                    "top_k": 25
                                },
                                named_outputs={
                                    "mechanism_entities": "similar_entities"
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
                
            # Handle different output keys
            entities_key = None
            for key in ["found_entities", "narrative_entities", "temporal_entities", "mechanism_entities", "similar_entities"]:
                if key in step_results:
                    entities_key = key
                    break
            
            if entities_key and step_results[entities_key]:
                entities = step_results[entities_key]
                
                # Handle both list and object responses
                if hasattr(entities, 'similar_entities'):
                    entities = entities.similar_entities
                
                insight = {
                    "interrogative": view["interrogative"],
                    "focus": view["focus"],
                    "type": "entities",
                    "findings": []
                }
                
                # Extract top entities
                for e in entities[:5]:  # Top 5
                    if isinstance(e, dict):
                        finding = {
                            "entity": e.get("entity_name", e.get("node_id", "Unknown")),
                            "score": float(e.get("score", 0.0)),
                            "description": e.get("description", "")
                        }
                    else:
                        # Handle other formats
                        finding = {
                            "entity": str(e),
                            "score": 0.0,
                            "description": ""
                        }
                    insight["findings"].append(finding)
                
                insights.append(insight)
                
                # Trace findings
                self._trace("insights_found", {
                    "interrogative": view["interrogative"],
                    "entity_count": len(entities),
                    "top_entities": [f["entity"] for f in insight["findings"]]
                })
        
        return insights
    
    async def execute_all_scenarios(self, scenarios: List[Dict], dataset_info: Dict) -> Dict[str, Any]:
        """Execute all analysis scenarios with full tracing"""
        try:
            self._trace("execution_start", {
                "scenario_count": len(scenarios),
                "dataset": dataset_info
            })
            
            # Update progress
            self._trace("progress", {"percent": 0, "message": "Initializing DIGIMON..."})
            
            # Initialize if not already done
            if not self.orchestrator:
                success = await self.initialize()
                if not success:
                    return {"error": "Failed to initialize DIGIMON"}
            
            # Update progress
            self._trace("progress", {"percent": 10, "message": "Preparing dataset..."})
            
            # Prepare dataset
            dataset_name = "covid_conspiracy_tweets"
            dataset_path = dataset_info.get("path", "COVID-19-conspiracy-theories-tweets.csv")
            
            success = await self.prepare_dataset(dataset_path, dataset_name)
            if not success:
                return {"error": "Failed to prepare dataset"}
            
            # Update progress
            self._trace("progress", {"percent": 20, "message": "Starting scenario analysis..."})
            
            # Execute each scenario
            all_results = {
                "execution_id": datetime.now().isoformat(),
                "dataset": dataset_name,
                "total_scenarios": len(scenarios),
                "scenario_results": [],
                "execution_summary": {
                    "total_graphs_built": 0,
                    "total_entities_found": 0,
                    "total_insights_generated": 0
                }
            }
            
            progress_per_scenario = 70 / len(scenarios)  # 70% for all scenarios
            
            for i, scenario in enumerate(scenarios):
                scenario_progress = 20 + (i * progress_per_scenario)
                
                self._trace("progress", {
                    "percent": int(scenario_progress),
                    "message": f"Analyzing scenario {i+1}/{len(scenarios)}: {scenario['title']}"
                })
                
                logger.info(f"Executing scenario {i+1}/{len(scenarios)}: {scenario['title']}")
                
                # Build appropriate graph
                self._trace("progress", {
                    "percent": int(scenario_progress + progress_per_scenario * 0.3),
                    "message": f"Building graph for: {scenario['title']}"
                })
                
                graph_id = await self.build_graph_for_scenario(scenario, dataset_name)
                if not graph_id:
                    all_results["scenario_results"].append({
                        "scenario": scenario["title"],
                        "error": "Failed to build graph"
                    })
                    continue
                
                all_results["execution_summary"]["total_graphs_built"] += 1
                
                # Analyze scenario
                self._trace("progress", {
                    "percent": int(scenario_progress + progress_per_scenario * 0.7),
                    "message": f"Executing analysis for: {scenario['title']}"
                })
                
                analysis_results = await self.analyze_scenario(scenario, graph_id)
                all_results["scenario_results"].append(analysis_results)
                
                # Update summary
                if "insights" in analysis_results:
                    all_results["execution_summary"]["total_insights_generated"] += len(analysis_results["insights"])
                    for insight in analysis_results["insights"]:
                        if "findings" in insight:
                            all_results["execution_summary"]["total_entities_found"] += len(insight["findings"])
                
                # Save intermediate results
                self._save_results(all_results)
            
            # Final progress
            self._trace("progress", {"percent": 100, "message": "Analysis complete!"})
            
            self._trace("execution_complete", {
                "total_scenarios": len(scenarios),
                "summary": all_results["execution_summary"]
            })
            
            return all_results
            
        except Exception as e:
            self._trace("execution_error", {"error": str(e)})
            logger.error(f"Error executing scenarios: {str(e)}")
            import traceback
            traceback.print_exc()
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
            
            self._trace("results_saved", {"path": str(output_file)})
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            self._trace("save_error", {"error": str(e)})
            logger.error(f"Failed to save results: {str(e)}")


# Test function
async def test_traced_execution():
    """Test the traced execution"""
    def print_trace(event_type, data):
        print(f"[TRACE] {event_type}: {json.dumps(data, indent=2)}")
    
    executor = TracedSocialMediaAnalysisExecutor(trace_callback=print_trace)
    
    test_scenarios = [{
        "title": "Test Traced Analysis",
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
        "total_rows": 100
    }
    
    results = await executor.execute_all_scenarios(test_scenarios, dataset_info)
    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(test_traced_execution())