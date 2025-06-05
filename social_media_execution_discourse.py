"""Social Media Analysis with Discourse Framework Integration

This module provides a discourse-analysis-aware execution engine that uses
the five interrogatives framework (Who/What/To Whom/In What Setting/With What Effect)
to guide sophisticated social media analysis.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, DynamicToolChainConfig
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from Option.Config2 import Config
from Core.AgentTools.discourse_enhanced_planner import (
    DiscourseEnhancedPlanner,
    DiscourseAnalysisScenario,
    DiscourseInterrogativeView
)
from Core.AgentTools.discourse_analysis_prompts import (
    DISCOURSE_ANALYSIS_SYSTEM_PROMPT,
    generate_analysis_prompt
)


class DiscourseEnhancedSocialMediaExecutor:
    """Executes social media analysis using discourse analysis framework
    
    This executor understands and applies the five interrogatives:
    - Who: Actors, influencers, communities
    - Says What: Narratives, claims, themes
    - To Whom: Audiences, targets, recipients
    - In What Setting: Platforms, contexts, conditions
    - With What Effect: Outcomes, impacts, responses
    """
    
    def __init__(self, config_path: str = "Option/Config2.yaml", trace_callback: Optional[Callable] = None):
        """Initialize with discourse analysis capabilities"""
        try:
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
        
        # Initialize discourse planner
        self.discourse_planner = DiscourseEnhancedPlanner()
        
    def _trace(self, event_type: str, data: Dict[str, Any]):
        """Send trace event"""
        try:
            self.trace_callback(event_type, data)
        except Exception as e:
            logger.error(f"Trace callback error: {e}")
    
    async def initialize(self):
        """Initialize DIGIMON with discourse analysis context"""
        try:
            self._trace("init_start", {
                "component": "DIGIMON",
                "mode": "discourse_enhanced"
            })
            
            # Import necessary components
            from Core.Provider.LLMProviderRegister import LLM_REGISTRY, create_llm_instance
            from Core.Provider.EnhancedLiteLLMProvider import EnhancedLiteLLMProvider
            from Core.Chunk.ChunkFactory import ChunkFactory
            from Config.LLMConfig import LLMType
            
            # Register LLM provider
            self._trace("init_step", {"step": "Registering LLM provider"})
            LLM_REGISTRY.register(LLMType.LITELLM, EnhancedLiteLLMProvider)
            
            # Create LLM instance with discourse context
            self._trace("init_step", {"step": "Creating discourse-aware LLM"})
            self.llm = create_llm_instance(self.config.llm)
            
            # Set discourse system prompt
            if hasattr(self.llm, 'set_system_prompt'):
                self.llm.set_system_prompt(DISCOURSE_ANALYSIS_SYSTEM_PROMPT)
            
            # Create encoder
            self._trace("init_step", {"step": "Creating encoder"})
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
                target_dataset_name="social_media_discourse_analysis",
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
            
            self._trace("init_complete", {
                "status": "success",
                "discourse_enabled": True
            })
            return True
            
        except Exception as e:
            self._trace("init_failed", {"error": str(e)})
            logger.error(f"Failed to initialize: {str(e)}")
            return False
    
    async def generate_discourse_scenarios(self, research_focus: str) -> List[DiscourseAnalysisScenario]:
        """Generate discourse-based analysis scenarios"""
        self._trace("scenario_generation_start", {
            "research_focus": research_focus,
            "using_discourse_framework": True
        })
        
        # Generate scenarios using discourse planner
        scenarios = self.discourse_planner.generate_scenarios(
            research_focus=research_focus,
            num_scenarios=5
        )
        
        self._trace("scenario_generation_complete", {
            "scenarios_generated": len(scenarios),
            "total_views": sum(len(s.interrogative_views) for s in scenarios),
            "complexity_distribution": {
                "simple": sum(1 for s in scenarios if s.complexity_level == "Simple"),
                "medium": sum(1 for s in scenarios if s.complexity_level == "Medium"),
                "complex": sum(1 for s in scenarios if s.complexity_level == "Complex")
            }
        })
        
        return scenarios
    
    async def execute_discourse_analysis(self, scenario: DiscourseAnalysisScenario, graph_id: str) -> Dict[str, Any]:
        """Execute a discourse-based analysis scenario"""
        try:
            self._trace("discourse_analysis_start", {
                "scenario": scenario.title,
                "research_question": scenario.research_question,
                "views": len(scenario.interrogative_views),
                "unified_ontology_size": len(scenario.unified_ontology.get("entities", [])) + 
                                        len(scenario.unified_ontology.get("relationships", []))
            })
            
            analysis_results = {
                "scenario": scenario.title,
                "research_question": scenario.research_question,
                "timestamp": datetime.now().isoformat(),
                "discourse_framework": {
                    "interrogative_views": scenario.interrogative_views,
                    "mini_ontologies": scenario.mini_ontologies,
                    "unified_ontology": scenario.unified_ontology
                },
                "insights": [],
                "entities_by_interrogative": {},
                "relationships_by_interrogative": {},
                "cross_interrogative_patterns": [],
                "execution_trace": []
            }
            
            # Build specialized VDBs for discourse analysis
            await self._build_discourse_vdbs(graph_id, scenario)
            
            # Execute each interrogative view
            for view_idx, view in enumerate(scenario.interrogative_views):
                self._trace("analyzing_interrogative", {
                    "view_index": view_idx,
                    "interrogative": view.interrogative,
                    "focus": view.focus,
                    "entities_focus": view.entities,
                    "relationships_focus": view.relationships
                })
                
                # Execute view-specific retrieval chains
                view_results = await self._execute_interrogative_view(
                    view, graph_id, scenario.retrieval_chains[view_idx]
                )
                
                # Store results by interrogative
                analysis_results["entities_by_interrogative"][view.interrogative] = view_results.get("entities", [])
                analysis_results["relationships_by_interrogative"][view.interrogative] = view_results.get("relationships", [])
                
                # Generate discourse-aware insights
                insights = await self._generate_discourse_insights(view, view_results)
                analysis_results["insights"].extend(insights)
            
            # Apply transformation chains for cross-interrogative analysis
            cross_patterns = await self._apply_discourse_transformations(
                analysis_results, scenario.transformation_chains
            )
            analysis_results["cross_interrogative_patterns"] = cross_patterns
            
            # Generate synthesis
            synthesis = await self._synthesize_discourse_findings(analysis_results)
            analysis_results["synthesis"] = synthesis
            
            self._trace("discourse_analysis_complete", {
                "scenario": scenario.title,
                "total_insights": len(analysis_results["insights"]),
                "cross_patterns": len(cross_patterns),
                "synthesis_generated": bool(synthesis)
            })
            
            return analysis_results
            
        except Exception as e:
            self._trace("discourse_analysis_error", {"error": str(e)})
            logger.error(f"Discourse analysis error: {str(e)}")
            return {"error": str(e)}
    
    async def _build_discourse_vdbs(self, graph_id: str, scenario: DiscourseAnalysisScenario):
        """Build specialized VDBs for discourse analysis"""
        self._trace("building_discourse_vdbs", {
            "graph_id": graph_id,
            "ontology_entities": len(scenario.unified_ontology.get("entities", [])),
            "ontology_relationships": len(scenario.unified_ontology.get("relationships", []))
        })
        
        # Build entity VDB
        vdb_plan = ExecutionPlan(
            plan_id=f"build_discourse_vdb_{graph_id}",
            plan_description="Build discourse-focused vector databases",
            target_dataset_name=graph_id,
            steps=[
                ExecutionStep(
                    step_id="build_entity_vdb",
                    description="Build entity VDB with discourse focus",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDB.Build",
                                inputs={
                                    "graph_id": graph_id,
                                    "embed_dim": 768,
                                    "focus_entities": scenario.unified_ontology.get("entities", [])
                                }
                            )
                        ]
                    )
                ),
                ExecutionStep(
                    step_id="build_relationship_vdb",
                    description="Build relationship VDB with discourse focus",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Relationship.VDB.Build",
                                inputs={
                                    "graph_id": graph_id,
                                    "embed_dim": 768,
                                    "focus_relationships": scenario.unified_ontology.get("relationships", [])
                                }
                            )
                        ]
                    )
                )
            ]
        )
        
        results = await self.orchestrator.execute_plan(vdb_plan)
        self._trace("discourse_vdbs_built", {"results": str(results)})
    
    async def _execute_interrogative_view(self, view, graph_id: str, retrieval_chain: List[Dict]) -> Dict:
        """Execute retrieval for a specific interrogative view"""
        view_results = {
            "interrogative": view.interrogative,
            "focus": view.focus,
            "entities": [],
            "relationships": [],
            "chunks": []
        }
        
        # Convert retrieval chain to execution steps
        steps = []
        for chain_step in retrieval_chain:
            operator = chain_step.get("operator", "")
            params = chain_step.get("parameters", {})
            
            # Add graph_id
            params["graph_id"] = graph_id
            
            # Map to DIGIMON tools based on operator
            if operator == "by_ppr":
                # Use PPR for influence analysis
                tool_id = "Entity.PPRSearch"
                params["seed_entities"] = view.entities[:3]  # Top entities as seeds
            elif operator == "by_vdb":
                # Vector search with interrogative focus
                tool_id = "Entity.VDBSearch"
                params["query"] = " ".join(view.entities + view.relationships)
            elif operator == "by_relationship":
                # Relationship search
                tool_id = "Relationship.Search"
                params["relationship_types"] = view.relationships
            elif operator == "entity_occurrence":
                # Find chunks containing entities
                tool_id = "Chunk.EntityOccurrence"
                params["entity_names"] = view.entities
            else:
                continue
            
            steps.append(ExecutionStep(
                step_id=f"{view.interrogative}_{operator}_{len(steps)}",
                description=f"{view.interrogative} analysis using {operator}",
                action=DynamicToolChainConfig(
                    tools=[ToolCall(
                        tool_id=tool_id,
                        inputs=params,
                        named_outputs={
                            f"{view.interrogative}_results": "results"
                        }
                    )]
                )
            ))
        
        # Execute the plan
        plan = ExecutionPlan(
            plan_id=f"interrogative_{view.interrogative}_{graph_id}",
            plan_description=f"Execute {view.interrogative} interrogative analysis",
            target_dataset_name=graph_id,
            steps=steps
        )
        
        results = await self.orchestrator.execute_plan(plan)
        
        # Process results by type
        for step_id, step_results in results.items():
            if "error" not in step_results:
                # Extract entities
                if "similar_entities" in step_results:
                    view_results["entities"].extend(step_results["similar_entities"][:10])
                elif "found_entities" in step_results:
                    view_results["entities"].extend(step_results["found_entities"][:10])
                
                # Extract relationships
                if "relationships" in step_results:
                    view_results["relationships"].extend(step_results["relationships"][:10])
                
                # Extract chunks
                if "chunks" in step_results:
                    view_results["chunks"].extend(step_results["chunks"][:5])
        
        return view_results
    
    async def _generate_discourse_insights(self, view, view_results: Dict) -> List[Dict]:
        """Generate insights specific to discourse analysis"""
        insights = []
        
        # Create interrogative-specific prompt
        prompt = generate_analysis_prompt(
            interrogative=view.interrogative,
            focus=view.focus,
            entities=view_results.get("entities", []),
            relationships=view_results.get("relationships", []),
            chunks=view_results.get("chunks", [])
        )
        
        # Use LLM to generate insights
        try:
            response = await self.llm.agenerate(prompt)
            
            insight = {
                "interrogative": view.interrogative,
                "focus": view.focus,
                "type": "discourse_insight",
                "generated_analysis": response,
                "supporting_entities": [e.get("entity_name", "") for e in view_results.get("entities", [])[:5]],
                "supporting_relationships": [r.get("relationship", "") for r in view_results.get("relationships", [])[:5]]
            }
            insights.append(insight)
            
        except Exception as e:
            logger.error(f"Failed to generate discourse insight: {e}")
        
        return insights
    
    async def _apply_discourse_transformations(self, analysis_results: Dict, transformation_chains: List[Dict]) -> List[Dict]:
        """Apply cross-interrogative transformations"""
        patterns = []
        
        for transform in transformation_chains:
            operator = transform.get("operator", "")
            
            if operator == "map_to_causal_model":
                # Analyze causal relationships across interrogatives
                pattern = {
                    "type": "causal_pattern",
                    "description": "Causal relationship mapping",
                    "who_to_what": self._find_causal_links(
                        analysis_results["entities_by_interrogative"].get("Who", []),
                        analysis_results["entities_by_interrogative"].get("What", [])
                    ),
                    "what_to_effect": self._find_causal_links(
                        analysis_results["entities_by_interrogative"].get("What", []),
                        analysis_results["entities_by_interrogative"].get("With What Effect", [])
                    )
                }
                patterns.append(pattern)
                
            elif operator == "to_network_graph":
                # Build network representation
                pattern = {
                    "type": "network_pattern",
                    "description": "Social network structure",
                    "central_actors": self._identify_central_actors(
                        analysis_results["entities_by_interrogative"].get("Who", [])
                    ),
                    "information_flow": self._trace_information_flow(
                        analysis_results["entities_by_interrogative"],
                        analysis_results["relationships_by_interrogative"]
                    )
                }
                patterns.append(pattern)
        
        return patterns
    
    def _find_causal_links(self, source_entities: List, target_entities: List) -> List[Dict]:
        """Find potential causal links between entity sets"""
        links = []
        for source in source_entities[:3]:
            for target in target_entities[:3]:
                links.append({
                    "source": source.get("entity_name", "Unknown"),
                    "target": target.get("entity_name", "Unknown"),
                    "strength": 0.5  # Would be calculated based on co-occurrence
                })
        return links
    
    def _identify_central_actors(self, who_entities: List) -> List[str]:
        """Identify central actors from Who analysis"""
        return [e.get("entity_name", "Unknown") for e in who_entities[:5]]
    
    def _trace_information_flow(self, entities_by_int: Dict, relationships_by_int: Dict) -> Dict:
        """Trace information flow patterns"""
        return {
            "actors": len(entities_by_int.get("Who", [])),
            "messages": len(entities_by_int.get("What", [])),
            "audiences": len(entities_by_int.get("To Whom", [])),
            "mechanisms": len(relationships_by_int.get("How", []))
        }
    
    async def _synthesize_discourse_findings(self, analysis_results: Dict) -> Dict:
        """Synthesize findings across all interrogatives"""
        synthesis_prompt = f"""
        Based on the discourse analysis using the five interrogatives framework, synthesize the key findings:
        
        Total insights generated: {len(analysis_results['insights'])}
        Cross-interrogative patterns found: {len(analysis_results['cross_interrogative_patterns'])}
        
        Key actors (Who): {len(analysis_results['entities_by_interrogative'].get('Who', []))} identified
        Key narratives (What): {len(analysis_results['entities_by_interrogative'].get('What', []))} identified
        
        Provide a concise synthesis that answers:
        1. What are the main discourse patterns?
        2. How do the interrogatives relate to each other?
        3. What are the key implications?
        """
        
        try:
            synthesis = await self.llm.agenerate(synthesis_prompt)
            return {
                "synthesis": synthesis,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "total_insights": len(analysis_results['insights']),
                    "cross_patterns": len(analysis_results['cross_interrogative_patterns']),
                    "interrogatives_analyzed": len(analysis_results['entities_by_interrogative'])
                }
            }
        except Exception as e:
            logger.error(f"Failed to synthesize: {e}")
            return {"error": str(e)}
    
    async def execute_all_scenarios(self, scenarios: List[Dict], dataset_info: Dict) -> Dict[str, Any]:
        """Execute all scenarios using discourse framework"""
        try:
            self._trace("execution_start", {
                "scenario_count": len(scenarios),
                "dataset": dataset_info,
                "mode": "discourse_enhanced"
            })
            
            # Convert dict scenarios to DiscourseAnalysisScenario objects if needed
            discourse_scenarios = []
            for scenario_dict in scenarios:
                if isinstance(scenario_dict, dict):
                    # Create DiscourseAnalysisScenario from dict
                    views = []
                    for view_dict in scenario_dict.get("interrogative_views", []):
                        view = DiscourseInterrogativeView(**view_dict)
                        views.append(view)
                    
                    scenario = DiscourseAnalysisScenario(
                        title=scenario_dict.get("title", "Analysis"),
                        research_question=scenario_dict.get("research_question", ""),
                        interrogative_views=views,
                        mini_ontologies=scenario_dict.get("mini_ontologies", {}),
                        unified_ontology=scenario_dict.get("unified_ontology", {}),
                        retrieval_chains=scenario_dict.get("retrieval_chains", []),
                        transformation_chains=scenario_dict.get("transformation_chains", []),
                        expected_insights=scenario_dict.get("expected_insights", []),
                        complexity_level=scenario_dict.get("complexity_level", "Medium"),
                        analysis_pipeline=scenario_dict.get("analysis_pipeline", [])
                    )
                    discourse_scenarios.append(scenario)
                else:
                    discourse_scenarios.append(scenario_dict)
            
            # Initialize if needed
            if not self.initialized:
                success = await self.initialize()
                if not success:
                    return {
                        "success": False,
                        "error": "Failed to initialize DIGIMON"
                    }
            
            # Prepare dataset
            dataset_path = dataset_info.get("path", "COVID-19-conspiracy-theories-tweets.csv")
            await self.prepare_dataset(dataset_path)
            
            # Execute each scenario
            results = {
                "scenario_results": [],
                "execution_summary": {
                    "total_scenarios": len(discourse_scenarios),
                    "successful": 0,
                    "failed": 0,
                    "total_insights_generated": 0,
                    "total_entities_found": 0,
                    "cross_interrogative_patterns": []
                }
            }
            
            for idx, scenario in enumerate(discourse_scenarios):
                self._trace("progress", {
                    "percent": int((idx / len(discourse_scenarios)) * 100),
                    "message": f"Analyzing scenario: {scenario.title}"
                })
                
                # Build graph
                graph_id = await self._build_graph(scenario)
                if not graph_id:
                    results["scenario_results"].append({
                        "scenario": scenario.title,
                        "error": "Failed to build graph"
                    })
                    results["execution_summary"]["failed"] += 1
                    continue
                
                # Execute discourse analysis
                scenario_result = await self.execute_discourse_analysis(scenario, graph_id)
                
                if scenario_result.get("success"):
                    results["execution_summary"]["successful"] += 1
                    results["execution_summary"]["total_insights_generated"] += len(
                        scenario_result.get("insights", [])
                    )
                    results["execution_summary"]["total_entities_found"] += len(
                        scenario_result.get("entities_found", [])
                    )
                    
                    # Extract cross-interrogative patterns
                    if "cross_interrogative_patterns" in scenario_result:
                        results["execution_summary"]["cross_interrogative_patterns"].extend(
                            scenario_result["cross_interrogative_patterns"]
                        )
                else:
                    results["execution_summary"]["failed"] += 1
                
                results["scenario_results"].append(scenario_result)
            
            self._trace("execution_complete", {
                "total_scenarios": len(discourse_scenarios),
                "successful": results["execution_summary"]["successful"],
                "failed": results["execution_summary"]["failed"]
            })
            
            return results
            
        except Exception as e:
            self._trace("execution_error", {"error": str(e)})
            logger.error(f"Execution error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute_full_discourse_analysis(self, research_focus: str, dataset_path: str) -> Dict[str, Any]:
        """Execute complete discourse analysis workflow"""
        try:
            self._trace("full_analysis_start", {
                "research_focus": research_focus,
                "dataset": dataset_path
            })
            
            # Initialize system
            if not await self.initialize():
                return {"error": "Failed to initialize"}
            
            # Generate discourse scenarios
            scenarios = await self.generate_discourse_scenarios(research_focus)
            
            # Prepare dataset
            dataset_name = "covid_discourse_analysis"
            success = await self._prepare_dataset(dataset_path, dataset_name)
            if not success:
                return {"error": "Failed to prepare dataset"}
            
            # Execute each scenario
            all_results = {
                "execution_id": datetime.now().isoformat(),
                "research_focus": research_focus,
                "discourse_framework": "Five Interrogatives",
                "total_scenarios": len(scenarios),
                "scenario_results": []
            }
            
            for scenario in scenarios:
                # Build appropriate graph
                graph_id = await self._build_discourse_graph(scenario, dataset_name)
                if not graph_id:
                    continue
                
                # Execute discourse analysis
                results = await self.execute_discourse_analysis(scenario, graph_id)
                all_results["scenario_results"].append(results)
            
            # Generate executive summary
            all_results["executive_summary"] = await self._generate_executive_summary(all_results)
            
            self._trace("full_analysis_complete", {
                "scenarios_analyzed": len(all_results["scenario_results"]),
                "total_insights": sum(len(r.get("insights", [])) for r in all_results["scenario_results"])
            })
            
            return all_results
            
        except Exception as e:
            self._trace("full_analysis_error", {"error": str(e)})
            return {"error": str(e)}
    
    async def _build_graph(self, scenario: DiscourseAnalysisScenario) -> Optional[str]:
        """Build graph for discourse analysis"""
        try:
            # Select graph type based on complexity
            complexity = scenario.complexity_level
            graph_types = {
                "Simple": "er_graph",
                "Medium": "rk_graph",
                "Complex": "tree_graph_balanced"
            }
            graph_type = graph_types.get(complexity, "er_graph")
            graph_id = f"discourse_{graph_type}_{int(time.time())}"
            
            self._trace("graph_build_start", {
                "graph_type": graph_type,
                "graph_id": graph_id,
                "scenario": scenario.title
            })
            
            # For now, simulate graph building
            # In real implementation, would use graph construction tools
            self._trace("graph_build_complete", {
                "graph_id": graph_id,
                "status": "success"
            })
            
            return graph_id
            
        except Exception as e:
            self._trace("graph_build_error", {"error": str(e)})
            return None
    
    async def prepare_dataset(self, dataset_path: str) -> bool:
        """Prepare dataset for analysis"""
        try:
            self._trace("dataset_prep_start", {"path": dataset_path})
            # For now, return success
            # In real implementation, would prepare corpus
            self._trace("dataset_prep_complete", {"status": "success"})
            return True
        except Exception as e:
            self._trace("dataset_prep_error", {"error": str(e)})
            return False
    
    async def _prepare_dataset(self, dataset_path: str, dataset_name: str) -> bool:
        """Prepare dataset for discourse analysis"""
        # Similar to traced executor but with discourse focus
        # Implementation would be similar to TracedSocialMediaAnalysisExecutor
        return True
    
    async def _build_discourse_graph(self, scenario: DiscourseAnalysisScenario, dataset_name: str) -> Optional[str]:
        """Build graph optimized for discourse analysis"""
        # Select graph type based on complexity
        # Implementation would build appropriate graph
        return f"{dataset_name}_discourse_graph"
    
    async def _generate_executive_summary(self, all_results: Dict) -> Dict:
        """Generate executive summary of discourse analysis"""
        total_insights = sum(len(r.get("insights", [])) for r in all_results["scenario_results"])
        total_patterns = sum(len(r.get("cross_interrogative_patterns", [])) for r in all_results["scenario_results"])
        
        summary_prompt = f"""
        Generate an executive summary of the discourse analysis:
        
        Research Focus: {all_results['research_focus']}
        Scenarios Analyzed: {all_results['total_scenarios']}
        Total Insights: {total_insights}
        Cross-Interrogative Patterns: {total_patterns}
        
        Provide a high-level summary suitable for decision makers that includes:
        1. Key findings across all scenarios
        2. Major discourse patterns identified
        3. Strategic implications
        4. Recommended actions or areas for further investigation
        """
        
        try:
            summary = await self.llm.agenerate(summary_prompt)
            return {
                "executive_summary": summary,
                "key_metrics": {
                    "scenarios_analyzed": all_results['total_scenarios'],
                    "total_insights": total_insights,
                    "cross_patterns": total_patterns
                },
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return {"error": str(e)}


# Integration function for use with UI
async def run_discourse_analysis_with_ui(research_focus: str, dataset_path: str, trace_callback: Callable) -> Dict:
    """Run discourse analysis with UI integration"""
    executor = DiscourseEnhancedSocialMediaExecutor(trace_callback=trace_callback)
    return await executor.execute_full_discourse_analysis(research_focus, dataset_path)


# Test function
async def test_discourse_execution():
    """Test the discourse-enhanced execution"""
    def print_trace(event_type, data):
        print(f"[DISCOURSE] {event_type}: {json.dumps(data, indent=2)}")
    
    executor = DiscourseEnhancedSocialMediaExecutor(trace_callback=print_trace)
    
    results = await executor.execute_full_discourse_analysis(
        research_focus="How do COVID-19 conspiracy theories spread on social media and what effects do they have?",
        dataset_path="COVID-19-conspiracy-theories-tweets.csv"
    )
    
    print("\n=== DISCOURSE ANALYSIS RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(test_discourse_execution())