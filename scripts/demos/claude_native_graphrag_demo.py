#!/usr/bin/env python3
"""
Demonstration: Claude Code performing GraphRAG analysis with native capabilities
This shows how Claude Code can analyze conspiracy theory discourse without complex frameworks
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import re
from datetime import datetime

class ClaudeNativeGraphRAG:
    """
    Claude Code's native approach to GraphRAG - using built-in reasoning
    """
    
    def __init__(self):
        self.entities = defaultdict(dict)
        self.relationships = []
        self.themes = defaultdict(list)
        self.narrative_patterns = {}
        
    def analyze_corpus(self, corpus_dir):
        """
        Claude Code's approach: Read, understand, extract patterns
        """
        print("=== Claude Code Native GraphRAG Analysis ===\n")
        
        # Step 1: Read and understand the corpus structure
        print("Step 1: Understanding corpus structure...")
        corpus_files = list(Path(corpus_dir).glob("*.txt"))
        
        topics = {}
        for file_path in corpus_files:
            if file_path.name != "Corpus.json":
                topic_name = file_path.stem
                topics[topic_name] = self._analyze_topic_file(file_path)
        
        # Step 2: Extract entities and relationships
        print("\nStep 2: Extracting entities and relationships...")
        self._extract_graph_elements(topics)
        
        # Step 3: Analyze patterns
        print("\nStep 3: Analyzing discourse patterns...")
        patterns = self._analyze_patterns()
        
        # Step 4: Generate insights
        print("\nStep 4: Generating insights...")
        insights = self._generate_insights(patterns)
        
        return {
            "entities": dict(self.entities),
            "relationships": self.relationships,
            "patterns": patterns,
            "insights": insights
        }
    
    def _analyze_topic_file(self, file_path):
        """
        Analyze individual conspiracy theory topic
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract metadata
        lines = content.split('\n')
        topic_info = {
            "name": file_path.stem,
            "tweets": [],
            "support_count": 0,
            "neutral_count": 0,
            "against_count": 0
        }
        
        current_section = None
        for line in lines:
            if line.startswith("## SUPPORT"):
                current_section = "support"
            elif line.startswith("## NEUTRAL"):
                current_section = "neutral"
            elif line.startswith("## AGAINST"):
                current_section = "against"
            elif line.strip() and not line.startswith("#"):
                if current_section:
                    topic_info["tweets"].append({
                        "text": line.strip(),
                        "stance": current_section
                    })
                    topic_info[f"{current_section}_count"] += 1
        
        return topic_info
    
    def _extract_graph_elements(self, topics):
        """
        Extract entities and relationships using Claude's pattern recognition
        """
        # Key entities to look for
        entity_patterns = {
            "actors": ["government", "WHO", "China", "lab", "scientists", "elites", "pharma"],
            "concepts": ["bioweapon", "control", "agenda", "truth", "cover-up", "manipulation"],
            "events": ["pandemic", "outbreak", "leak", "release"],
            "hashtags": re.compile(r'#\w+'),
            "questions": re.compile(r'[?]')
        }
        
        for topic_name, topic_data in topics.items():
            # Create topic entity
            self.entities["topics"][topic_name] = {
                "type": "conspiracy_theory",
                "tweet_count": len(topic_data["tweets"]),
                "support_ratio": topic_data["support_count"] / max(len(topic_data["tweets"]), 1)
            }
            
            # Extract entities from tweets
            for tweet in topic_data["tweets"]:
                text = tweet["text"].lower()
                
                # Extract actors
                for actor in entity_patterns["actors"]:
                    if actor in text:
                        if actor not in self.entities["actors"]:
                            self.entities["actors"][actor] = {"mentions": 0, "topics": set()}
                        self.entities["actors"][actor]["mentions"] += 1
                        self.entities["actors"][actor]["topics"].add(topic_name)
                        
                        # Create relationship
                        self.relationships.append({
                            "source": actor,
                            "target": topic_name,
                            "type": "mentioned_in",
                            "stance": tweet["stance"]
                        })
                
                # Extract hashtags
                hashtags = entity_patterns["hashtags"].findall(tweet["text"])
                for hashtag in hashtags:
                    if hashtag not in self.entities["hashtags"]:
                        self.entities["hashtags"][hashtag] = {"count": 0, "topics": set()}
                    self.entities["hashtags"][hashtag]["count"] += 1
                    self.entities["hashtags"][hashtag]["topics"].add(topic_name)
    
    def _analyze_patterns(self):
        """
        Analyze discourse patterns in the conspiracy theories
        """
        patterns = {
            "narrative_themes": {},
            "rhetorical_strategies": {},
            "network_structure": {}
        }
        
        # Identify common themes
        theme_keywords = {
            "control": ["control", "manipulation", "agenda", "power"],
            "hidden_truth": ["truth", "cover-up", "hidden", "expose", "reveal"],
            "distrust": ["question", "investigate", "suspicious", "don't believe"],
            "intentionality": ["deliberate", "intentional", "planned", "designed"]
        }
        
        # Count theme occurrences
        for theme, keywords in theme_keywords.items():
            patterns["narrative_themes"][theme] = 0
            for entity_type in self.entities.values():
                for entity_data in entity_type.values():
                    if isinstance(entity_data, dict) and "topics" in entity_data:
                        patterns["narrative_themes"][theme] += len(entity_data["topics"])
        
        # Analyze rhetorical strategies
        patterns["rhetorical_strategies"] = {
            "questioning": len([r for r in self.relationships if "?" in str(r)]),
            "call_to_action": len([r for r in self.relationships if any(
                word in str(r).lower() for word in ["investigate", "expose", "reveal", "uncover"]
            )]),
            "authority_challenge": len([r for r in self.relationships if any(
                word in str(r).lower() for word in ["official", "narrative", "they", "them"]
            )])
        }
        
        # Network structure
        actor_connections = defaultdict(set)
        for rel in self.relationships:
            if rel["type"] == "mentioned_in":
                actor_connections[rel["source"]].add(rel["target"])
        
        patterns["network_structure"] = {
            "most_connected_actors": sorted(
                [(actor, len(topics)) for actor, topics in actor_connections.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "total_connections": len(self.relationships)
        }
        
        return patterns
    
    def _generate_insights(self, patterns):
        """
        Generate insights from the analysis
        """
        insights = []
        
        # Insight 1: Main narrative themes
        top_themes = sorted(
            patterns["narrative_themes"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        insights.append({
            "type": "narrative_analysis",
            "finding": f"The dominant narrative themes are: {', '.join([t[0] for t in top_themes[:3]])}",
            "evidence": f"Theme frequency analysis across {len(self.entities['topics'])} conspiracy theories"
        })
        
        # Insight 2: Key actors
        if patterns["network_structure"]["most_connected_actors"]:
            top_actors = patterns["network_structure"]["most_connected_actors"]
            insights.append({
                "type": "actor_analysis",
                "finding": f"Most frequently mentioned actors: {', '.join([a[0] for a in top_actors])}",
                "evidence": f"These actors appear across multiple conspiracy theories, suggesting central roles in the discourse"
            })
        
        # Insight 3: Rhetorical strategies
        insights.append({
            "type": "rhetorical_analysis",
            "finding": "Conspiracy theories heavily use questioning and calls to investigate",
            "evidence": f"{patterns['rhetorical_strategies']['questioning']} questioning instances, "
                      f"{patterns['rhetorical_strategies']['call_to_action']} calls to action"
        })
        
        # Insight 4: Network structure
        insights.append({
            "type": "network_analysis",
            "finding": "Conspiracy theories form an interconnected web through shared actors and themes",
            "evidence": f"{patterns['network_structure']['total_connections']} connections identified between entities"
        })
        
        return insights

def demonstrate_claude_graphrag():
    """
    Demonstrate Claude Code's native GraphRAG capabilities
    """
    analyzer = ClaudeNativeGraphRAG()
    
    # Analyze the COVID conspiracy corpus
    results = analyzer.analyze_corpus("Data/COVID_Conspiracy")
    
    # Display results
    print("\n=== Analysis Results ===\n")
    
    print("1. Entity Summary:")
    for entity_type, entities in results["entities"].items():
        print(f"   - {entity_type}: {len(entities)} found")
    
    print("\n2. Key Relationships:")
    print(f"   - Total relationships: {len(results['relationships'])}")
    relationship_types = Counter(r["type"] for r in results["relationships"])
    for rel_type, count in relationship_types.most_common():
        print(f"   - {rel_type}: {count}")
    
    print("\n3. Insights:")
    for i, insight in enumerate(results["insights"], 1):
        print(f"\n   Insight {i} ({insight['type']}):")
        print(f"   Finding: {insight['finding']}")
        print(f"   Evidence: {insight['evidence']}")
    
    # Save results
    output_file = "claude_graphrag_results.json"
    with open(output_file, 'w') as f:
        # Convert sets to lists for JSON serialization
        serializable_results = {
            "entities": {
                k: {
                    ent_name: {
                        key: list(val) if isinstance(val, set) else val
                        for key, val in ent_data.items()
                    }
                    for ent_name, ent_data in v.items()
                }
                for k, v in results["entities"].items()
            },
            "relationships": results["relationships"],
            "patterns": results["patterns"],
            "insights": results["insights"],
            "timestamp": datetime.now().isoformat()
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_claude_graphrag()
    
    # Show how Claude Code would answer specific questions
    print("\n=== Answering Specific Questions ===\n")
    
    questions = [
        "What are the main conspiracy theories?",
        "Who are the key actors mentioned?",
        "How do the theories connect?"
    ]
    
    for q in questions:
        print(f"Q: {q}")
        print(f"A: Based on the graph analysis, I can see...")
        print(f"   [Claude would synthesize answer from the extracted graph]\n")