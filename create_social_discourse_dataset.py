#!/usr/bin/env python3
"""
Create Synthetic Social Discourse Dataset
Optimized for testing DIGIMON's social network analysis, content analysis, 
and discourse analysis capabilities at small scale.

This dataset includes:
1. Dense social network with clear communities and influencers
2. Rich discourse patterns with multiple viewpoints
3. Temporal dynamics showing opinion evolution
4. Clear narrative arcs and themes
5. Explicit relationships between actors
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import random

def create_social_discourse_dataset():
    """Create a rich synthetic dataset for social/discourse analysis"""
    
    # Create dataset directory
    dataset_dir = Path("Data/Social_Discourse_Test")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the social network actors with clear roles
    actors = {
        # Tech influencers (pro-AI community)
        "@TechVisionary": {
            "name": "Alex Chen", 
            "role": "Tech CEO",
            "community": "tech_optimists",
            "influence": "high",
            "followers": 50000
        },
        "@AIResearcher": {
            "name": "Dr. Sarah Johnson",
            "role": "AI Researcher", 
            "community": "tech_optimists",
            "influence": "high",
            "followers": 30000
        },
        "@StartupFounder": {
            "name": "Mike Williams",
            "role": "Startup Founder",
            "community": "tech_optimists", 
            "influence": "medium",
            "followers": 15000
        },
        
        # AI Ethics advocates (cautious community)
        "@EthicsProf": {
            "name": "Prof. Emily Brown",
            "role": "Ethics Professor",
            "community": "ai_ethics",
            "influence": "high",
            "followers": 25000
        },
        "@PolicyExpert": {
            "name": "James Taylor",
            "role": "Policy Analyst",
            "community": "ai_ethics",
            "influence": "medium", 
            "followers": 12000
        },
        "@SafetyAdvocate": {
            "name": "Lisa Martinez",
            "role": "AI Safety Advocate",
            "community": "ai_ethics",
            "influence": "medium",
            "followers": 8000
        },
        
        # Labor activists (concerned community)
        "@UnionLeader": {
            "name": "Robert Davis",
            "role": "Union Representative", 
            "community": "labor_rights",
            "influence": "high",
            "followers": 35000
        },
        "@WorkerRights": {
            "name": "Maria Garcia",
            "role": "Labor Activist",
            "community": "labor_rights",
            "influence": "medium",
            "followers": 10000
        },
        
        # Journalists (bridge community)
        "@TechJournalist": {
            "name": "Chris Anderson",
            "role": "Tech Reporter",
            "community": "media",
            "influence": "high",
            "followers": 40000
        },
        "@InvestigativeRep": {
            "name": "Amanda White",
            "role": "Investigative Journalist",
            "community": "media",
            "influence": "medium",
            "followers": 20000
        }
    }
    
    # Create temporal discourse around a major AI announcement
    base_date = datetime(2024, 3, 15, 9, 0)
    
    discourse_timeline = []
    
    # Phase 1: Initial announcement (Hour 0-2)
    discourse_timeline.extend([
        {
            "author": "@TechVisionary",
            "timestamp": base_date,
            "content": "🚀 Excited to announce our new AI assistant that can automate 70% of office tasks! This will free humans to focus on creative work. The future is here! #AIRevolution #FutureOfWork",
            "retweets": 5000,
            "likes": 15000,
            "replies": 500,
            "mentions": [],
            "hashtags": ["#AIRevolution", "#FutureOfWork"]
        },
        {
            "author": "@AIResearcher", 
            "timestamp": base_date + timedelta(minutes=15),
            "content": "Congratulations @TechVisionary! The benchmark results are impressive. Our research shows 3x productivity gains when humans collaborate with AI. #AIResearch",
            "retweets": 1000,
            "likes": 3000,
            "replies": 50,
            "mentions": ["@TechVisionary"],
            "hashtags": ["#AIResearch"]
        },
        {
            "author": "@StartupFounder",
            "timestamp": base_date + timedelta(minutes=30),
            "content": "This is game-changing! @TechVisionary's announcement validates what we've been building. Small businesses can now compete with enterprises. #StartupLife #AITools",
            "retweets": 500,
            "likes": 2000,
            "replies": 30,
            "mentions": ["@TechVisionary"],
            "hashtags": ["#StartupLife", "#AITools"]
        }
    ])
    
    # Phase 2: Critical responses emerge (Hour 2-4)
    discourse_timeline.extend([
        {
            "author": "@UnionLeader",
            "timestamp": base_date + timedelta(hours=2),
            "content": "Hold on @TechVisionary. What about the millions of office workers whose jobs will disappear? We need regulations and retraining programs BEFORE deployment. #WorkersRights #AIRegulation", 
            "retweets": 3000,
            "likes": 8000,
            "replies": 200,
            "mentions": ["@TechVisionary"],
            "hashtags": ["#WorkersRights", "#AIRegulation"]
        },
        {
            "author": "@EthicsProf",
            "timestamp": base_date + timedelta(hours=2, minutes=30),
            "content": "Important questions from @UnionLeader. Also concerned about bias in these systems. Who audited the AI? What safeguards exist? We need transparency. @TechVisionary #AIEthics #ResponsibleAI",
            "retweets": 2000,
            "likes": 5000,
            "replies": 100,
            "mentions": ["@UnionLeader", "@TechVisionary"],
            "hashtags": ["#AIEthics", "#ResponsibleAI"]
        },
        {
            "author": "@TechJournalist",
            "timestamp": base_date + timedelta(hours=3),
            "content": "BREAKING: Sources tell me @TechVisionary's AI has not undergone independent safety testing. Company refused to comment on job displacement numbers. Full investigation coming. #TechNews",
            "retweets": 4000,
            "likes": 10000,
            "replies": 300,
            "mentions": ["@TechVisionary"],
            "hashtags": ["#TechNews"]
        }
    ])
    
    # Phase 3: Community polarization (Hour 4-8)
    discourse_timeline.extend([
        {
            "author": "@TechVisionary",
            "timestamp": base_date + timedelta(hours=4),
            "content": "To address concerns: 1) We're creating MORE jobs in AI supervision 2) Offering free retraining 3) Gradual rollout with worker input. Innovation and responsibility go hand-in-hand. @UnionLeader @EthicsProf",
            "retweets": 2000,
            "likes": 6000,
            "replies": 150,
            "mentions": ["@UnionLeader", "@EthicsProf"],
            "hashtags": []
        },
        {
            "author": "@PolicyExpert",
            "timestamp": base_date + timedelta(hours=4, minutes=30),
            "content": "Not enough @TechVisionary. We need: mandatory impact assessments, worker compensation funds, algorithmic audits. The EU AI Act provides a good framework. #AIPolicy #Regulation",
            "retweets": 1500,
            "likes": 4000,
            "replies": 80,
            "mentions": ["@TechVisionary"],
            "hashtags": ["#AIPolicy", "#Regulation"]
        },
        {
            "author": "@WorkerRights",
            "timestamp": base_date + timedelta(hours=5),
            "content": "My sister just learned her data entry job will be eliminated. @TechVisionary's 'retraining' is 2-week online course. This is not enough! Real people are suffering. #JobLoss #AIImpact",
            "retweets": 5000,
            "likes": 12000,
            "replies": 400,
            "mentions": ["@TechVisionary"],
            "hashtags": ["#JobLoss", "#AIImpact"]
        },
        {
            "author": "@SafetyAdvocate",
            "timestamp": base_date + timedelta(hours=5, minutes=30),
            "content": "Thread: Why @TechVisionary's AI could lead to catastrophic failures 🧵 1/ No external audits 2/ Rushing to market 3/ Ignoring long-term risks 4/ Prioritizing profits over safety [1/8]",
            "retweets": 3000,
            "likes": 7000,
            "replies": 200,
            "mentions": ["@TechVisionary"],
            "hashtags": []
        }
    ])
    
    # Phase 4: Media amplification and bridge-building attempts (Hour 8-12)
    discourse_timeline.extend([
        {
            "author": "@InvestigativeRep",
            "timestamp": base_date + timedelta(hours=8),
            "content": "EXCLUSIVE: Internal emails show @TechVisionary knew about job losses but buried report. 'Focus on positive messaging' exec wrote. Documents: [link] #AIScandal",
            "retweets": 8000,
            "likes": 20000,
            "replies": 1000,
            "mentions": ["@TechVisionary"],
            "hashtags": ["#AIScandal"]
        },
        {
            "author": "@AIResearcher",
            "timestamp": base_date + timedelta(hours=9),
            "content": "We need nuanced discussion. Yes, AI displaces some jobs BUT creates others. History shows tech transitions work IF we support workers. Let's find common ground @UnionLeader @EthicsProf",
            "retweets": 2500,
            "likes": 6000,
            "replies": 200,
            "mentions": ["@UnionLeader", "@EthicsProf"],
            "hashtags": []
        },
        {
            "author": "@EthicsProf",
            "timestamp": base_date + timedelta(hours=10),
            "content": "Agree with @AIResearcher. Proposing roundtable with all stakeholders: tech leaders, workers, ethicists, policymakers. @TechVisionary will you participate? Time for dialogue not debate.",
            "retweets": 3500,
            "likes": 9000,
            "replies": 300,
            "mentions": ["@AIResearcher", "@TechVisionary"],
            "hashtags": []
        },
        {
            "author": "@TechVisionary",
            "timestamp": base_date + timedelta(hours=11),
            "content": "I hear you all. @InvestigativeRep's report missing context but concerns are valid. Accepting @EthicsProf's invitation. Also establishing $50M worker transition fund. We must do better.",
            "retweets": 4000,
            "likes": 10000,
            "replies": 500,
            "mentions": ["@InvestigativeRep", "@EthicsProf"],
            "hashtags": []
        }
    ])
    
    # Phase 5: Resolution and ongoing dialogue (Hour 12-24)
    discourse_timeline.extend([
        {
            "author": "@UnionLeader",
            "timestamp": base_date + timedelta(hours=12),
            "content": "Progress. @TechVisionary's fund is a start. But we need: 1) Worker representation on AI deployment decisions 2) Profit-sharing from automation 3) Right to human review. The fight continues.",
            "retweets": 4000,
            "likes": 11000,
            "replies": 400,
            "mentions": ["@TechVisionary"],
            "hashtags": []
        },
        {
            "author": "@TechJournalist",
            "timestamp": base_date + timedelta(hours=16),
            "content": "UPDATE: Industry-wide movement forming. 5 major tech companies joining @TechVisionary's worker fund. @UnionLeader calling it 'small victory'. The AI debate is reshaping Silicon Valley. #TechNews",
            "retweets": 5000,
            "likes": 15000,
            "replies": 600,
            "mentions": ["@TechVisionary", "@UnionLeader"],
            "hashtags": ["#TechNews"]
        },
        {
            "author": "@StartupFounder",
            "timestamp": base_date + timedelta(hours=20),
            "content": "Today showed why diverse voices matter in tech. Thanks @EthicsProf @UnionLeader for pushing us to think bigger. Innovation must include everyone. #InclusiveTech #AIForAll",
            "retweets": 2000,
            "likes": 7000,
            "replies": 150,
            "mentions": ["@EthicsProf", "@UnionLeader"],
            "hashtags": ["#InclusiveTech", "#AIForAll"]
        }
    ])
    
    # Create additional background conversations showing network effects
    background_conversations = [
        {
            "author": "@PolicyExpert",
            "timestamp": base_date + timedelta(hours=6),
            "content": "Working with @EthicsProf on policy framework. Key insight from @UnionLeader: automation impact varies by region. One-size-fits-all won't work. #PolicyDevelopment",
            "retweets": 800,
            "likes": 2500,
            "replies": 40,
            "mentions": ["@EthicsProf", "@UnionLeader"],
            "hashtags": ["#PolicyDevelopment"]
        },
        {
            "author": "@SafetyAdvocate", 
            "timestamp": base_date + timedelta(hours=7),
            "content": "@AIResearcher makes good points but we can't ignore existential risks. @EthicsProf's framework addresses near-term issues. We need both perspectives. #AISafety #LongTermism",
            "retweets": 1200,
            "likes": 3500,
            "replies": 90,
            "mentions": ["@AIResearcher", "@EthicsProf"],
            "hashtags": ["#AISafety", "#LongTermism"]
        },
        {
            "author": "@WorkerRights",
            "timestamp": base_date + timedelta(hours=13),
            "content": "Powerful testimony from displaced workers at @UnionLeader's rally. Their stories matter more than any tech announcement. Real impact of @TechVisionary's AI. #WorkerStories",
            "retweets": 3500,
            "likes": 9000,
            "replies": 350,
            "mentions": ["@UnionLeader", "@TechVisionary"],
            "hashtags": ["#WorkerStories"]
        }
    ]
    
    # Combine all discourse
    all_discourse = discourse_timeline + background_conversations
    
    # Sort by timestamp
    all_discourse.sort(key=lambda x: x['timestamp'])
    
    # Format as social media data
    formatted_posts = []
    post_id = 1000
    
    for post in all_discourse:
        formatted_posts.append({
            "post_id": str(post_id),
            "author": post["author"],
            "author_info": actors.get(post["author"], {"role": "Unknown", "community": "unknown"}),
            "timestamp": post["timestamp"].isoformat(),
            "content": post["content"],
            "metrics": {
                "retweets": post["retweets"],
                "likes": post["likes"], 
                "replies": post["replies"]
            },
            "mentions": post["mentions"],
            "hashtags": post["hashtags"],
            "discourse_phase": get_discourse_phase(post["timestamp"], base_date)
        })
        post_id += 1
    
    # Create text files for each discourse phase
    phases = {
        "announcement": "Tech leader announces AI automation tool",
        "criticism": "Labor and ethics concerns emerge", 
        "polarization": "Communities take sides and debate intensifies",
        "investigation": "Media reveals hidden information",
        "resolution": "Stakeholders work toward compromise"
    }
    
    for phase_name, description in phases.items():
        phase_posts = [p for p in formatted_posts if p["discourse_phase"] == phase_name]
        
        with open(dataset_dir / f"{phase_name}_phase.txt", "w") as f:
            f.write(f"DISCOURSE PHASE: {description.upper()}\n")
            f.write(f"Number of posts: {len(phase_posts)}\n")
            f.write("=" * 80 + "\n\n")
            
            for post in phase_posts:
                f.write(f"Author: {post['author']} ({post['author_info']['role']})\n")
                f.write(f"Community: {post['author_info']['community']}\n")
                f.write(f"Timestamp: {post['timestamp']}\n")
                f.write(f"Content: {post['content']}\n")
                f.write(f"Engagement: {post['metrics']['retweets']} retweets, {post['metrics']['likes']} likes, {post['metrics']['replies']} replies\n")
                if post['mentions']:
                    f.write(f"Mentions: {', '.join(post['mentions'])}\n")
                if post['hashtags']:
                    f.write(f"Hashtags: {', '.join(post['hashtags'])}\n")
                f.write("-" * 80 + "\n\n")
    
    # Create a comprehensive summary file
    with open(dataset_dir / "dataset_overview.txt", "w") as f:
        f.write("SOCIAL DISCOURSE TEST DATASET\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DATASET CHARACTERISTICS:\n")
        f.write("- Dense social network with clear communities\n")
        f.write("- Rich discourse patterns showing opinion evolution\n") 
        f.write("- Multiple stakeholder perspectives\n")
        f.write("- Clear narrative arc with 5 phases\n")
        f.write("- Explicit mentions creating network connections\n\n")
        
        f.write("ACTORS AND COMMUNITIES:\n")
        for actor, info in actors.items():
            f.write(f"- {actor}: {info['name']} ({info['role']}) - {info['community']} community, {info['influence']} influence\n")
        
        f.write("\n\nDISCOURSE PHASES:\n")
        for phase, desc in phases.items():
            f.write(f"- {phase}: {desc}\n")
            
        f.write("\n\nKEY THEMES:\n")
        f.write("- AI automation and job displacement\n")
        f.write("- Tech innovation vs worker protection\n")
        f.write("- Ethics and responsible AI development\n")
        f.write("- Multi-stakeholder dialogue and compromise\n")
        f.write("- Media's role in tech accountability\n")
        
        f.write("\n\nNETWORK ANALYSIS OPPORTUNITIES:\n")
        f.write("- Community detection (4 clear communities)\n")
        f.write("- Influencer identification (high-follower accounts)\n")
        f.write("- Information flow tracing (retweets and mentions)\n")
        f.write("- Sentiment evolution over time\n")
        f.write("- Bridge actors connecting communities\n")
    
    # Create actor profiles file
    with open(dataset_dir / "actor_profiles.txt", "w") as f:
        f.write("SOCIAL NETWORK ACTOR PROFILES\n")
        f.write("=" * 80 + "\n\n")
        
        for community in ["tech_optimists", "ai_ethics", "labor_rights", "media"]:
            f.write(f"\n{community.upper().replace('_', ' ')} COMMUNITY:\n")
            f.write("-" * 40 + "\n")
            
            community_actors = {k: v for k, v in actors.items() if v["community"] == community}
            for actor, info in community_actors.items():
                f.write(f"\n{actor} - {info['name']}\n")
                f.write(f"Role: {info['role']}\n")
                f.write(f"Influence: {info['influence']}\n")
                f.write(f"Followers: {info['followers']:,}\n")
    
    # Create JSON data for easier parsing
    dataset_json = {
        "dataset_name": "Social_Discourse_Test",
        "description": "Synthetic dataset for testing social network and discourse analysis",
        "actors": actors,
        "posts": formatted_posts,
        "discourse_phases": phases,
        "network_metrics": {
            "total_actors": len(actors),
            "total_posts": len(formatted_posts),
            "communities": 4,
            "time_span_hours": 24,
            "total_mentions": sum(len(p["mentions"]) for p in formatted_posts),
            "unique_hashtags": len(set(tag for p in formatted_posts for tag in p["hashtags"]))
        }
    }
    
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"Created Social Discourse Test dataset at: {dataset_dir}")
    print(f"- {len(actors)} actors across 4 communities")
    print(f"- {len(formatted_posts)} posts over 24-hour period")
    print(f"- 5 discourse phases showing opinion evolution")
    print(f"- Rich mention network for relationship analysis")

def get_discourse_phase(timestamp, base_date):
    """Determine which discourse phase a post belongs to"""
    hours_elapsed = (timestamp - base_date).total_seconds() / 3600
    
    if hours_elapsed < 2:
        return "announcement"
    elif hours_elapsed < 4:
        return "criticism"
    elif hours_elapsed < 8:
        return "polarization"
    elif hours_elapsed < 12:
        return "investigation"
    else:
        return "resolution"

if __name__ == "__main__":
    create_social_discourse_dataset()