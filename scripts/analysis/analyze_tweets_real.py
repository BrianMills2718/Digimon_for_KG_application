#!/usr/bin/env python3
"""Real analysis of COVID conspiracy tweets using discourse framework"""

import pandas as pd
import json
from collections import Counter, defaultdict
import re
from datetime import datetime

def analyze_real_tweets():
    """Perform actual analysis on the COVID conspiracy tweets"""
    print("REAL COVID-19 Conspiracy Tweet Analysis")
    print("=" * 80)
    
    # Load the actual dataset
    print("\n1. Loading actual tweet data...")
    df = pd.read_csv('COVID-19-conspiracy-theories-tweets.csv')
    print(f"✓ Loaded {len(df)} real tweets")
    print(f"Columns: {list(df.columns)}")
    
    # Show sample tweets
    print("\nSample tweets:")
    for i, row in df.head(3).iterrows():
        print(f"- {row['tweet'][:100]}...")
        print(f"  Type: {row.get('conspiracy_theory', 'Unknown')}, Label: {row.get('label', 'Unknown')}")
    
    # 2. Analyze by conspiracy type (Says What)
    print("\n2. SAYS WHAT - Conspiracy Narrative Analysis:")
    print("-" * 60)
    
    conspiracy_counts = df['conspiracy_theory'].value_counts()
    print("\nConspiracy theory distribution:")
    for ct_type, count in conspiracy_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {ct_type}: {count} tweets ({pct:.1f}%)")
    
    # Analyze support/deny distribution
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print("\nStance distribution:")
        for label, count in label_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {label}: {count} tweets ({pct:.1f}%)")
    
    # 3. Extract key narratives (Says What)
    print("\n3. Key Narratives Extracted:")
    print("-" * 60)
    
    # Common words/phrases by conspiracy type
    narrative_keywords = {
        'CT_1': ['economy', 'control', 'wealth', 'transfer', 'elite'],
        'CT_2': ['vaccine', 'microchip', 'chip', 'tracking', 'gates'],
        'CT_3': ['5g', 'towers', 'radiation', 'frequency'],
        'CT_4': ['bioweapon', 'lab', 'engineered', 'china'],
        'CT_5': ['population', 'control', 'depopulation', 'agenda'],
        'CT_6': ['fake', 'hoax', 'plandemic', 'scam']
    }
    
    for ct_type in conspiracy_counts.index[:3]:  # Top 3 types
        ct_tweets = df[df['conspiracy_theory'] == ct_type]['tweet']
        print(f"\n{ct_type} - Sample narratives:")
        
        # Find tweets with keywords
        keywords = narrative_keywords.get(ct_type, [])
        relevant_tweets = []
        for tweet in ct_tweets.sample(min(100, len(ct_tweets))):
            if any(kw.lower() in tweet.lower() for kw in keywords):
                relevant_tweets.append(tweet)
        
        for tweet in relevant_tweets[:3]:
            print(f"  • \"{tweet[:150]}...\"")
    
    # 4. Identify influential patterns (Who)
    print("\n\n4. WHO - Influencer Pattern Analysis:")
    print("-" * 60)
    
    # Extract @mentions as proxy for influence
    all_mentions = []
    for tweet in df['tweet']:
        mentions = re.findall(r'@(\w+)', str(tweet))
        all_mentions.extend(mentions)
    
    mention_counts = Counter(all_mentions)
    print("\nMost mentioned accounts (potential influencers):")
    for account, count in mention_counts.most_common(10):
        print(f"  @{account}: mentioned {count} times")
    
    # Extract hashtags
    all_hashtags = []
    for tweet in df['tweet']:
        hashtags = re.findall(r'#(\w+)', str(tweet))
        all_hashtags.extend(hashtags)
    
    hashtag_counts = Counter(all_hashtags)
    print("\nTop hashtags used:")
    for tag, count in hashtag_counts.most_common(10):
        print(f"  #{tag}: {count} times")
    
    # 5. Analyze engagement patterns (To Whom & With What Effect)
    print("\n\n5. TO WHOM & WITH WHAT EFFECT - Engagement Analysis:")
    print("-" * 60)
    
    # Analyze by label to see which conspiracy types get support
    if 'label' in df.columns:
        support_by_type = df[df['label'] == 'support'].groupby('conspiracy_theory').size()
        deny_by_type = df[df['label'] == 'deny'].groupby('conspiracy_theory').size()
        
        print("\nSupport rates by conspiracy type:")
        for ct_type in conspiracy_counts.index:
            total = conspiracy_counts[ct_type]
            support = support_by_type.get(ct_type, 0)
            deny = deny_by_type.get(ct_type, 0)
            support_rate = (support / total) * 100 if total > 0 else 0
            print(f"  {ct_type}: {support_rate:.1f}% support rate ({support} support, {deny} deny)")
    
    # 6. Cross-pattern analysis
    print("\n\n6. CROSS-INTERROGATIVE PATTERNS:")
    print("-" * 60)
    
    # Find accounts that spread multiple conspiracy types
    multi_conspiracy_patterns = []
    for mention, count in mention_counts.most_common(5):
        # Check which conspiracy types this account is associated with
        mention_tweets = df[df['tweet'].str.contains(f'@{mention}', case=False, na=False)]
        ct_types = mention_tweets['conspiracy_theory'].unique()
        if len(ct_types) > 1:
            multi_conspiracy_patterns.append({
                'account': f'@{mention}',
                'conspiracy_types': list(ct_types),
                'tweet_count': len(mention_tweets)
            })
    
    print("\nAccounts spreading multiple conspiracy types:")
    for pattern in multi_conspiracy_patterns:
        print(f"  {pattern['account']}: {', '.join(pattern['conspiracy_types'])} ({pattern['tweet_count']} tweets)")
    
    # 7. Summary insights
    summary = {
        'total_tweets': len(df),
        'conspiracy_types': len(conspiracy_counts),
        'most_common_type': conspiracy_counts.index[0],
        'most_mentioned_accounts': [f"@{acc}" for acc, _ in mention_counts.most_common(5)],
        'top_hashtags': [f"#{tag}" for tag, _ in hashtag_counts.most_common(5)],
        'support_rate': (len(df[df['label'] == 'support']) / len(df) * 100) if 'label' in df.columns else None,
        'multi_conspiracy_spreaders': len(multi_conspiracy_patterns)
    }
    
    print("\n\n7. SUMMARY OF REAL FINDINGS:")
    print("=" * 80)
    print(f"📊 Analyzed {summary['total_tweets']} real tweets")
    print(f"📈 Found {summary['conspiracy_types']} different conspiracy types")
    print(f"🔝 Most common: {summary['most_common_type']}")
    print(f"👥 Key influencers: {', '.join(summary['most_mentioned_accounts'][:3])}")
    print(f"#️⃣ Top hashtags: {', '.join(summary['top_hashtags'][:3])}")
    if summary['support_rate']:
        print(f"📊 Overall support rate: {summary['support_rate']:.1f}%")
    print(f"🔗 Multi-conspiracy spreaders: {summary['multi_conspiracy_spreaders']} accounts")
    
    # Save real results
    with open('real_tweet_analysis_results.json', 'w') as f:
        json.dump({
            'analysis_date': datetime.now().isoformat(),
            'dataset_stats': {
                'total_tweets': len(df),
                'columns': list(df.columns),
                'conspiracy_distribution': conspiracy_counts.to_dict(),
                'label_distribution': label_counts.to_dict() if 'label' in df.columns else None
            },
            'influencer_analysis': {
                'top_mentions': dict(mention_counts.most_common(20)),
                'top_hashtags': dict(hashtag_counts.most_common(20))
            },
            'narrative_patterns': {
                'multi_conspiracy_accounts': multi_conspiracy_patterns
            },
            'summary': summary
        }, f, indent=2)
    
    print("\n✓ Real analysis results saved to real_tweet_analysis_results.json")
    
    return summary

if __name__ == "__main__":
    analyze_real_tweets()