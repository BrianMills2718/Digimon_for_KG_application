#!/usr/bin/env python3
"""Download COVID-19 conspiracy theory tweets dataset from Hugging Face"""

import sys
from pathlib import Path

try:
    import pandas as pd
    from datasets import load_dataset
except ImportError:
    print("Error: Required libraries not installed.")
    print("Please run: pip install datasets pandas")
    sys.exit(1)

def download_covid_conspiracy_dataset():
    """Download and save the COVID conspiracy dataset as CSV"""
    print("Downloading COVID-19 conspiracy theories tweets dataset from Hugging Face...")
    
    try:
        # Load dataset
        dataset = load_dataset("webimmunization/COVID-19-conspiracy-theories-tweets", split="train")
        
        # Convert to pandas
        df = dataset.to_pandas()
        
        # Save as CSV
        output_path = Path("COVID-19-conspiracy-theories-tweets.csv")
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Dataset saved to: {output_path}")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {', '.join(df.columns)}")
        print(f"\n  Conspiracy types:")
        for ct in sorted(df['conspiracy_theory'].unique()):
            count = len(df[df['conspiracy_theory'] == ct])
            print(f"    {ct}: {count} tweets")
        
        print(f"\n  Label distribution:")
        for label, count in df['label'].value_counts().items():
            print(f"    {label}: {count} tweets")
        
        # Also save a smaller sample for testing
        sample_path = Path("COVID-19-conspiracy-sample.csv")
        df.head(100).to_csv(sample_path, index=False)
        print(f"\n✓ Sample dataset (100 rows) saved to: {sample_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    success = download_covid_conspiracy_dataset()
    if not success:
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have the required libraries: pip install datasets pandas")
        print("3. Try updating datasets: pip install -U datasets")
        sys.exit(1)