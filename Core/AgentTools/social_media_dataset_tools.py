from typing import Dict, List, Any, Optional
import pandas as pd
from datasets import load_dataset
import json
from pathlib import Path
from pydantic import BaseModel, Field
from Core.AgentSchema.tool_contracts import BaseToolParams, BaseToolOutput
from Core.Common.Logger import logger

class DatasetIngestionInput(BaseToolParams):
    """Input for dataset ingestion tool"""
    dataset_name: str = Field(description="Hugging Face dataset name")
    split: str = Field(default="train", description="Dataset split to load")
    output_path: str = Field(description="Path to save processed dataset")
    max_rows: Optional[int] = Field(default=None, description="Maximum rows to process")

class DatasetIngestionOutput(BaseToolOutput):
    """Output from dataset ingestion"""
    success: bool = Field(description="Whether ingestion succeeded")
    dataset_path: str = Field(description="Path to saved dataset")
    total_rows: int = Field(description="Total rows processed")
    schema_info: Dict[str, Any] = Field(description="Dataset schema information")
    sample_data: List[Dict] = Field(description="Sample rows for validation")

class SocialMediaDatasetIngestor:
    """Tool for ingesting social media datasets from Hugging Face"""
    
    def __init__(self):
        self.logger = logger
    
    def ingest_covid_conspiracy_dataset(self, input_data: DatasetIngestionInput) -> DatasetIngestionOutput:
        """Ingest COVID conspiracy tweets dataset from Hugging Face"""
        try:
            self.logger.info(f"Loading dataset: {input_data.dataset_name}")
            
            # Load dataset from Hugging Face
            dataset = load_dataset(input_data.dataset_name, split=input_data.split)
            
            # Convert to pandas for easier processing
            df = dataset.to_pandas()
            
            # Limit rows if specified
            if input_data.max_rows:
                df = df.head(input_data.max_rows)
            
            # Add derived fields for analysis
            df['tweet_id'] = range(len(df))
            df['tweet_length'] = df['tweet'].str.len()
            df['hashtag_count'] = df['tweet'].str.count('#')
            df['mention_count'] = df['tweet'].str.count('@')
            
            # Extract hashtags and mentions
            df['hashtags'] = df['tweet'].str.findall(r'#\w+')
            df['mentions'] = df['tweet'].str.findall(r'@\w+')
            
            # Create conspiracy theory mapping
            ct_mapping = {
                'CT_1': 'Economic Instability Strategy',
                'CT_2': 'Public Misinformation Campaign', 
                'CT_3': 'Human-Made Bioweapon',
                'CT_4': 'Government Disinformation',
                'CT_5': 'Chinese Intentional Spread',
                'CT_6': 'Vaccine Population Control'
            }
            df['conspiracy_theory_name'] = df['conspiracy_theory'].map(ct_mapping)
            
            # Save processed dataset
            output_path = Path(input_data.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as both JSON and CSV for flexibility
            json_path = output_path.with_suffix('.json')
            csv_path = output_path.with_suffix('.csv')
            df.to_json(str(json_path), orient='records', indent=2)
            df.to_csv(str(csv_path), index=False)
            
            # Generate schema info
            schema_info = {
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'conspiracy_types': df['conspiracy_theory'].value_counts().to_dict(),
                'label_distribution': df['label'].value_counts().to_dict(),
                'tweet_length_stats': df['tweet_length'].describe().to_dict()
            }
            
            # Sample data for validation
            sample_data = df.head(5).to_dict('records')
            
            self.logger.info(f"Successfully ingested {len(df)} rows to {output_path}")
            
            return DatasetIngestionOutput(
                success=True,
                dataset_path=str(output_path),
                total_rows=len(df),
                schema_info=schema_info,
                sample_data=sample_data
            )
            
        except Exception as e:
            self.logger.error(f"Dataset ingestion failed: {str(e)}")
            return DatasetIngestionOutput(
                success=False,
                dataset_path="",
                total_rows=0,
                schema_info={},
                sample_data=[]
            )

# Tool function following DIGIMON pattern
async def ingest_covid_conspiracy_dataset(
    tool_input: DatasetIngestionInput,
    context: Optional[Any] = None  # GraphRAGContext would go here if needed
) -> DatasetIngestionOutput:
    """Ingest COVID-19 conspiracy theories tweet dataset from Hugging Face"""
    ingestor = SocialMediaDatasetIngestor()
    return ingestor.ingest_covid_conspiracy_dataset(tool_input)

# Entity extraction models for social media analysis
class SocialMediaEntity(BaseModel):
    """Base class for social media entities"""
    id: str = Field(description="Unique identifier")
    type: str = Field(description="Entity type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")

class TwitterUser(SocialMediaEntity):
    """Represents a Twitter user (inferred from tweet patterns)"""
    username: Optional[str] = Field(default=None, description="Username if extractable")
    tweet_count: int = Field(default=0, description="Number of tweets by this user")
    hashtag_usage: List[str] = Field(default_factory=list, description="Hashtags used")
    conspiracy_stance: Dict[str, str] = Field(default_factory=dict, description="Stance per conspiracy type")
    influence_score: float = Field(default=0.0, description="Computed influence score")

class Tweet(SocialMediaEntity):
    """Represents a tweet"""
    text: str = Field(description="Tweet text content")
    conspiracy_theory: str = Field(description="Conspiracy theory type")
    label: str = Field(description="Support/deny/neutral stance")
    hashtags: List[str] = Field(default_factory=list, description="Hashtags in tweet")
    mentions: List[str] = Field(default_factory=list, description="User mentions")
    toxicity_score: Optional[float] = Field(default=None, description="Computed toxicity score")
    engagement_score: Optional[float] = Field(default=None, description="Simulated engagement")

class ConspiracyTopic(SocialMediaEntity):
    """Represents a conspiracy theory topic"""
    name: str = Field(description="Topic name")
    description: str = Field(description="Topic description")
    support_count: int = Field(default=0, description="Number of supporting tweets")
    deny_count: int = Field(default=0, description="Number of denying tweets")
    neutral_count: int = Field(default=0, description="Number of neutral tweets")
    key_hashtags: List[str] = Field(default_factory=list, description="Associated hashtags")

# Export tools for registration
__all__ = [
    'ingest_covid_conspiracy_dataset',
    'SocialMediaDatasetIngestor', 
    'DatasetIngestionInput',
    'DatasetIngestionOutput',
    'TwitterUser',
    'Tweet', 
    'ConspiracyTopic'
]