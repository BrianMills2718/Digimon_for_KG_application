"""Tests for social media analysis tools"""

import asyncio
import json
import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock, patch, AsyncMock

from Core.AgentTools.social_media_dataset_tools import (
    ingest_covid_conspiracy_dataset, 
    DatasetIngestionInput,
    DatasetIngestionOutput,
    SocialMediaDatasetIngestor
)
from Core.AgentTools.automated_interrogative_planner import (
    generate_interrogative_analysis_plans,
    AutoInterrogativePlanInput,
    AutoInterrogativePlanOutput,
    AnalysisScenario,
    InterrogativeView,
    AutomatedInterrogativePlanner
)


class TestSocialMediaDatasetIngestion:
    """Test social media dataset ingestion functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_dataset(self):
        """Mock Hugging Face dataset"""
        class MockDataset:
            def __init__(self):
                self.data = [
                    {
                        'tweet': 'The COVID vaccine is part of a population control agenda! #VaccineControl',
                        'conspiracy_theory': 'CT_6',
                        'label': 'support'
                    },
                    {
                        'tweet': 'Vaccines are safe and effective. Get vaccinated! #COVID19',
                        'conspiracy_theory': 'CT_6',
                        'label': 'deny'
                    },
                    {
                        'tweet': 'New COVID variants detected in several countries #COVIDUpdate',
                        'conspiracy_theory': 'CT_6',
                        'label': 'neutral'
                    }
                ]
            
            def to_pandas(self):
                import pandas as pd
                return pd.DataFrame(self.data)
        
        return MockDataset()
    
    def test_dataset_ingestion_tool_exists(self):
        """Test that dataset ingestion tool function exists"""
        assert callable(ingest_covid_conspiracy_dataset)
        assert asyncio.iscoroutinefunction(ingest_covid_conspiracy_dataset)
    
    @pytest.mark.asyncio
    @patch('Core.AgentTools.social_media_dataset_tools.load_dataset')
    async def test_dataset_ingestion_success(self, mock_load_dataset, mock_dataset, temp_dir):
        """Test successful dataset ingestion"""
        # Setup mock
        mock_load_dataset.return_value = mock_dataset
        
        # Create input
        input_data = DatasetIngestionInput(
            dataset_name="webimmunization/COVID-19-conspiracy-theories-tweets",
            split="train",
            output_path=str(Path(temp_dir) / "covid_dataset"),
            max_rows=3
        )
        
        # Execute
        output = await ingest_covid_conspiracy_dataset(input_data)
        
        # Verify output
        assert output.success is True
        assert output.total_rows == 3
        assert 'columns' in output.schema_info
        assert 'conspiracy_types' in output.schema_info
        assert 'label_distribution' in output.schema_info
        assert len(output.sample_data) > 0
        
        # Check files were created
        json_path = Path(temp_dir) / "covid_dataset.json"
        csv_path = Path(temp_dir) / "covid_dataset.csv"
        assert json_path.exists()
        assert csv_path.exists()
        
        # Verify JSON content
        with open(json_path) as f:
            data = json.load(f)
            assert len(data) == 3
            assert all('tweet_id' in item for item in data)
            assert all('hashtags' in item for item in data)
            assert all('conspiracy_theory_name' in item for item in data)
    
    def test_dataset_ingestion_extracts_features(self, temp_dir):
        """Test that ingestion extracts additional features"""
        ingestor = SocialMediaDatasetIngestor()
        
        # Create mock dataset with pandas
        import pandas as pd
        df = pd.DataFrame({
            'tweet': [
                'Check out #COVID19 vaccine info from @WHO',
                'The #vaccine is dangerous! @user123 knows the truth #conspiracy',
                'Getting my shot today'
            ],
            'conspiracy_theory': ['CT_2', 'CT_6', 'CT_6'],
            'label': ['neutral', 'support', 'deny']
        })
        
        # Mock the load_dataset function
        with patch('Core.AgentTools.social_media_dataset_tools.load_dataset') as mock_load:
            mock_load.return_value.to_pandas.return_value = df
            
            input_data = DatasetIngestionInput(
                dataset_name="test_dataset",
                output_path=str(Path(temp_dir) / "test_output")
            )
            
            output = ingestor.ingest_covid_conspiracy_dataset(input_data)
            
            # Load and verify extracted features
            json_path = Path(temp_dir) / "test_output.json"
            with open(json_path) as f:
                data = json.load(f)
                
                # Check hashtag extraction
                assert data[0]['hashtags'] == ['#COVID19']  # Only has #COVID19
                assert data[1]['hashtags'] == ['#vaccine', '#conspiracy']
                assert data[2]['hashtags'] == []
                
                # Check mention extraction
                assert data[0]['mentions'] == ['@WHO']
                assert data[1]['mentions'] == ['@user123']
                assert data[2]['mentions'] == []
                
                # Check computed features
                assert all('tweet_length' in item for item in data)
                assert all('hashtag_count' in item for item in data)
                assert all('mention_count' in item for item in data)


class TestAutomatedInterrogativePlanner:
    """Test automated interrogative planning functionality"""
    
    def test_planner_tool_exists(self):
        """Test that planner tool function exists"""
        assert callable(generate_interrogative_analysis_plans)
        assert asyncio.iscoroutinefunction(generate_interrogative_analysis_plans)
    
    def test_generate_interrogative_views(self):
        """Test interrogative view generation"""
        from Core.AgentTools.automated_interrogative_planner import AutomatedInterrogativePlanner
        
        planner = AutomatedInterrogativePlanner()
        views = planner.generate_interrogative_views("conspiracy theories on social media", num_views=2)
        
        assert len(views) == 2
        assert all(isinstance(view, InterrogativeView) for view in views)
        assert all(view.interrogative in ["Who", "What", "When", "Where", "Why", "How"] for view in views)
        assert all(len(view.entities) > 0 for view in views)
        assert all(len(view.relationships) > 0 for view in views)
        assert all(len(view.analysis_goals) > 0 for view in views)
    
    @pytest.mark.asyncio
    async def test_planner_generates_scenarios(self):
        """Test that planner generates diverse analysis scenarios"""
        input_data = AutoInterrogativePlanInput(
            domain="COVID-19 conspiracy theories on Twitter",
            dataset_info={
                "rows": 6591,
                "columns": ["tweet", "conspiracy_theory", "label"],
                "conspiracy_types": ["CT_1", "CT_2", "CT_3", "CT_4", "CT_5", "CT_6"]
            },
            num_scenarios=3,
            complexity_range=["Simple", "Medium"]
        )
        
        output = await generate_interrogative_analysis_plans(input_data)
        
        assert output.success is True
        assert len(output.scenarios) <= 3
        assert all(isinstance(s, AnalysisScenario) for s in output.scenarios)
        assert all(s.complexity_level in ["Simple", "Medium"] for s in output.scenarios)
        assert len(output.execution_order) == len(output.scenarios)
        assert all(title in output.estimated_complexity for title in output.execution_order)
    
    @pytest.mark.asyncio
    async def test_scenario_structure(self):
        """Test that generated scenarios have proper structure"""
        input_data = AutoInterrogativePlanInput(
            domain="conspiracy theories",
            dataset_info={},
            num_scenarios=1
        )
        
        output = await generate_interrogative_analysis_plans(input_data)
        
        if output.success and len(output.scenarios) > 0:
            scenario = output.scenarios[0]
            
            # Check scenario structure
            assert hasattr(scenario, 'title')
            assert hasattr(scenario, 'research_question')
            assert hasattr(scenario, 'interrogative_views')
            assert hasattr(scenario, 'analysis_pipeline')
            assert hasattr(scenario, 'expected_insights')
            assert hasattr(scenario, 'complexity_level')
            
            # Check interrogative views
            assert len(scenario.interrogative_views) > 0
            for view in scenario.interrogative_views:
                assert view.interrogative in ["Who", "What", "When", "Where", "Why", "How"]
                assert len(view.focus) > 0
                assert len(view.description) > 0
            
            # Check analysis pipeline
            assert len(scenario.analysis_pipeline) > 0
            assert all(isinstance(step, str) for step in scenario.analysis_pipeline)
    
    @pytest.mark.asyncio
    async def test_complexity_filtering(self):
        """Test that planner respects complexity filtering"""
        # Request only complex scenarios
        input_data = AutoInterrogativePlanInput(
            domain="social media analysis",
            dataset_info={},
            num_scenarios=5,
            complexity_range=["Complex"]
        )
        
        output = await generate_interrogative_analysis_plans(input_data)
        
        if output.success:
            assert all(s.complexity_level == "Complex" for s in output.scenarios)
    
    def test_entities_and_relationships_mapping(self):
        """Test that interrogatives map to appropriate entities and relationships"""
        from Core.AgentTools.automated_interrogative_planner import AutomatedInterrogativePlanner
        
        planner = AutomatedInterrogativePlanner()
        
        # Test Who interrogative
        entities, relationships = planner._get_entities_relationships("Who")
        assert "User" in entities
        assert "Influencer" in entities
        assert "FOLLOWS" in relationships
        assert "MENTIONS" in relationships
        
        # Test What interrogative
        entities, relationships = planner._get_entities_relationships("What")
        assert "Tweet" in entities
        assert "Topic" in entities
        assert "DISCUSSES" in relationships
        assert "CONTAINS" in relationships
        
        # Test How interrogative
        entities, relationships = planner._get_entities_relationships("How")
        assert "Process" in entities
        assert "Mechanism" in entities
        assert "ENABLES" in relationships
        assert "FACILITATES" in relationships


class TestIntegration:
    """Test integration between dataset ingestion and planning"""
    
    @pytest.fixture
    def ingested_dataset_info(self):
        """Mock ingested dataset information"""
        return {
            "total_rows": 6591,
            "columns": ["tweet", "conspiracy_theory", "label", "hashtags", "mentions"],
            "conspiracy_types": {
                "CT_1": 1000,
                "CT_2": 1200,
                "CT_3": 900,
                "CT_4": 1100,
                "CT_5": 1000,
                "CT_6": 1391
            },
            "label_distribution": {
                "support": 3000,
                "deny": 2000,
                "neutral": 1591
            }
        }
    
    @pytest.mark.asyncio
    async def test_planner_uses_dataset_info(self, ingested_dataset_info):
        """Test that planner can use ingested dataset information"""
        input_data = AutoInterrogativePlanInput(
            domain="COVID-19 conspiracy theories from ingested dataset",
            dataset_info=ingested_dataset_info,
            num_scenarios=2,
            focus_areas=["influence networks", "narrative evolution"]
        )
        
        output = await generate_interrogative_analysis_plans(input_data)
        
        assert output.success is True
        assert len(output.scenarios) > 0
        
        # Check that scenarios reference the dataset info
        for scenario in output.scenarios:
            # At least one scenario should focus on the requested areas
            if "influence" in scenario.title.lower() or "narrative" in scenario.title.lower():
                assert True
                break
        else:
            pytest.fail("No scenarios focused on requested areas")


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test async operations if tools support them"""
    
    async def test_async_compatibility(self):
        """Test that tools can be used in async context"""
        with patch('Core.AgentTools.social_media_dataset_tools.load_dataset') as mock_load:
            mock_load.return_value.to_pandas.return_value = MagicMock()
            
            input_data = DatasetIngestionInput(
                dataset_name="test",
                output_path="/tmp/test"
            )
            
            # Tools are already async
            result = await ingest_covid_conspiracy_dataset(input_data)
            assert isinstance(result, DatasetIngestionOutput)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])