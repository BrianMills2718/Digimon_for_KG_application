"""
Test script for the corpus preparation tool in isolation.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs, PrepareCorpusOutputs
from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
from Core.Common.Logger import logger

async def test_corpus_preparation_tool():
    """
    Test the prepare_corpus_from_directory tool with actual text files.
    """
    logger.info("Starting test_corpus_preparation_tool...")
    
    try:
        # Define paths
        project_root = Path(__file__).resolve().parent.parent
        input_txt_dir = str(project_root / "Data" / "MySampleTexts")
        output_dir_name = "MySampleTexts_Corpus_Test"
        output_dir = str(project_root / "results" / output_dir_name)
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Log the input directory content
        input_files = list(Path(input_txt_dir).glob("*.txt"))
        logger.info(f"Found {len(input_files)} text files in {input_txt_dir}")
        for file in input_files:
            logger.info(f"  - {file.name} ({file.stat().st_size} bytes)")
        
        # Prepare the input model
        corpus_inputs = PrepareCorpusInputs(
            input_directory_path=input_txt_dir,
            output_directory_path=output_dir,
            target_corpus_name=output_dir_name
        )
        
        # Call the tool function
        logger.info(f"Calling prepare_corpus_from_directory with inputs: {corpus_inputs}")
        result = await prepare_corpus_from_directory(corpus_inputs)
        
        # Log the result
        logger.info(f"Tool execution completed with status: {result.status}")
        logger.info(f"Message: {result.message}")
        logger.info(f"Documents processed: {result.document_count}")
        logger.info(f"Corpus JSON path: {result.corpus_json_path}")
        
        # Verify corpus file exists
        corpus_path = Path(result.corpus_json_path)
        if corpus_path.exists():
            logger.info(f"Corpus file exists at {corpus_path} ({corpus_path.stat().st_size} bytes)")
            
            # Show the first few lines of the corpus
            with open(corpus_path, 'r', encoding='utf-8') as f:
                lines = [next(f) for _ in range(min(3, result.document_count))]
                logger.info(f"First few lines of the corpus:\n{lines}")
        else:
            logger.error(f"Corpus file not found at {corpus_path}")
        
        logger.info("test_corpus_preparation_tool completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error in test_corpus_preparation_tool: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

async def main():
    """Run the test"""
    result = await test_corpus_preparation_tool()
    return result

if __name__ == "__main__":
    asyncio.run(main())
