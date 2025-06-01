"""
Test script for the prepare_corpus_from_directory tool.
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path to ensure imports work
sys.path.append(str(Path(__file__).parent.parent))

from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs

async def create_test_files(test_dir):
    """Create sample text files for testing."""
    # Create a few test files
    file1 = test_dir / "document1.txt"
    file2 = test_dir / "document2.txt"
    file3 = test_dir / "document3.txt"
    
    with open(file1, 'w', encoding='utf-8') as f:
        f.write("This is the content of document 1. It contains some sample text for testing.")
    
    with open(file2, 'w', encoding='utf-8') as f:
        f.write("Document 2 has different content. This is used to verify corpus creation.")
    
    with open(file3, 'w', encoding='utf-8') as f:
        f.write("The third document completes our test set. Three documents should be enough for testing.")
    
    return [file1, file2, file3]

async def test_prepare_corpus_with_corpus_name():
    """Test prepare_corpus_from_directory with explicit corpus name."""
    with tempfile.TemporaryDirectory() as temp_input_dir:
        temp_input_path = Path(temp_input_dir)
        # Create test output dir
        temp_output_path = Path(__file__).parent / "test_output"
        temp_output_path.mkdir(exist_ok=True)
        
        # Create test files
        test_files = await create_test_files(temp_input_path)
        print(f"Created {len(test_files)} test files in {temp_input_dir}")
        
        # Run the tool with explicit corpus name
        inputs = PrepareCorpusInputs(
            input_directory_path=str(temp_input_path),
            output_directory_path=str(temp_output_path),
            target_corpus_name="TestCorpusExplicit"
        )
        
        result = await prepare_corpus_from_directory(inputs)
        print(f"\nTest with explicit corpus name result:\n{result}")
        
        # Verify output file exists
        corpus_json_path = Path(result.corpus_json_path) if result.corpus_json_path else None
        if corpus_json_path and corpus_json_path.exists():
            print(f"Corpus file created successfully at: {corpus_json_path}")
            print(f"First few lines of corpus file:")
            with open(corpus_json_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 3:  # Show only first 3 lines
                        print(f"  {line.strip()}")
        else:
            print(f"Failed to create corpus file")

async def test_prepare_corpus_without_corpus_name():
    """Test prepare_corpus_from_directory without explicit corpus name."""
    with tempfile.TemporaryDirectory() as temp_input_dir:
        temp_input_path = Path(temp_input_dir)
        # Create test output dir with a meaningful name (for derived corpus name)
        temp_output_path = Path(__file__).parent / "derived_name_test"
        temp_output_path.mkdir(exist_ok=True)
        
        # Create test files
        test_files = await create_test_files(temp_input_path)
        print(f"Created {len(test_files)} test files in {temp_input_dir}")
        
        # Run the tool without explicit corpus name (should derive from output dir)
        inputs = PrepareCorpusInputs(
            input_directory_path=str(temp_input_path),
            output_directory_path=str(temp_output_path),
            # target_corpus_name intentionally not provided
        )
        
        result = await prepare_corpus_from_directory(inputs)
        print(f"\nTest with derived corpus name result:\n{result}")
        
        # Verify output file exists
        corpus_json_path = Path(result.corpus_json_path) if result.corpus_json_path else None
        if corpus_json_path and corpus_json_path.exists():
            print(f"Corpus file created successfully at: {corpus_json_path}")
            print(f"First few lines of corpus file:")
            with open(corpus_json_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 3:  # Show only first 3 lines
                        print(f"  {line.strip()}")
        else:
            print(f"Failed to create corpus file")

async def test_prepare_corpus_error_handling():
    """Test prepare_corpus_from_directory error handling."""
    # Test with non-existent input directory
    non_existent_dir = Path("/path/does/not/exist")
    temp_output_path = Path(__file__).parent / "error_test_output"
    temp_output_path.mkdir(exist_ok=True)
    
    inputs = PrepareCorpusInputs(
        input_directory_path=str(non_existent_dir),
        output_directory_path=str(temp_output_path),
        target_corpus_name="ErrorTest"
    )
    
    result = await prepare_corpus_from_directory(inputs)
    print(f"\nTest with non-existent input directory result:\n{result}")
    
    # Test with input directory that has no .txt files
    with tempfile.TemporaryDirectory() as temp_input_dir:
        temp_input_path = Path(temp_input_dir)
        # Create a non-txt file
        with open(temp_input_path / "not_a_text_file.pdf", 'w') as f:
            f.write("This won't be processed")
        
        inputs = PrepareCorpusInputs(
            input_directory_path=str(temp_input_path),
            output_directory_path=str(temp_output_path),
            target_corpus_name="NoTxtFilesTest"
        )
        
        result = await prepare_corpus_from_directory(inputs)
        print(f"\nTest with no .txt files result:\n{result}")

async def main():
    """Run all tests."""
    print("Running tests for prepare_corpus_from_directory...")
    
    await test_prepare_corpus_with_corpus_name()
    await test_prepare_corpus_without_corpus_name()
    await test_prepare_corpus_error_handling()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
