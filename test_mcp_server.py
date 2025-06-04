"""
Test script for DIGIMON MCP Server
"""

import asyncio
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_mcp_connection(host='127.0.0.1', port=8765):
    """Test basic connection to MCP server"""
    try:
        reader, writer = await asyncio.open_connection(host, port)
        logger.info(f"Successfully connected to MCP server at {host}:{port}")
        
        # Test Corpus.Prepare tool
        logger.info("\n=== Testing Corpus.Prepare ===")
        request = {
            "id": "test-001",
            "tool_name": "Corpus.Prepare",
            "params": {
                "input_directory_path": str(Path("Data/MySampleTexts")),
                "output_directory_path": str(Path("Data/MySampleTexts_Test")),
                "target_corpus_name": "test_corpus"
            },
            "context": {},
            "session_id": "test-session",
            "timestamp": "2025-01-06T12:00:00"
        }
        
        # Send request
        request_data = json.dumps(request).encode()
        writer.write(len(request_data).to_bytes(4, 'big'))
        writer.write(request_data)
        await writer.drain()
        
        # Read response
        length_bytes = await reader.readexactly(4)
        response_length = int.from_bytes(length_bytes, 'big')
        response_data = await reader.readexactly(response_length)
        response = json.loads(response_data.decode())
        
        logger.info(f"Corpus.Prepare response: {json.dumps(response, indent=2)}")
        
        # Test Graph.Build tool
        logger.info("\n=== Testing Graph.Build ===")
        request = {
            "id": "test-002",
            "tool_name": "Graph.Build",
            "params": {
                "target_dataset_name": "MySampleTexts_Test",
                "force_rebuild": True
            },
            "context": {},
            "session_id": "test-session",
            "timestamp": "2025-01-06T12:01:00"
        }
        
        request_data = json.dumps(request).encode()
        writer.write(len(request_data).to_bytes(4, 'big'))
        writer.write(request_data)
        await writer.drain()
        
        length_bytes = await reader.readexactly(4)
        response_length = int.from_bytes(length_bytes, 'big')
        response_data = await reader.readexactly(response_length)
        response = json.loads(response_data.decode())
        
        logger.info(f"Graph.Build response: {json.dumps(response, indent=2)}")
        
        # Test Entity.VDBSearch tool
        logger.info("\n=== Testing Entity.VDBSearch ===")
        request = {
            "id": "test-003",
            "tool_name": "Entity.VDBSearch",
            "params": {
                "vdb_reference_id": "MySampleTexts_Test_entity_vdb",
                "query_text": "revolution",
                "top_k_results": 5,
                "dataset_name": "MySampleTexts_Test"
            },
            "context": {},
            "session_id": "test-session",
            "timestamp": "2025-01-06T12:02:00"
        }
        
        request_data = json.dumps(request).encode()
        writer.write(len(request_data).to_bytes(4, 'big'))
        writer.write(request_data)
        await writer.drain()
        
        length_bytes = await reader.readexactly(4)
        response_length = int.from_bytes(length_bytes, 'big')
        response_data = await reader.readexactly(response_length)
        response = json.loads(response_data.decode())
        
        logger.info(f"Entity.VDBSearch response: {json.dumps(response, indent=2)}")
        
        # Test Answer.Generate tool
        logger.info("\n=== Testing Answer.Generate ===")
        request = {
            "id": "test-004",
            "tool_name": "Answer.Generate",
            "params": {
                "query": "What were the main causes of the French Revolution?",
                "context": "The French Revolution was caused by financial crisis, social inequality, and Enlightenment ideas.",
                "response_type": "default"
            },
            "context": {},
            "session_id": "test-session",
            "timestamp": "2025-01-06T12:03:00"
        }
        
        request_data = json.dumps(request).encode()
        writer.write(len(request_data).to_bytes(4, 'big'))
        writer.write(request_data)
        await writer.drain()
        
        length_bytes = await reader.readexactly(4)
        response_length = int.from_bytes(length_bytes, 'big')
        response_data = await reader.readexactly(response_length)
        response = json.loads(response_data.decode())
        
        logger.info(f"Answer.Generate response: {json.dumps(response, indent=2)}")
        
        # Close connection
        writer.close()
        await writer.wait_closed()
        logger.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


async def main():
    """Run tests"""
    logger.info("Starting MCP server tests...")
    
    # Give server time to start if running concurrently
    await asyncio.sleep(1)
    
    await test_mcp_connection()


if __name__ == "__main__":
    asyncio.run(main())