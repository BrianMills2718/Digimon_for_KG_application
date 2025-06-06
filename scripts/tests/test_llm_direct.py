#!/usr/bin/env python3
"""Test LLM directly to ensure it's working"""

import asyncio
from Option.Config2 import Config
from Core.Provider.LiteLLMProvider import LiteLLMProvider

async def test_llm():
    # Load config
    config = Config.default()
    
    # Create LLM provider
    llm = LiteLLMProvider(config.llm)
    print(f"Testing LLM: {config.llm.model}")
    
    # Simple test message
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Return a JSON array."},
        {"role": "user", "content": 'Return this JSON array: ["step 1", "step 2", "step 3"]'}
    ]
    
    try:
        print("Calling LLM...")
        response = await llm.acompletion(messages=messages, temperature=0.0)
        
        if response and hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            print(f"LLM Response: {content}")
        else:
            print("Invalid response structure")
            
    except Exception as e:
        print(f"Error calling LLM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llm())