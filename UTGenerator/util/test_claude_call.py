#!/usr/bin/env python3
"""
Simple script to test Anthropic Claude-3 API
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api_requests import call_claude_model, create_anthropic_config, request_anthropic_engine

def test_claude_simple_call():
    """Test using the simple call_claude_model function"""
    prompt = "Please explain what decorators are in Python and provide a simple example."
    
    print("\n===== Using call_claude_model function =====")
    response = call_claude_model(
        prompt=prompt,
        model="claude-3-sonnet-20240229",
        system_message="You are a professional Python tutorial writer who answers questions clearly and concisely.",
        max_tokens=800,
        temperature=0.7
    )
    
    print(f"Claude response:\n{response}")

def test_claude_detailed_call():
    """Test using the complete API configuration"""
    prompt = "Please explain what decorators are in Python and provide a simple example."
    
    print("\n===== Using complete API call =====")
    config = create_anthropic_config(
        message=prompt,
        max_tokens=800,
        temperature=0.7,
        system_message="You are a professional Python tutorial writer who answers questions clearly and concisely.",
        model="claude-3-sonnet-20240229"
    )
    
    response, usage = request_anthropic_engine(config)
    
    print(f"Claude response:\n{response}")
    print(f"\nUsage statistics:")
    print(f"- Input tokens: {usage['input_tokens']}")
    print(f"- Output tokens: {usage['output_tokens']}")
    print(f"- Total tokens: {usage['input_tokens'] + usage['output_tokens']}")

if __name__ == "__main__":
    # Ensure API key from .env file is loaded
    load_dotenv()
    
    # Check if API key exists
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not found. Please set it in the .env file.")
        exit(1)
    
    # Run tests
    test_claude_simple_call()
    test_claude_detailed_call() 