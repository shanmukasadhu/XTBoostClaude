#!/usr/bin/env python3
"""
Simple script to test Anthropic API
"""

import os
import sys
import anthropic
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api_requests import call_claude_model, create_anthropic_config, request_anthropic_engine

# Load environment variables
load_dotenv()

def test_direct_anthropic_api():
    """Test API directly using Anthropic Python client"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not found. Please set it in the .env file.")
        sys.exit(1)
    
    print("\n===== Using Anthropic Python client directly =====")
    client = anthropic.Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        temperature=0.7,
        system="You are a professional Python tutorial writer who answers questions clearly and concisely.",
        messages=[
            {"role": "user", "content": "Please explain what decorators are in Python and provide a simple example."}
        ]
    )
    
    print(f"Claude response:\n{message.content[0].text}")
    print(f"\nUsage statistics:")
    print(f"- Input tokens: {message.usage.input_tokens}")
    print(f"- Output tokens: {message.usage.output_tokens}")
    print(f"- Total tokens: {message.usage.input_tokens + message.usage.output_tokens}")

def test_our_wrapper_functions():
    """Test our own API wrapper functions"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not found. Please set it in the .env file.")
        sys.exit(1)
    
    prompt = "Please explain recursive functions in simple language and provide an example of a recursive function that calculates Fibonacci numbers."
    
    # Using the simplified function call
    print("\n===== Using call_claude_model function =====")
    response = call_claude_model(
        prompt=prompt,
        model="claude-3-sonnet-20240229",
        system_message="You are a programming teaching expert who explains complex concepts in simple, easy-to-understand language.",
        max_tokens=800,
        temperature=0.7
    )
    
    print(f"Claude response:\n{response}")
    
    # Using the full API call
    print("\n===== Using create_anthropic_config and request_anthropic_engine functions =====")
    config = create_anthropic_config(
        message=prompt,
        max_tokens=800,
        temperature=0.7,
        system_message="You are a programming teaching expert who explains complex concepts in simple, easy-to-understand language.",
        model="claude-3-sonnet-20240229"
    )
    
    response, usage = request_anthropic_engine(config)
    
    print(f"Claude response:\n{response}")
    print(f"\nUsage statistics:")
    print(f"- Input tokens: {usage['input_tokens']}")
    print(f"- Output tokens: {usage['output_tokens']}")
    print(f"- Total tokens: {usage['input_tokens'] + usage['output_tokens']}")

if __name__ == "__main__":
    # Run tests
    test_direct_anthropic_api()
    test_our_wrapper_functions() 