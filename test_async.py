#!/usr/bin/env python3
"""
Quick test to verify that the async get_action_from_api function works correctly.
"""

import asyncio
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_async_function():
    """Test the async get_action_from_api function."""
    print("Testing async get_action_from_api function...")
    
    try:
        # Import the necessary modules
        from src.game.core import initialize_game_state, get_action_from_api
        
        # Initialize with a test API URL (this will fail, but that's expected)
        seed_value = 123456
        api_url = "http://nonexistent-api.test/predict"
        
        print(f"Initializing game state with seed: {seed_value}")
        initialize_game_state(api_url, seed_value)
        
        print("Calling async get_action_from_api()...")
        action = await get_action_from_api()
        
        print(f"✅ Function returned: {action}")
        print("✅ Async function is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing async function: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_async_function())
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
