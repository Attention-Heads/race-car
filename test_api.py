#!/usr/bin/env python3
"""
Quick test script to verify the API server is running and responding correctly.
Run this before running the game to make sure the API is working.
"""

import asyncio
import aiohttp
import json

async def test_api():
    """Test the API server with a sample request."""
    
    api_url = "http://localhost:8000/predict"
    
    # Sample request data that matches what the game sends
    test_request = {
        'did_crash': False,
        'elapsed_ticks': 1,
        'distance': 10.5,
        'velocity': {'x': 10.0, 'y': 0.0},
        'sensors': {
            'front': 100.0,
            'left_side': 50.0,
            'right_side': 50.0,
            'back': 25.0
        }
    }
    
    print(f"Testing API at: {api_url}")
    print(f"Sending test request: {json.dumps(test_request, indent=2)}")
    print("-" * 50)
    
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(api_url, json=test_request) as response:
                
                print(f"Response status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"Response data: {json.dumps(result, indent=2)}")
                    
                    # Check if response has the expected format
                    if 'actions' in result and result['actions']:
                        print(f"✅ API is working! First action: {result['actions'][0]}")
                        return True
                    elif 'action' in result:
                        print(f"✅ API is working! Action: {result['action']}")
                        return True
                    else:
                        print("❌ API response doesn't contain 'action' or 'actions' field")
                        return False
                else:
                    response_text = await response.text()
                    print(f"❌ API call failed with status {response.status}")
                    print(f"Response body: {response_text}")
                    return False
                    
    except aiohttp.ClientConnectorError as e:
        print(f"❌ Cannot connect to API server: {e}")
        print("Make sure to start the API server first by running: python api.py")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("Testing Race Car API Server...")
    print("=" * 50)
    
    success = asyncio.run(test_api())
    
    print("=" * 50)
    if success:
        print("✅ API test passed! You can now run the game.")
    else:
        print("❌ API test failed. Please check the API server.")
        print("\nTo start the API server, run:")
        print("  python api.py")
